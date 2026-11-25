"""INT8 ONNX Runtime inference engine with calibrated abstention gate and lazy Grad-CAM."""

from __future__ import annotations

import hashlib
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from scipy import ndimage, signal

from app.audit_db import ProvenanceLedger
from ml.watermark import SpreadSpectrumWatermarker

try:
    import torch
    from ml.gradcam import SpectrogramGradCAM
    from ml.model import SpectraGuardNet, compute_log_mel

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on slim serverless runtimes (e.g. Vercel)
    torch = None  # type: ignore[assignment]
    SpectrogramGradCAM = None  # type: ignore[assignment,misc]
    SpectraGuardNet = None  # type: ignore[assignment,misc]
    compute_log_mel = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


DEFAULT_WATERMARK_HEX = "53504543"  # "SPEC"
MIN_WATERMARK_SAMPLES = 4096
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=4)
def _mel_filterbank_numpy(
    sample_rate: int = 16000,
    n_fft: int = 512,
    n_mels: int = 64,
) -> np.ndarray:
    """Build HTK Mel filterbank matrix matching torchaudio.transforms.MelSpectrogram."""
    all_freqs = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    mel_min = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0)
    m_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up)).T.astype(np.float32)


def compute_log_mel_numpy(
    waveforms: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 64,
) -> np.ndarray:
    """Pure NumPy/SciPy 64-bin Log-Mel extractor matching torchaudio for torch-free runtimes."""
    x = np.asarray(waveforms, dtype=np.float32)
    if x.ndim == 2:
        x = x[0]
    pad = n_fft // 2
    x_pad = np.pad(x, (pad, pad), mode="reflect")
    win = signal.windows.hann(n_fft, sym=False).astype(np.float32)
    n_frames = 1 + (len(x_pad) - n_fft) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(x_pad, n_fft)[::hop_length][:n_frames]
    spec = np.fft.rfft(frames * win[None, :], axis=1)
    power = (spec.real**2 + spec.imag**2).T
    mel = _mel_filterbank_numpy(sample_rate, n_fft, n_mels) @ power
    log_mel = np.log(np.clip(mel, 1e-6, None)).astype(np.float32)
    return log_mel[None, None, :, :]


def _compute_numpy_saliency(log_mel_np: np.ndarray, spoof_probability: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Fast deterministic spectral-energy saliency fallback when PyTorch is unavailable."""
    spec = log_mel_np[0, 0].astype(np.float32)  # [64, T]
    freq_bins, time_frames = spec.shape

    # Contrast energy above per-bin temporal baseline, weighted toward upper-band spoof harmonics
    baseline = np.median(spec, axis=1, keepdims=True)
    contrast = np.maximum(0.0, spec - baseline)
    freq_weights = np.linspace(0.6, 1.6 if spoof_probability >= 0.5 else 1.0, freq_bins, dtype=np.float32)[:, None]
    raw_map = ndimage.gaussian_filter(contrast * freq_weights, sigma=(1.2, 1.5))

    map_min = float(raw_map.min())
    map_max = float(raw_map.max())
    if map_max - map_min > 1e-8:
        heatmap = ((raw_map - map_min) / (map_max - map_min)).astype(np.float32)
    else:
        heatmap = np.zeros((freq_bins, time_frames), dtype=np.float32)

    heatmap = np.clip(heatmap, 0.0, 1.0).astype(np.float32)
    peak_flat = int(np.argmax(heatmap))
    peak_freq_bin = int(peak_flat // time_frames)
    peak_time_frame = int(peak_flat % time_frames)
    high_freq_energy_ratio = float(heatmap[freq_bins // 2 :, :].sum() / (float(heatmap.sum()) + 1e-12))

    summary: dict[str, Any] = {
        "peak_freq_bin": peak_freq_bin,
        "peak_time_frame": peak_time_frame,
        "high_freq_energy_ratio": round(high_freq_energy_ratio, 4),
        "spoof_probability": round(float(spoof_probability), 4),
        "saliency_mode": "spectral_energy_fallback",
    }
    return heatmap, summary


def hex_to_bits(hex_str: str) -> list[int]:
    """Convert an 8-character hexadecimal string (32 bits) into a 32-element bit list."""
    cleaned = hex_str.strip().lower().removeprefix("0x")
    if len(cleaned) != 8:
        raise ValueError(f"watermark_hex must be exactly 8 hex characters (32 bits), got '{hex_str}'")
    val = int(cleaned, 16)
    return [(val >> (31 - i)) & 1 for i in range(32)]


def bits_to_hex(bits: list[int]) -> str:
    """Convert a 32-bit list into an 8-character lowercase hexadecimal string."""
    if len(bits) != 32:
        raise ValueError(f"Expected 32 bits, got {len(bits)}")
    val = 0
    for bit in bits:
        val = (val << 1) | (1 if bit else 0)
    return f"{val:08x}"


def apply_abstention_gate(spoof_probability: float) -> str:
    """Route calibrated spoof probability into AUTHENTIC, STEP_UP_OTP, or SPOOF_DETECTED."""
    if spoof_probability < 0.40:
        return "AUTHENTIC"
    if spoof_probability <= 0.60:
        return "STEP_UP_OTP"
    return "SPOOF_DETECTED"


def _resolve_artifact_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    candidate = PROJECT_ROOT / p
    return candidate if candidate.exists() else p


class SpectraGuardEngine:
    """Real-time audio provenance, INT8 ONNX spoof detection, and Grad-CAM attribution engine."""

    def __init__(
        self,
        onnx_path: str | Path = "artifacts/spectraguard_int8.onnx",
        pt_path: str | Path = "artifacts/spectraguard.pt",
        ledger: ProvenanceLedger | None = None,
    ) -> None:
        self.onnx_path = _resolve_artifact_path(onnx_path)
        self.pt_path = _resolve_artifact_path(pt_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"Missing INT8 ONNX artifact: {self.onnx_path}")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.onnx_provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.watermarker = SpreadSpectrumWatermarker(sample_rate=16000, chip_length=128, seed=42)
        self.ledger = ledger

        # Lazily initialized only when include_gradcam=True and PyTorch is available
        self._pt_model: Any = None
        self._gradcam: Any = None

    def _ensure_gradcam(self) -> Any:
        """Load PyTorch checkpoint and Grad-CAM hooks lazily when torch and .pt are available."""
        if not TORCH_AVAILABLE or not self.pt_path.exists():
            return None
        if self._gradcam is None:
            model = SpectraGuardNet(n_mels=64, dropout=0.1)
            state = torch.load(self.pt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self._pt_model = model
            self._gradcam = SpectrogramGradCAM(model)
        return self._gradcam

    def predict_onnx(self, log_mel_np: np.ndarray) -> float:
        """Execute a single INT8 ONNX forward pass and return calibrated spoof probability."""
        logits = self.session.run([self.output_name], {self.input_name: log_mel_np})[0]
        logit_val = float(logits[0, 0])
        return float(1.0 / (1.0 + np.exp(-logit_val)))

    def analyze(
        self,
        audio: np.ndarray,
        *,
        embed_watermark: bool = False,
        watermark_hex: str | None = None,
        include_gradcam: bool = False,
        log_to_ledger: bool = True,
    ) -> dict[str, Any]:
        """Run full active provenance extraction, INT8 ONNX inference, abstention gate, and audit logging."""
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("Audio must be a non-empty 1D array")
        arr = np.clip(arr, -1.0, 1.0).astype(np.float32)
        if len(arr) < MIN_WATERMARK_SAMPLES:
            arr = np.pad(arr, (0, MIN_WATERMARK_SAMPLES - len(arr)), mode="constant")

        embed_snr_db: float | None = None
        if embed_watermark:
            target_hex = watermark_hex or DEFAULT_WATERMARK_HEX
            payload_bits = hex_to_bits(target_hex)
            arr, embed_snr_db = self.watermarker.embed(arr, payload_bits)

        t_start = time.perf_counter()

        # 1. Active spread-spectrum watermark extraction
        recovered_bits, wm_conf = self.watermarker.extract(arr)
        recovered_hex = bits_to_hex(recovered_bits)

        # 2. Log-Mel feature extraction + INT8 ONNX Runtime inference
        if TORCH_AVAILABLE and compute_log_mel is not None:
            log_mel_tensor = compute_log_mel(arr, sample_rate=16000, n_fft=512, hop_length=256, n_mels=64)
            log_mel_np = log_mel_tensor.numpy().astype(np.float32)
        else:
            log_mel_tensor = None
            log_mel_np = compute_log_mel_numpy(arr, sample_rate=16000, n_fft=512, hop_length=256, n_mels=64)

        spoof_prob = self.predict_onnx(log_mel_np)

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        verdict = apply_abstention_gate(spoof_prob)
        audio_sha256 = hashlib.sha256(arr.tobytes()).hexdigest()

        result: dict[str, Any] = {
            "verdict": verdict,
            "spoof_probability": round(spoof_prob, 6),
            "watermark_hex": recovered_hex,
            "watermark_confidence": round(wm_conf, 4),
            "watermark_embedded": embed_watermark,
            "embed_snr_db": round(embed_snr_db, 2) if embed_snr_db is not None else None,
            "inference_latency_ms": round(latency_ms, 3),
            "audio_sha256": audio_sha256,
        }

        if include_gradcam:
            gradcam = self._ensure_gradcam()
            if gradcam is not None and log_mel_tensor is not None:
                heatmap, summary = gradcam.generate(log_mel_tensor)
            else:
                heatmap, summary = _compute_numpy_saliency(log_mel_np, spoof_prob)
            result["gradcam"] = {
                "heatmap": heatmap.tolist(),
                "summary": summary,
            }

        if log_to_ledger and self.ledger is not None:
            record = self.ledger.append_event(
                audio_sha256=audio_sha256,
                verdict=verdict,
                spoof_prob=spoof_prob,
                watermark_hex=recovered_hex,
                watermark_confidence=wm_conf,
                inference_latency_ms=latency_ms,
            )
            result["audit_id"] = record["id"]
            result["audit_row_hash"] = record["row_hash"]

        return result
