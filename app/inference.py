"""INT8 ONNX Runtime inference engine with calibrated abstention gate and lazy Grad-CAM."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from app.audit_db import ProvenanceLedger
from ml.gradcam import SpectrogramGradCAM
from ml.model import SpectraGuardNet, compute_log_mel
from ml.watermark import SpreadSpectrumWatermarker


DEFAULT_WATERMARK_HEX = "53504543"  # "SPEC"
MIN_WATERMARK_SAMPLES = 4096


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


class SpectraGuardEngine:
    """Real-time audio provenance, INT8 ONNX spoof detection, and Grad-CAM attribution engine."""

    def __init__(
        self,
        onnx_path: str | Path = "artifacts/spectraguard_int8.onnx",
        pt_path: str | Path = "artifacts/spectraguard.pt",
        ledger: ProvenanceLedger | None = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.pt_path = Path(pt_path)
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

        # Lazily initialized only when include_gradcam=True
        self._pt_model: SpectraGuardNet | None = None
        self._gradcam: SpectrogramGradCAM | None = None

    def _ensure_gradcam(self) -> SpectrogramGradCAM:
        """Load PyTorch checkpoint and Grad-CAM hooks lazily on first saliency request."""
        if self._gradcam is None:
            if not self.pt_path.exists():
                raise FileNotFoundError(f"Missing PyTorch checkpoint for Grad-CAM: {self.pt_path}")
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
        log_mel = compute_log_mel(arr, sample_rate=16000, n_fft=512, hop_length=256, n_mels=64)
        log_mel_np = log_mel.numpy().astype(np.float32)
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
            heatmap, summary = gradcam.generate(log_mel)
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
