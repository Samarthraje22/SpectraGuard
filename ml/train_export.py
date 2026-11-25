"""Training, Temperature Scaling calibration, benchmarking, and INT8 ONNX export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from scipy import signal
from torch.utils.data import DataLoader, TensorDataset

from ml.corruptions import (
    add_gaussian_noise,
    high_freq_adversarial_noise,
    lossy_quantization,
    pitch_gain_shift,
    resampling_jitter,
    telephony_bandpass,
)
from ml.model import SpectraGuardNet, compute_log_mel


SAMPLE_RATE = 16000
CLIP_SECONDS = 2.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)  # 32,000 samples
ARTIFACTS_DIR = Path("artifacts")


def _generate_genuine_clip(rng: np.random.Generator) -> np.ndarray:
    """Simulate natural human speech: smooth pitch contour, formants, and syllabic envelope."""
    t = np.arange(CLIP_SAMPLES, dtype=np.float64) / SAMPLE_RATE
    f0_base = rng.uniform(95.0, 250.0)
    contour_rate = rng.uniform(0.8, 2.5)
    contour_depth = rng.uniform(0.03, 0.12)
    f0 = f0_base * (1.0 + contour_depth * np.sin(2.0 * np.pi * contour_rate * t + rng.uniform(0, 6.28)))
    phase = 2.0 * np.pi * np.cumsum(f0) / SAMPLE_RATE

    # Natural glottal spectrum decays smoothly (~1/k^1.35)
    voiced = np.zeros_like(t)
    num_harmonics = int(rng.integers(10, 18))
    for k in range(1, num_harmonics + 1):
        amp = 1.0 / (k**1.35)
        jitter = rng.normal(0.0, 0.002, size=CLIP_SAMPLES)
        voiced += amp * np.sin(k * phase + jitter)

    # Natural vocal-tract formants (F1, F2, gentle F3) with natural rolloff above 3.8 kHz
    f1_lo = rng.uniform(250.0, 450.0)
    f1_hi = f1_lo + rng.uniform(350.0, 550.0)
    f2_lo = rng.uniform(1000.0, 1500.0)
    f2_hi = f2_lo + rng.uniform(600.0, 1100.0)

    sos1 = signal.butter(4, [f1_lo, f1_hi], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    sos2 = signal.butter(4, [f2_lo, f2_hi], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    aspiration = rng.standard_normal(CLIP_SAMPLES) * rng.uniform(0.05, 0.14)
    speech = 0.72 * signal.sosfiltfilt(sos1, voiced) + 0.32 * signal.sosfiltfilt(sos2, voiced + aspiration)

    # Multi-rate smooth syllabic modulation + natural pauses
    syl_rate = rng.uniform(2.5, 5.5)
    env = 0.55 + 0.45 * np.sin(2.0 * np.pi * syl_rate * t + rng.uniform(0, 6.28))
    env *= 0.75 + 0.25 * np.cos(2.0 * np.pi * 1.1 * t + rng.uniform(0, 6.28))
    out = speech * np.clip(env, 0.1, 1.0)
    peak = float(np.max(np.abs(out))) + 1e-12
    return (rng.uniform(0.55, 0.9) * out / peak).astype(np.float32)


def _generate_spoof_clip(rng: np.random.Generator) -> np.ndarray:
    """Simulate neural vocoder spoof: phase discontinuities, >4kHz metallic buzz, spectral smoothing."""
    t = np.arange(CLIP_SAMPLES, dtype=np.float64) / SAMPLE_RATE
    f0_base = rng.uniform(100.0, 250.0)
    # Over-smoothed / robotic pitch trajectory with periodic frame phase resets
    hop = int(rng.choice([160, 200, 256, 320]))  # 10-20 ms vocoder frame hops
    phase = np.zeros_like(t)
    cur_phase = 0.0
    dt_phase = 2.0 * np.pi * f0_base / SAMPLE_RATE
    for i in range(CLIP_SAMPLES):
        if i > 0 and i % hop == 0:
            # Phase discontinuity at vocoder frame boundary
            cur_phase += rng.uniform(0.45, 1.35) * rng.choice([-1.0, 1.0])
        cur_phase += dt_phase
        phase[i] = cur_phase

    # Spectrally smoothed / flat harmonic envelope + in-band buzz
    voiced = np.zeros_like(t)
    for k in range(1, 24):
        amp = 1.0 / (k**0.75)  # flatter spectral slope (vocoder over-brightening)
        voiced += amp * np.sin(k * phase)

    # Metallic upper-band (> 4.0 kHz) harmonic buzz / aliasing artifacts
    hf_f0 = rng.uniform(4200.0, 5600.0)
    hf_buzz = (
        0.55 * np.sin(2.0 * np.pi * hf_f0 * t)
        + 0.40 * np.sin(2.0 * np.pi * min(7600.0, hf_f0 + 1300.0) * t + 0.7)
        + 0.30 * np.sin(2.0 * np.pi * rng.uniform(5800.0, 7500.0) * t)
    )
    sos_hf = signal.butter(4, 4000.0, btype="highpass", fs=SAMPLE_RATE, output="sos")
    hf_component = signal.sosfiltfilt(sos_hf, hf_buzz + 0.25 * rng.standard_normal(CLIP_SAMPLES))

    # Narrowband vocoder formant buzz inside 300-3400 Hz (survives telephony bandpass)
    sos_mid = signal.butter(4, [400.0, 3200.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    mid_component = signal.sosfiltfilt(sos_mid, voiced)

    # Periodic frame-edge micro-glitches (audible vocoder concatenation pulse)
    frame_pulse = np.zeros_like(t)
    frame_pulse[::hop] = rng.uniform(0.15, 0.35) * rng.choice([-1.0, 1.0], size=len(frame_pulse[::hop]))
    frame_pulse = signal.sosfiltfilt(sos_mid, frame_pulse)

    syl_rate = rng.uniform(2.5, 5.5)
    env = 0.6 + 0.4 * np.sin(2.0 * np.pi * syl_rate * t + rng.uniform(0, 6.28))
    combined = (mid_component + 0.45 * frame_pulse + 0.28 * hf_component) * env
    peak = float(np.max(np.abs(combined))) + 1e-12
    return (rng.uniform(0.55, 0.9) * combined / peak).astype(np.float32)


def _apply_random_corruption(clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Deterministically apply clean pass-through or one of the 6 acoustic/adversarial corruptions."""
    choice = int(rng.integers(0, 8))
    seed = int(rng.integers(0, 100000))
    if choice <= 1:
        return clip
    if choice == 2:
        return add_gaussian_noise(clip, snr_db=float(rng.uniform(15.0, 28.0)), seed=seed)
    if choice == 3:
        return telephony_bandpass(clip, low_hz=300.0, high_hz=3400.0)
    if choice == 4:
        return resampling_jitter(clip, intermediate_sr=int(rng.choice([8000, 11025, 12000])))
    if choice == 5:
        return pitch_gain_shift(
            clip,
            gain_db=float(rng.uniform(-6.0, 2.0)),
            time_stretch_factor=float(rng.uniform(0.98, 1.03)),
        )
    if choice == 6:
        return lossy_quantization(clip, bits=int(rng.choice([8, 10, 12])))
    return high_freq_adversarial_noise(clip, eps=float(rng.uniform(0.008, 0.02)), seed=seed)


def generate_synthetic_dataset(
    num_samples: int,
    seed: int = 42,
    apply_corruptions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate balanced [num_samples, 1, 64, T] Log-Mel spectrograms and binary labels."""
    rng = np.random.default_rng(seed)
    waveforms = np.zeros((num_samples, CLIP_SAMPLES), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.float32)

    for idx in range(num_samples):
        is_spoof = idx % 2 == 1
        clip = _generate_spoof_clip(rng) if is_spoof else _generate_genuine_clip(rng)
        if apply_corruptions:
            clip = _apply_random_corruption(clip, rng)
        waveforms[idx] = clip
        labels[idx] = 1.0 if is_spoof else 0.0

    # Shuffle deterministically
    perm = rng.permutation(num_samples)
    waveforms = waveforms[perm]
    labels = labels[perm]

    log_mels = compute_log_mel(waveforms, sample_rate=SAMPLE_RATE, n_fft=512, hop_length=256, n_mels=64)
    return log_mels, torch.from_numpy(labels).unsqueeze(1)


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute exact ROC-AUC via trapezoidal integration over sorted score thresholds."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5

    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    tps = np.cumsum(sorted_true == 1)
    fps = np.cumsum(sorted_true == 0)
    tpr = np.concatenate(([0.0], tps / pos))
    fpr = np.concatenate(([0.0], fps / neg))
    return float(np.trapezoid(tpr, fpr))


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Equal Error Rate (EER %) via linear interpolation on the ROC curve."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 50.0

    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    tpr = np.concatenate(([0.0], np.cumsum(sorted_true == 1) / pos))
    fpr = np.concatenate(([0.0], np.cumsum(sorted_true == 0) / neg))
    fnr = 1.0 - tpr

    diff = fpr - fnr
    idx = int(np.argmin(np.abs(diff)))
    if idx > 0 and diff[idx] * diff[idx - 1] < 0.0:
        # Linearly interpolate crossing point between idx-1 and idx
        w = abs(diff[idx - 1]) / (abs(diff[idx - 1]) + abs(diff[idx]) + 1e-12)
        eer = fpr[idx - 1] + w * (fpr[idx] - fpr[idx - 1])
    else:
        eer = 0.5 * (fpr[idx] + fnr[idx])
    return float(eer * 100.0)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) over probability confidence bins."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    # Binary confidence = max(p, 1-p) vs correctness of predicted class
    preds = (y_prob >= 0.5).astype(np.float64)
    confs = np.maximum(y_prob, 1.0 - y_prob)
    accs = (preds == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        count = int(np.sum(mask))
        if count > 0:
            ece += (count / n) * abs(float(np.mean(accs[mask])) - float(np.mean(confs[mask])))
    return float(ece)


def calibrate_temperature(
    model: SpectraGuardNet,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
) -> tuple[float, float]:
    """Optimize model.temperature on held-out validation logits via L-BFGS NLL minimization."""
    model.eval()
    with torch.no_grad():
        raw_logits = model.forward_unscaled(val_x)
        pre_probs = torch.sigmoid(raw_logits).cpu().numpy().ravel()
        pre_ece = compute_ece(val_y.cpu().numpy().ravel(), pre_probs)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([model.temperature], lr=0.1, max_iter=100)

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        temp = torch.clamp(model.temperature, min=1e-3)
        loss = criterion(raw_logits / temp, val_y)
        loss.backward()
        return loss

    optimizer.step(_closure)
    with torch.no_grad():
        model.temperature.clamp_(min=0.05, max=20.0)
        cal_probs = torch.sigmoid(raw_logits / model.temperature).cpu().numpy().ravel()
        post_ece = compute_ece(val_y.cpu().numpy().ravel(), cal_probs)
    return pre_ece, post_ece


def run_pipeline(
    train_samples: int = 1200,
    val_samples: int = 400,
    epochs: int = 18,
    batch_size: int = 32,
    seed: int = 42,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict[str, float]:
    """Train SpectraGuardNet, calibrate Temperature Scaling, benchmark, and export INT8 ONNX."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Generating synthetic training and validation datasets (with 6 corruptions)...")
    train_x, train_y = generate_synthetic_dataset(train_samples, seed=seed, apply_corruptions=True)
    val_x, val_y = generate_synthetic_dataset(val_samples, seed=seed + 999, apply_corruptions=True)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    model = SpectraGuardNet(n_mels=64, dropout=0.1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = -1.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model.forward_unscaled(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(bx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model.forward_unscaled(val_x)
            val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()
            val_auc = compute_roc_auc(val_y.cpu().numpy().ravel(), val_probs)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 6 == 0 or epoch == epochs:
            avg_loss = total_loss / len(train_x)
            print(f"Epoch {epoch:02d}/{epochs:02d} | train_loss={avg_loss:.4f} | val_auc={val_auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    # Post-hoc Temperature Scaling calibration
    pre_ece, post_ece = calibrate_temperature(model, val_x, val_y)
    with torch.no_grad():
        cal_logits = model(val_x)
        cal_probs = torch.sigmoid(cal_logits).cpu().numpy().ravel()
        y_true_np = val_y.cpu().numpy().ravel()

    roc_auc = compute_roc_auc(y_true_np, cal_probs)
    eer_pct = compute_eer(y_true_np, cal_probs)
    learned_temp = float(model.temperature.item())

    print("\n=== Held-Out Validation Benchmarks ===")
    print(f"ROC-AUC:                {roc_auc:.4f}")
    print(f"Equal Error Rate (EER): {eer_pct:.2f}%")
    print(f"ECE (Before TempScale): {pre_ece:.4f}")
    print(f"ECE (After TempScale):  {post_ece:.4f} (T = {learned_temp:.4f})")

    # Save PyTorch weights & export FP32 + INT8 ONNX
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pt_path = artifacts_dir / "spectraguard.pt"
    fp32_onnx_path = artifacts_dir / "spectraguard_fp32.onnx"
    int8_onnx_path = artifacts_dir / "spectraguard_int8.onnx"

    torch.save(model.state_dict(), pt_path)

    dummy_input = torch.randn(1, 1, 64, val_x.shape[-1], dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy_input,),
        str(fp32_onnx_path),
        input_names=["log_mel"],
        output_names=["logits"],
        dynamic_axes={"log_mel": {0: "batch", 3: "time"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        weight_type=QuantType.QInt8,
    )

    # Verify ONNX Runtime INT8 output parity
    sess = ort.InferenceSession(str(int8_onnx_path), providers=["CPUExecutionProvider"])
    ort_logits = sess.run(["logits"], {"log_mel": val_x[:16].numpy()})[0]
    with torch.no_grad():
        pt_logits = model(val_x[:16]).numpy()
    max_prob_diff = float(
        np.max(np.abs(1.0 / (1.0 + np.exp(-ort_logits)) - 1.0 / (1.0 + np.exp(-pt_logits))))
    )

    fp32_bytes = fp32_onnx_path.stat().st_size
    int8_bytes = int8_onnx_path.stat().st_size
    compression_pct = (1.0 - int8_bytes / fp32_bytes) * 100.0

    print("\n=== ONNX Export & Quantization ===")
    print(f"PyTorch Checkpoint: {pt_path} ({pt_path.stat().st_size / 1024:.1f} KB)")
    print(f"FP32 ONNX Model:    {fp32_onnx_path} ({fp32_bytes / 1024:.1f} KB)")
    print(f"INT8 ONNX Model:    {int8_onnx_path} ({int8_bytes / 1024:.1f} KB)")
    print(f"Compression Ratio:  {compression_pct:.1f}% size reduction")
    print(f"Max Prob Diff (PT vs INT8 ORT): {max_prob_diff:.5f}")

    return {
        "roc_auc": roc_auc,
        "eer_pct": eer_pct,
        "pre_ece": pre_ece,
        "post_ece": post_ece,
        "temperature": learned_temp,
        "fp32_bytes": float(fp32_bytes),
        "int8_bytes": float(int8_bytes),
        "compression_pct": compression_pct,
        "max_prob_diff": max_prob_diff,
    }


if __name__ == "__main__":
    run_pipeline()
