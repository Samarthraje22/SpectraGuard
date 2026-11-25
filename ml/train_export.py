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
from ml.signals import CLIP_SAMPLES, CLIP_SECONDS, SAMPLE_RATE, _generate_genuine_clip, _generate_spoof_clip


ARTIFACTS_DIR = Path("artifacts")


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

    # Save PyTorch weights, compressed NumPy weights, & export FP32 + INT8 ONNX
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pt_path = artifacts_dir / "spectraguard.pt"
    npz_path = artifacts_dir / "spectraguard_weights.npz"
    fp32_onnx_path = artifacts_dir / "spectraguard_fp32.onnx"
    int8_onnx_path = artifacts_dir / "spectraguard_int8.onnx"

    state_dict = model.state_dict()
    torch.save(state_dict, pt_path)
    np.savez_compressed(
        npz_path,
        **{
            k: v.detach().cpu().numpy().astype(np.float32)
            for k, v in state_dict.items()
            if not k.endswith("num_batches_tracked")
        },
    )

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
    print(f"NumPy Weights NPZ:  {npz_path} ({npz_path.stat().st_size / 1024:.1f} KB)")
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
