"""Phase 2 verification tests for CNN-Conformer, Grad-CAM, calibration, and INT8 ONNX export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from ml.gradcam import SpectrogramGradCAM
from ml.model import SpectraGuardNet, compute_log_mel
from ml.train_export import (
    _generate_spoof_clip,
    compute_ece,
    compute_eer,
    compute_roc_auc,
    generate_synthetic_dataset,
)


ARTIFACTS_DIR = Path("artifacts")
PT_PATH = ARTIFACTS_DIR / "spectraguard.pt"
FP32_ONNX_PATH = ARTIFACTS_DIR / "spectraguard_fp32.onnx"
INT8_ONNX_PATH = ARTIFACTS_DIR / "spectraguard_int8.onnx"


def _load_trained_model() -> SpectraGuardNet:
    assert PT_PATH.exists(), f"Missing trained checkpoint {PT_PATH}; run python -m ml.train_export first"
    model = SpectraGuardNet(n_mels=64, dropout=0.1)
    state = torch.load(PT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def test_onnx_int8_loads_and_matches_pytorch() -> None:
    assert FP32_ONNX_PATH.exists(), f"Missing {FP32_ONNX_PATH}"
    assert INT8_ONNX_PATH.exists(), f"Missing {INT8_ONNX_PATH}"

    model = _load_trained_model()
    val_x, _ = generate_synthetic_dataset(num_samples=24, seed=777, apply_corruptions=True)

    session = ort.InferenceSession(str(INT8_ONNX_PATH), providers=["CPUExecutionProvider"])
    ort_logits = session.run(["logits"], {"log_mel": val_x.numpy()})[0]
    with torch.no_grad():
        pt_logits = model(val_x).numpy()

    ort_probs = 1.0 / (1.0 + np.exp(-ort_logits))
    pt_probs = 1.0 / (1.0 + np.exp(-pt_logits))

    assert ort_probs.shape == pt_probs.shape == (24, 1)
    max_diff = float(np.max(np.abs(ort_probs - pt_probs)))
    assert max_diff < 0.08, f"INT8 quantization probability drift {max_diff:.4f} >= 0.08"
    assert INT8_ONNX_PATH.stat().st_size < FP32_ONNX_PATH.stat().st_size


def test_validation_benchmarks_and_calibration() -> None:
    model = _load_trained_model()
    val_x, val_y = generate_synthetic_dataset(num_samples=400, seed=42 + 999, apply_corruptions=True)

    with torch.no_grad():
        probs = torch.sigmoid(model(val_x)).cpu().numpy().ravel()
    y_true = val_y.cpu().numpy().ravel()

    roc_auc = compute_roc_auc(y_true, probs)
    eer_pct = compute_eer(y_true, probs)
    ece = compute_ece(y_true, probs)

    assert roc_auc > 0.92, f"Expected ROC-AUC > 0.92, got {roc_auc:.4f}"
    assert eer_pct < 10.0, f"Expected EER < 10.0%, got {eer_pct:.2f}%"
    assert ece < 0.08, f"Expected post-calibration ECE < 0.08, got {ece:.4f}"


def test_gradcam_produces_valid_saliency_on_spoof() -> None:
    model = _load_trained_model()
    rng = np.random.default_rng(314)
    spoof_wav = _generate_spoof_clip(rng)
    log_mel = compute_log_mel(spoof_wav)
    time_frames = log_mel.shape[-1]

    with SpectrogramGradCAM(model) as gradcam:
        heatmap, summary = gradcam.generate(log_mel)

    assert heatmap.shape == (64, time_frames)
    assert heatmap.dtype == np.float32
    assert np.all(np.isfinite(heatmap))
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0
    assert float(heatmap.max()) > 0.0, "Expected non-zero Grad-CAM saliency on spoof input"
    assert float(heatmap.sum()) > 0.0

    assert 0 <= summary["peak_freq_bin"] < 64
    assert 0 <= summary["peak_time_frame"] < time_frames
    assert 0.0 <= summary["high_freq_energy_ratio"] <= 1.0
