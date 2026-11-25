"""Zero-dependency pure-NumPy inference engine for SpectraGuardNet."""

from __future__ import annotations

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias


def _conv2d_bn_silu(
    x: np.ndarray,
    weight: np.ndarray,
    conv_bias: np.ndarray,
    bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    stride: int = 2,
    padding: int = 1,
    eps: float = 1e-5,
) -> np.ndarray:
    """Pure-NumPy Conv2d(k=3, stride=2, pad=1) + BatchNorm2d + SiLU."""
    batch, c_in, h, w = x.shape
    c_out, _, kh, kw = weight.shape

    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    # sliding_window_view over (H, W) -> [B, C_in, H_out_all, W_out_all, kh, kw]
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, (kh, kw), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride, :, :]  # [B, C_in, H_out, W_out, kh, kw]
    _, _, h_out, w_out, _, _ = windows.shape

    # Reshape to [B * H_out * W_out, C_in * kh * kw]
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch * h_out * w_out, c_in * kh * kw)
    w_mat = weight.reshape(c_out, c_in * kh * kw).T

    conv_out = (cols @ w_mat) + conv_bias[None, :]
    conv_out = conv_out.reshape(batch, h_out, w_out, c_out).transpose(0, 3, 1, 2)

    # BatchNorm2d eval inference
    scale = bn_weight / np.sqrt(bn_var + eps)
    normed = (conv_out - bn_mean[None, :, None, None]) * scale[None, :, None, None] + bn_bias[
        None, :, None, None
    ]
    return _silu(normed)


def _transformer_layer(
    x: np.ndarray,
    weights: dict[str, np.ndarray],
    prefix: str,
    nhead: int = 4,
) -> np.ndarray:
    """Pure-NumPy TransformerEncoderLayer (batch_first=True, norm_first=False, ReLU)."""
    batch, seq_len, d_model = x.shape
    head_dim = d_model // nhead

    in_w = weights[f"{prefix}.self_attn.in_proj_weight"]  # [3*D, D]
    in_b = weights[f"{prefix}.self_attn.in_proj_bias"]  # [3*D]
    qkv = x @ in_w.T + in_b[None, None, :]  # [B, T, 3*D]
    q, k, v = np.split(qkv, 3, axis=-1)

    q = q.reshape(batch, seq_len, nhead, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, nhead, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, nhead, head_dim).transpose(0, 2, 1, 3)

    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(float(head_dim))
    attn = _softmax(scores, axis=-1)
    ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

    out_w = weights[f"{prefix}.self_attn.out_proj.weight"]
    out_b = weights[f"{prefix}.self_attn.out_proj.bias"]
    attn_out = ctx @ out_w.T + out_b[None, None, :]

    x1 = _layer_norm(
        x + attn_out,
        weights[f"{prefix}.norm1.weight"],
        weights[f"{prefix}.norm1.bias"],
    )

    ff1 = _relu(
        x1 @ weights[f"{prefix}.linear1.weight"].T + weights[f"{prefix}.linear1.bias"][None, None, :]
    )
    ff2 = ff1 @ weights[f"{prefix}.linear2.weight"].T + weights[f"{prefix}.linear2.bias"][None, None, :]

    return _layer_norm(
        x1 + ff2,
        weights[f"{prefix}.norm2.weight"],
        weights[f"{prefix}.norm2.bias"],
    )


class NumpySpectraGuard:
    """Zero-dependency pure-NumPy SpectraGuardNet forward engine."""

    def __init__(self, weights_path: str | Path = "artifacts/spectraguard_weights.npz") -> None:
        p = Path(weights_path)
        if not p.exists():
            candidate = PROJECT_ROOT / p
            if candidate.exists():
                p = candidate
        if not p.exists():
            raise FileNotFoundError(f"Missing NumPy weights archive: {weights_path}")

        with np.load(p) as data:
            self.weights = {k: data[k].astype(np.float32) for k in data.files}

    def predict_logit(self, log_mel: np.ndarray) -> float:
        """Compute calibrated logit for a single [1, 1, 64, T] Log-Mel spectrogram."""
        x = np.asarray(log_mel, dtype=np.float32)
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != 64:
            raise ValueError(f"Expected shape [B, 1, 64, T], got {x.shape}")

        w = self.weights
        h = _conv2d_bn_silu(
            x,
            w["block1.0.weight"],
            w["block1.0.bias"],
            w["block1.1.weight"],
            w["block1.1.bias"],
            w["block1.1.running_mean"],
            w["block1.1.running_var"],
        )
        h = _conv2d_bn_silu(
            h,
            w["block2.0.weight"],
            w["block2.0.bias"],
            w["block2.1.weight"],
            w["block2.1.bias"],
            w["block2.1.running_mean"],
            w["block2.1.running_var"],
        )
        h = _conv2d_bn_silu(
            h,
            w["target_conv.0.weight"],
            w["target_conv.0.bias"],
            w["target_conv.1.weight"],
            w["target_conv.1.bias"],
            w["target_conv.1.running_mean"],
            w["target_conv.1.running_var"],
        )

        # Frequency pooling -> [B, T', 64]
        seq = h.mean(axis=2).transpose(0, 2, 1)
        seq = _transformer_layer(seq, w, "transformer.layers.0", nhead=4)
        seq = _transformer_layer(seq, w, "transformer.layers.1", nhead=4)

        pooled = seq.mean(axis=1)  # [B, 64]
        raw_logit = pooled @ w["classifier.weight"].T + w["classifier.bias"][None, :]
        temp = max(float(w["temperature"].item()), 1e-3)
        return float((raw_logit / temp)[0, 0])

    def predict_proba(self, log_mel: np.ndarray) -> float:
        """Compute calibrated spoof probability in [0.0, 1.0]."""
        logit = self.predict_logit(log_mel)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0))))


_DEFAULT_ENGINE: NumpySpectraGuard | None = None


def predict_spoof_numpy(
    log_mel: np.ndarray,
    weights_path: str | Path = "artifacts/spectraguard_weights.npz",
) -> float:
    """Module-level helper returning calibrated spoof probability for [B, 1, 64, T] Log-Mel."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = NumpySpectraGuard(weights_path=weights_path)
    return _DEFAULT_ENGINE.predict_proba(log_mel)
