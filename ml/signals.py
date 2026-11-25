"""Torch-free and SciPy-optional deterministic 16 kHz genuine speech and vocoder spoof generators."""

from __future__ import annotations

import numpy as np

try:
    from scipy import signal

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    signal = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False


SAMPLE_RATE = 16000
CLIP_SECONDS = 2.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)  # 32,000 samples


def _fft_bandpass(x: np.ndarray, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    freqs = np.fft.rfftfreq(len(x), d=1.0 / SAMPLE_RATE)
    spec = np.fft.rfft(x)
    lp = 1.0 / (1.0 + (freqs / high_hz) ** (2 * order))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(freqs > 0.0, (low_hz / np.maximum(freqs, 1e-9)) ** (2 * order), np.inf)
    hp = np.where(freqs > 0.0, 1.0 / (1.0 + ratio), 0.0)
    return np.fft.irfft(spec * (lp * hp) ** 2, n=len(x))


def _fft_highpass(x: np.ndarray, cutoff_hz: float, order: int = 4) -> np.ndarray:
    freqs = np.fft.rfftfreq(len(x), d=1.0 / SAMPLE_RATE)
    spec = np.fft.rfft(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(freqs > 0.0, (cutoff_hz / np.maximum(freqs, 1e-9)) ** (2 * order), np.inf)
    hp = np.where(freqs > 0.0, 1.0 / (1.0 + ratio), 0.0)
    return np.fft.irfft(spec * (hp**2), n=len(x))


def _bandpass(x: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    if SCIPY_AVAILABLE and signal is not None:
        sos = signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=SAMPLE_RATE, output="sos")
        return signal.sosfiltfilt(sos, x)
    return _fft_bandpass(x, low_hz, high_hz, order=4)


def _highpass(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    if SCIPY_AVAILABLE and signal is not None:
        sos = signal.butter(4, cutoff_hz, btype="highpass", fs=SAMPLE_RATE, output="sos")
        return signal.sosfiltfilt(sos, x)
    return _fft_highpass(x, cutoff_hz, order=4)


def _generate_genuine_clip(rng: np.random.Generator) -> np.ndarray:
    """Simulate natural human speech: smooth pitch contour, formants, and syllabic envelope."""
    t = np.arange(CLIP_SAMPLES, dtype=np.float64) / SAMPLE_RATE
    f0_base = rng.uniform(95.0, 250.0)
    contour_rate = rng.uniform(0.8, 2.5)
    contour_depth = rng.uniform(0.03, 0.12)
    f0 = f0_base * (1.0 + contour_depth * np.sin(2.0 * np.pi * contour_rate * t + rng.uniform(0, 6.28)))
    phase = 2.0 * np.pi * np.cumsum(f0) / SAMPLE_RATE

    voiced = np.zeros_like(t)
    num_harmonics = int(rng.integers(10, 18))
    for k in range(1, num_harmonics + 1):
        amp = 1.0 / (k**1.35)
        jitter = rng.normal(0.0, 0.002, size=CLIP_SAMPLES)
        voiced += amp * np.sin(k * phase + jitter)

    f1_lo = rng.uniform(250.0, 450.0)
    f1_hi = f1_lo + rng.uniform(350.0, 550.0)
    f2_lo = rng.uniform(1000.0, 1500.0)
    f2_hi = f2_lo + rng.uniform(600.0, 1100.0)

    aspiration = rng.standard_normal(CLIP_SAMPLES) * rng.uniform(0.05, 0.14)
    speech = 0.72 * _bandpass(voiced, f1_lo, f1_hi) + 0.32 * _bandpass(voiced + aspiration, f2_lo, f2_hi)

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
    hop = int(rng.choice([160, 200, 256, 320]))
    phase = np.zeros_like(t)
    cur_phase = 0.0
    dt_phase = 2.0 * np.pi * f0_base / SAMPLE_RATE
    for i in range(CLIP_SAMPLES):
        if i > 0 and i % hop == 0:
            cur_phase += rng.uniform(0.45, 1.35) * rng.choice([-1.0, 1.0])
        cur_phase += dt_phase
        phase[i] = cur_phase

    voiced = np.zeros_like(t)
    for k in range(1, 24):
        amp = 1.0 / (k**0.75)
        voiced += amp * np.sin(k * phase)

    hf_f0 = rng.uniform(4200.0, 5600.0)
    hf_buzz = (
        0.55 * np.sin(2.0 * np.pi * hf_f0 * t)
        + 0.40 * np.sin(2.0 * np.pi * min(7600.0, hf_f0 + 1300.0) * t + 0.7)
        + 0.30 * np.sin(2.0 * np.pi * rng.uniform(5800.0, 7500.0) * t)
    )
    hf_component = _highpass(hf_buzz + 0.25 * rng.standard_normal(CLIP_SAMPLES), 4000.0)
    mid_component = _bandpass(voiced, 400.0, 3200.0)

    frame_pulse = np.zeros_like(t)
    frame_pulse[::hop] = rng.uniform(0.15, 0.35) * rng.choice([-1.0, 1.0], size=len(frame_pulse[::hop]))
    frame_pulse = _bandpass(frame_pulse, 400.0, 3200.0)

    syl_rate = rng.uniform(2.5, 5.5)
    env = 0.6 + 0.4 * np.sin(2.0 * np.pi * syl_rate * t + rng.uniform(0, 6.28))
    combined = (mid_component + 0.45 * frame_pulse + 0.28 * hf_component) * env
    peak = float(np.max(np.abs(combined))) + 1e-12
    return (rng.uniform(0.55, 0.9) * combined / peak).astype(np.float32)
