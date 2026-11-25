"""Deterministic 16 kHz acoustic and adversarial corruptions suite."""

from __future__ import annotations

import math

import numpy as np

try:
    from scipy import signal

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    signal = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False


SAMPLE_RATE = 16000


def _prepare_audio(audio: np.ndarray) -> np.ndarray:
    """Validate and convert input waveform to 1D float64 for processing."""
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("Audio array must not be empty")
    return arr


def _finalize_audio(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Ensure exact length, clip to [-1.0, 1.0], and cast to float32."""
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def add_gaussian_noise(
    audio: np.ndarray,
    snr_db: float = 20.0,
    seed: int = 42,
) -> np.ndarray:
    """Add deterministic zero-mean Gaussian noise at the requested SNR (dB)."""
    x = _prepare_audio(audio)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x.shape[0])

    sig_power = float(np.mean(x**2))
    if sig_power <= 1e-12:
        # Fallback to small noise floor if input is silent
        noise_scale = 10.0 ** (-snr_db / 20.0) * 1e-3
    else:
        target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise_power = float(np.mean(noise**2)) + 1e-12
        noise_scale = math.sqrt(target_noise_power / noise_power)

    return _finalize_audio(x + noise_scale * noise, len(x))


def telephony_bandpass(
    audio: np.ndarray,
    low_hz: float = 300.0,
    high_hz: float = 3400.0,
    sample_rate: int = SAMPLE_RATE,
    order: int = 5,
) -> np.ndarray:
    """Simulate narrowband telephony codec bandwidth using zero-phase Butterworth SOS filtering."""
    x = _prepare_audio(audio)
    nyquist = 0.5 * sample_rate
    if not (0.0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Bandpass cutoffs must satisfy 0 < low_hz < high_hz < {nyquist}, got ({low_hz}, {high_hz})"
        )
    if SCIPY_AVAILABLE and signal is not None:
        sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
        filtered = signal.sosfiltfilt(sos, x)
    else:
        freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
        spec = np.fft.rfft(x)
        lp = 1.0 / (1.0 + (freqs / high_hz) ** (2 * order))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(freqs > 0.0, (low_hz / np.maximum(freqs, 1e-9)) ** (2 * order), np.inf)
        hp = np.where(freqs > 0.0, 1.0 / (1.0 + ratio), 0.0)
        filtered = np.fft.irfft(spec * (lp * hp) ** 2, n=len(x))
    return _finalize_audio(filtered, len(x))


def resampling_jitter(
    audio: np.ndarray,
    intermediate_sr: int = 8000,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Downsample to intermediate_sr and upsample back to sample_rate to simulate codec resampling jitter."""
    x = _prepare_audio(audio)
    if intermediate_sr <= 0:
        raise ValueError(f"intermediate_sr must be positive, got {intermediate_sr}")
    if intermediate_sr == sample_rate:
        return _finalize_audio(x, len(x))

    if SCIPY_AVAILABLE and signal is not None:
        g1 = math.gcd(intermediate_sr, sample_rate)
        down = signal.resample_poly(x, intermediate_sr // g1, sample_rate // g1)
        up = signal.resample_poly(down, sample_rate // g1, intermediate_sr // g1)
    else:
        n_down = max(1, int(round(len(x) * intermediate_sr / sample_rate)))
        down = np.interp(np.linspace(0, len(x) - 1, n_down), np.arange(len(x)), x)
        up = np.interp(np.linspace(0, n_down - 1, len(x)), np.arange(n_down), down)
    return _finalize_audio(up, len(x))


def pitch_gain_shift(
    audio: np.ndarray,
    gain_db: float = -3.0,
    time_stretch_factor: float = 1.02,
) -> np.ndarray:
    """Apply deterministic gain scaling and time/pitch axis warping while preserving length."""
    x = _prepare_audio(audio)
    if time_stretch_factor <= 0.0:
        raise ValueError(f"time_stretch_factor must be positive, got {time_stretch_factor}")

    n = len(x)
    gain = 10.0 ** (gain_db / 20.0)
    scaled = x * gain

    # Resample time axis by time_stretch_factor (>1.0 slows/stretches, <1.0 speeds up)
    src_idx = np.arange(n, dtype=np.float64) / time_stretch_factor
    valid_mask = src_idx <= (n - 1)
    warped = np.zeros(n, dtype=np.float64)
    warped[valid_mask] = np.interp(src_idx[valid_mask], np.arange(n, dtype=np.float64), scaled)
    return _finalize_audio(warped, n)


def lossy_quantization(
    audio: np.ndarray,
    bits: int = 8,
) -> np.ndarray:
    """Apply deterministic uniform PCM quantization to the given bit depth."""
    x = _prepare_audio(audio)
    if bits < 2 or bits > 24:
        raise ValueError(f"bits must be in [2, 24], got {bits}")

    clipped = np.clip(x, -1.0, 1.0)
    levels = float((1 << (bits - 1)) - 1)
    quantized = np.round(clipped * levels) / levels
    return _finalize_audio(quantized, len(x))


def high_freq_adversarial_noise(
    audio: np.ndarray,
    eps: float = 0.015,
    cutoff_hz: float = 4000.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 42,
) -> np.ndarray:
    """Inject deterministic L-inf bounded high-frequency (>4 kHz) adversarial perturbation."""
    x = _prepare_audio(audio)
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    if eps == 0.0:
        return _finalize_audio(x, len(x))

    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(len(x))
    if SCIPY_AVAILABLE and signal is not None:
        sos = signal.butter(5, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
        hf = signal.sosfiltfilt(sos, raw)
    else:
        freqs = np.fft.rfftfreq(len(raw), d=1.0 / sample_rate)
        spec = np.fft.rfft(raw)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(freqs > 0.0, (cutoff_hz / np.maximum(freqs, 1e-9)) ** 10, np.inf)
        hp = np.where(freqs > 0.0, 1.0 / (1.0 + ratio), 0.0)
        hf = np.fft.irfft(spec * (hp**2), n=len(raw))
    max_abs = float(np.max(np.abs(hf)))
    if max_abs > 1e-12:
        hf = (hf / max_abs) * eps
    return _finalize_audio(x + hf, len(x))
