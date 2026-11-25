"""Phase 1 unit tests for active spread-spectrum watermarking and corruptions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from ml.corruptions import (
    add_gaussian_noise,
    high_freq_adversarial_noise,
    lossy_quantization,
    pitch_gain_shift,
    resampling_jitter,
    telephony_bandpass,
)
from ml.watermark import SpreadSpectrumWatermarker


SAMPLE_RATE = 16000


def _simulate_speech(duration_sec: float = 2.0, seed: int = 123) -> np.ndarray:
    """Generate a realistic 16 kHz simulated speech waveform with pitch, formants, and syllabic envelope."""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE

    # Glottal pulse train approximation (F0 ~ 135 Hz with slight vibrato)
    f0 = 135.0 + 4.0 * np.sin(2.0 * np.pi * 3.0 * t)
    phase = 2.0 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    voiced = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 20))

    # Add fricative/aspiration turbulence shaped into speech band
    aspiration = rng.standard_normal(n) * 0.25
    sos_formant1 = signal.butter(4, [300.0, 900.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    sos_formant2 = signal.butter(4, [1500.0, 3200.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    speech_raw = (
        0.7 * signal.sosfiltfilt(sos_formant1, voiced)
        + 0.35 * signal.sosfiltfilt(sos_formant2, voiced + aspiration)
    )

    # Syllabic amplitude envelope (~4 Hz modulation)
    envelope = 0.55 + 0.45 * np.sin(2.0 * np.pi * 3.8 * t + 0.4)
    speech = speech_raw * envelope
    peak = float(np.max(np.abs(speech))) + 1e-12
    return (0.75 * speech / peak).astype(np.float32)


def _bit_accuracy(expected: list[int], actual: list[int]) -> float:
    assert len(expected) == len(actual)
    matches = sum(int(e == a) for e, a in zip(expected, actual))
    return matches / len(expected)


def test_watermark_snr_and_clean_recovery() -> None:
    wm = SpreadSpectrumWatermarker(sample_rate=SAMPLE_RATE, chip_length=128, seed=42)
    speech = _simulate_speech(duration_sec=2.0, seed=7)
    payload = [1, 0, 1, 1, 0, 0, 1, 0] * 4  # 32 bits

    watermarked, snr_db = wm.embed(speech, payload)

    assert watermarked.shape == speech.shape
    assert watermarked.dtype == np.float32
    assert float(np.min(watermarked)) >= -1.0
    assert float(np.max(watermarked)) <= 1.0
    assert snr_db > 32.0, f"Expected SNR > 32.0 dB, got {snr_db:.2f} dB"

    recovered, confidence = wm.extract(watermarked)
    assert recovered == payload, f"Expected 32/32 clean recovery, got {_bit_accuracy(payload, recovered):.2%}"
    assert 0.0 < confidence <= 1.0


def test_watermark_survives_telephony_and_gaussian_noise() -> None:
    wm = SpreadSpectrumWatermarker(sample_rate=SAMPLE_RATE, chip_length=128, seed=42)
    speech = _simulate_speech(duration_sec=2.0, seed=99)
    payload = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1] * 2

    watermarked, snr_db = wm.embed(speech, payload)
    assert snr_db > 32.0

    # Individual degradations
    tel_audio = telephony_bandpass(watermarked, low_hz=300.0, high_hz=3400.0)
    tel_bits, _ = wm.extract(tel_audio)
    tel_acc = _bit_accuracy(payload, tel_bits)
    assert tel_acc >= 0.90, f"Telephony bit recovery {tel_acc:.2%} < 90%"

    noisy_audio = add_gaussian_noise(watermarked, snr_db=20.0)
    noisy_bits, _ = wm.extract(noisy_audio)
    noisy_acc = _bit_accuracy(payload, noisy_bits)
    assert noisy_acc >= 0.90, f"Gaussian noise bit recovery {noisy_acc:.2%} < 90%"

    # Combined telephony bandpass + additive Gaussian noise
    combined = add_gaussian_noise(tel_audio, snr_db=20.0)
    combined_bits, _ = wm.extract(combined)
    combined_acc = _bit_accuracy(payload, combined_bits)
    assert combined_acc >= 0.90, f"Combined degradation bit recovery {combined_acc:.2%} < 90%"


def test_watermark_deterministic_and_validates_inputs() -> None:
    wm1 = SpreadSpectrumWatermarker(sample_rate=SAMPLE_RATE, chip_length=128, seed=42)
    wm2 = SpreadSpectrumWatermarker(sample_rate=SAMPLE_RATE, chip_length=128, seed=42)
    assert np.array_equal(wm1.pn_sequences, wm2.pn_sequences)

    speech = _simulate_speech(duration_sec=1.0, seed=11)
    payload = [1, 0] * 16

    out1, snr1 = wm1.embed(speech, payload)
    out2, snr2 = wm2.embed(speech, payload)
    assert np.array_equal(out1, out2)
    assert snr1 == pytest.approx(snr2)

    with pytest.raises(ValueError):
        wm1.embed(speech, [1, 0, 1])  # Wrong payload length
    with pytest.raises(ValueError):
        wm1.embed(speech, [2] * 32)  # Non-binary bit
    with pytest.raises(ValueError):
        wm1.embed(np.zeros(1024, dtype=np.float32), payload)  # Too short


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (add_gaussian_noise, {"snr_db": 20.0}),
        (telephony_bandpass, {"low_hz": 300.0, "high_hz": 3400.0}),
        (resampling_jitter, {"intermediate_sr": 8000}),
        (pitch_gain_shift, {"gain_db": -3.0, "time_stretch_factor": 1.02}),
        (lossy_quantization, {"bits": 8}),
        (high_freq_adversarial_noise, {"eps": 0.015}),
    ],
)
def test_corruptions_preserve_shape_bounds_dtype_and_determinism(fn, kwargs: dict) -> None:
    speech = _simulate_speech(duration_sec=2.0, seed=21)
    # Also test clipping behavior with near-full-scale peaks
    speech_hot = np.clip(speech * 1.35, -1.0, 1.0).astype(np.float32)

    out1 = fn(speech_hot, **kwargs)
    out2 = fn(speech_hot, **kwargs)

    assert out1.shape == speech_hot.shape
    assert out1.dtype == np.float32
    assert np.all(np.isfinite(out1))
    assert float(np.min(out1)) >= -1.0
    assert float(np.max(out1)) <= 1.0
    assert np.array_equal(out1, out2), f"{fn.__name__} must be deterministic"
