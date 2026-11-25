"""Torch-free deterministic 16 kHz genuine speech and vocoder spoof waveform generators."""

from __future__ import annotations

import numpy as np
from scipy import signal


SAMPLE_RATE = 16000
CLIP_SECONDS = 2.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)  # 32,000 samples


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
    hop = int(rng.choice([160, 200, 256, 320]))  # 10-20 ms vocoder frame hops
    phase = np.zeros_like(t)
    cur_phase = 0.0
    dt_phase = 2.0 * np.pi * f0_base / SAMPLE_RATE
    for i in range(CLIP_SAMPLES):
        if i > 0 and i % hop == 0:
            cur_phase += rng.uniform(0.45, 1.35) * rng.choice([-1.0, 1.0])
        cur_phase += dt_phase
        phase[i] = cur_phase

    # Spectrally smoothed / flat harmonic envelope + in-band buzz
    voiced = np.zeros_like(t)
    for k in range(1, 24):
        amp = 1.0 / (k**0.75)
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
