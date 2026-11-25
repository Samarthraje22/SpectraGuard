"""Active spread-spectrum DSP watermarking in 1.5 kHz - 4.0 kHz speech-masked bands."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.linalg import hadamard


class SpreadSpectrumWatermarker:
    """Direct-Sequence Spread Spectrum (DSSS) audio watermarker for 16 kHz speech."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chip_length: int = 128,
        num_bits: int = 32,
        low_hz: float = 1500.0,
        high_hz: float = 4000.0,
        seed: int = 42,
        target_snr_db: float = 33.0,
    ) -> None:
        if chip_length <= 0:
            raise ValueError(f"chip_length must be positive, got {chip_length}")
        if num_bits <= 0:
            raise ValueError(f"num_bits must be positive, got {num_bits}")
        nyquist = 0.5 * sample_rate
        if not (0.0 < low_hz < high_hz < nyquist):
            raise ValueError(
                f"Bandpass range ({low_hz}, {high_hz}) must lie strictly within (0, {nyquist})"
            )

        self.sample_rate = sample_rate
        self.chip_length = chip_length
        self.num_bits = num_bits
        self.frame_length = num_bits * chip_length
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.seed = seed
        self.target_snr_db = target_snr_db

        # Deterministic bipolar PN sequence (seed=42) spread across the full frame via Hadamard codes
        rng = np.random.default_rng(self.seed)
        self.pn_sequences = rng.choice([-1.0, 1.0], size=self.frame_length).astype(np.float64)

        pow2 = 1 << (self.num_bits - 1).bit_length()
        h_mat = hadamard(pow2)[: self.num_bits, : self.num_bits].astype(np.float64)
        raw_codes = np.repeat(h_mat, self.chip_length, axis=1) * self.pn_sequences[None, :]

        # Zero-phase Butterworth bandpass filter for 1.5 kHz - 4.0 kHz shaping
        self._sos = signal.butter(
            6, [self.low_hz, self.high_hz], btype="bandpass", fs=self.sample_rate, output="sos"
        )

        # Restrict spread codes strictly to [low_hz, high_hz] and orthonormalize via QR
        freqs = np.fft.rfftfreq(self.frame_length, d=1.0 / self.sample_rate)
        self._inband_mask = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        codes_fft = np.fft.rfft(raw_codes, axis=1)
        codes_fft[:, ~self._inband_mask] = 0.0
        bp_codes = np.fft.irfft(codes_fft, n=self.frame_length, axis=1)
        q_mat, _ = np.linalg.qr(bp_codes.T)
        self._templates = q_mat.T  # shape: (num_bits, frame_length), orthonormal rows

    def _speech_mask_envelope(self, x: np.ndarray) -> np.ndarray:
        """Compute a smooth speech-masking amplitude envelope over ~32 ms windows."""
        win_len = max(1, int(0.032 * self.sample_rate))
        kernel = np.ones(win_len, dtype=np.float64) / float(win_len)
        local_rms = np.sqrt(np.convolve(x**2, kernel, mode="same") + 1e-12)
        global_rms = float(np.sqrt(np.mean(x**2) + 1e-12))
        return np.maximum(local_rms, 0.5 * global_rms)

    def embed(self, audio: np.ndarray, payload_bits: list[int]) -> tuple[np.ndarray, float]:
        """Embed a 32-bit payload into 1.5-4.0 kHz speech-masked bands.

        Returns (watermarked_audio_float32, measured_snr_db).
        """
        x = np.asarray(audio, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D audio array, got shape {x.shape}")
        if len(x) < self.frame_length:
            raise ValueError(
                f"Audio length ({len(x)}) must be at least {self.frame_length} samples "
                f"for {self.num_bits} bits at chip_length={self.chip_length}"
            )
        if len(payload_bits) != self.num_bits:
            raise ValueError(f"Expected {self.num_bits} payload bits, got {len(payload_bits)}")
        if any(bit not in (0, 1) for bit in payload_bits):
            raise ValueError("All payload_bits must be 0 or 1")

        bipolar = np.array([1.0 if b == 1 else -1.0 for b in payload_bits], dtype=np.float64)
        base_frame = bipolar @ self._templates  # shape: (frame_length,)

        n = len(x)
        repeats = (n + self.frame_length - 1) // self.frame_length
        raw_wm = np.tile(base_frame, repeats)[:n]

        env = self._speech_mask_envelope(x)
        shaped_wm = signal.sosfiltfilt(self._sos, raw_wm * env)

        sig_power = float(np.mean(x**2))
        wm_power = float(np.mean(shaped_wm**2))
        if sig_power <= 1e-12 or wm_power <= 1e-12:
            out = np.clip(x, -1.0, 1.0).astype(np.float32)
            return out, 0.0

        target_wm_power = sig_power / (10.0 ** (self.target_snr_db / 10.0))
        scale = float(np.sqrt(target_wm_power / wm_power))
        watermarked = np.clip(x + scale * shaped_wm, -1.0, 1.0)

        actual_noise_power = float(np.mean((watermarked - x) ** 2)) + 1e-15
        snr_db = float(10.0 * np.log10(sig_power / actual_noise_power))
        return watermarked.astype(np.float32), snr_db

    def extract(self, audio: np.ndarray) -> tuple[list[int], float]:
        """Extract the 32-bit payload and return (recovered_bits, mean_correlation_confidence)."""
        x = np.asarray(audio, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D audio array, got shape {x.shape}")
        if len(x) < self.frame_length:
            raise ValueError(
                f"Audio length ({len(x)}) must be at least {self.frame_length} samples"
            )

        # Bandpass-filter incoming audio to 1.5 - 4.0 kHz
        x_bp = signal.sosfiltfilt(self._sos, x)

        # Wiener-regularized local RMS normalization balances voiced vs quieter segments
        win_len = max(1, int(0.032 * self.sample_rate))
        kernel = np.ones(win_len, dtype=np.float64) / float(win_len)
        local_pow = np.convolve(x_bp**2, kernel, mode="same") + 0.25 * float(np.mean(x_bp**2)) + 1e-12
        x_norm = x_bp / np.sqrt(local_pow)

        # Coherently fold complete frames and apply folded in-band spectral whitening
        n_full = len(x_norm) // self.frame_length
        folded = x_norm[: n_full * self.frame_length].reshape(n_full, self.frame_length).mean(axis=0)

        spec = np.fft.rfft(folded)
        mag = np.abs(spec)
        inband_mags = mag[self._inband_mask]
        if inband_mags.size > 0:
            floor = 0.2 * float(np.median(inband_mags)) + 1e-12
            w_spec = np.zeros_like(spec)
            w_spec[self._inband_mask] = spec[self._inband_mask] / (mag[self._inband_mask] + floor)
            folded_w = np.fft.irfft(w_spec, n=self.frame_length)
        else:
            folded_w = folded

        corrs = self._templates @ folded_w
        recovered_bits = [1 if c >= 0.0 else 0 for c in corrs]

        frame_norm = float(np.linalg.norm(folded_w)) + 1e-12
        norm_corrs = np.abs(corrs) * np.sqrt(self.num_bits) / frame_norm
        mean_conf = float(np.clip(np.mean(norm_corrs), 0.0, 1.0))
        return recovered_bits, mean_conf
