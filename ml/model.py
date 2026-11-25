"""SpectraGuard CNN-Conformer deepfake detector and Log-Mel feature extractor."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torchaudio


@lru_cache(maxsize=4)
def _get_mel_transform(
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 64,
) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )


def compute_log_mel(
    waveforms: torch.Tensor | np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 64,
) -> torch.Tensor:
    """Convert 1D [N] or batched [B, N] waveforms into 64-bin Log-Mel spectrograms [B, 1, 64, T]."""
    if isinstance(waveforms, np.ndarray):
        wav = torch.from_numpy(np.asarray(waveforms, dtype=np.float32))
    else:
        wav = waveforms.to(dtype=torch.float32)

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim != 2:
        raise ValueError(f"Expected 1D [N] or 2D [B, N] waveforms, got shape {tuple(wav.shape)}")

    mel_transform = _get_mel_transform(sample_rate, n_fft, hop_length, n_mels).to(wav.device)
    mel = mel_transform(wav)  # [B, n_mels, T]
    log_mel = torch.log(torch.clamp(mel, min=1e-6))
    return log_mel.unsqueeze(1)  # [B, 1, n_mels, T]


class SpectraGuardNet(nn.Module):
    """Lightweight CNN-Conformer detector with Temperature Scaling calibration.

    Expects input Log-Mel spectrograms of shape [B, 1, 64, T].
    Outputs calibrated logits of shape [B, 1] (positive => spoof).
    """

    def __init__(self, n_mels: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        if n_mels != 64:
            raise ValueError(f"SpectraGuardNet expects 64 Mel bins, got {n_mels}")

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        # Exposed as target_conv for Time-Frequency Grad-CAM hooks
        self.target_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(64, 1)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward_unscaled(self, x: torch.Tensor) -> torch.Tensor:
        """Compute unscaled logits of shape [B, 1] prior to temperature division."""
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != 64:
            raise ValueError(f"Expected input shape [B, 1, 64, T], got {tuple(x.shape)}")

        h = self.block1(x)
        h = self.block2(h)
        h = self.target_conv(h)  # [B, 64, F', T']

        # Frequency pooling -> [B, T', 64]
        seq = h.mean(dim=2).transpose(1, 2)
        seq = self.transformer(seq)  # [B, T', 64]

        # Global temporal average pooling -> [B, 64] -> [B, 1]
        pooled = seq.mean(dim=1)
        return self.classifier(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute temperature-scaled logits of shape [B, 1]."""
        logits = self.forward_unscaled(x)
        temp = torch.clamp(self.temperature, min=1e-3)
        return logits / temp
