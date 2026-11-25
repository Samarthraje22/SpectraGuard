"""Time-Frequency Grad-CAM saliency attribution for SpectraGuardNet."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ml.model import SpectraGuardNet


class SpectrogramGradCAM:
    """Computes normalized [64, T] Time-Frequency Grad-CAM heatmaps on model.target_conv."""

    def __init__(self, model: SpectraGuardNet) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd_handle = self.model.target_conv.register_forward_hook(self._save_activation)
        self._bwd_handle = self.model.target_conv.register_full_backward_hook(self._save_gradient)

    def _save_activation(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self._activations = output

    def _save_gradient(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        if grad_output and grad_output[0] is not None:
            self._gradients = grad_output[0]

    def close(self) -> None:
        """Detach forward and backward hooks from the model."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self) -> SpectrogramGradCAM:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def generate(self, log_mel: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute normalized [64, T] Grad-CAM heatmap and summary dictionary for a single clip."""
        if log_mel.ndim == 2:
            x = log_mel.unsqueeze(0).unsqueeze(0)
        elif log_mel.ndim == 3:
            x = log_mel.unsqueeze(0)
        elif log_mel.ndim == 4:
            x = log_mel
        else:
            raise ValueError(f"Expected 2D/3D/4D spectrogram, got shape {tuple(log_mel.shape)}")

        if x.shape[0] != 1 or x.shape[1] != 1 or x.shape[2] != 64:
            raise ValueError(f"Expected single [1, 1, 64, T] spectrogram, got {tuple(x.shape)}")

        _, _, freq_bins, time_frames = x.shape
        was_training = self.model.training
        self.model.eval()
        self._activations = None
        self._gradients = None

        try:
            x_in = x.detach().clone().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x_in)
            score = logits[0, 0]
            score.backward()

            if self._activations is None or self._gradients is None:
                raise RuntimeError("Grad-CAM hooks failed to capture target_conv activations/gradients")

            acts = self._activations.detach()  # [1, 64, F', T']
            grads = self._gradients.detach()  # [1, 64, F', T']

            weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, 64, 1, 1]
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # [1, 1, F', T']
            cam_up = F.interpolate(
                cam,
                size=(freq_bins, time_frames),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            cam_np = cam_up.cpu().numpy().astype(np.float32)
            cam_min = float(cam_np.min())
            cam_max = float(cam_np.max())
            if cam_max - cam_min > 1e-8:
                heatmap = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                heatmap = np.zeros((freq_bins, time_frames), dtype=np.float32)

            heatmap = np.clip(heatmap, 0.0, 1.0).astype(np.float32)

            peak_flat = int(np.argmax(heatmap))
            peak_freq_bin = int(peak_flat // time_frames)
            peak_time_frame = int(peak_flat % time_frames)

            total_energy = float(heatmap.sum())
            high_band_energy = float(heatmap[freq_bins // 2 :, :].sum())
            high_freq_energy_ratio = float(high_band_energy / (total_energy + 1e-12))

            summary: dict[str, Any] = {
                "peak_freq_bin": peak_freq_bin,
                "peak_time_frame": peak_time_frame,
                "high_freq_energy_ratio": round(high_freq_energy_ratio, 4),
                "spoof_probability": round(float(torch.sigmoid(score.detach()).item()), 4),
            }
            return heatmap, summary
        finally:
            self.model.zero_grad(set_to_none=True)
            if was_training:
                self.model.train()

    def __call__(self, log_mel: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
        return self.generate(log_mel)
