"""
A convolutional variational autoencoder for two-channel signal windows.

The public tensor layout is (batch, time, channels). Reconstruction values are
unbounded, so the model can operate on channel-standardized PCM signals.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    sequence_length: int = 2048
    input_channels: int = 2
    latent_dim: int = 16
    logvar_min: float | None = None
    logvar_max: float | None = None

    def __post_init__(self) -> None:
        if self.sequence_length < 64 or self.sequence_length % 64:
            raise ValueError("sequence_length must be a positive multiple of 64")
        if self.input_channels < 1 or self.latent_dim < 1:
            raise ValueError("input_channels and latent_dim must be positive")
        if (self.logvar_min is None) != (self.logvar_max is None):
            raise ValueError("logvar_min and logvar_max must both be set or both be None")
        if self.logvar_min is not None and not (
            math.isfinite(self.logvar_min) and math.isfinite(self.logvar_max)
            and self.logvar_min < self.logvar_max
        ):
            raise ValueError("logvar bounds must be finite and satisfy logvar_min < logvar_max")


class VAE(nn.Module):
    """
    CNN1D-VAE with three stride-4 encoder and decoder stages.

    Evaluation uses the posterior mean by default. Pass ``sample=True`` to
    draw from the posterior, including during evaluation.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        self.encoded_length = self.config.sequence_length // 64
        self.encoder = nn.Sequential(
            nn.Conv1d(self.config.input_channels, 16, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.Flatten(),
        )
        features = 64 * self.encoded_length
        self.mu_head = nn.Linear(features, self.config.latent_dim)
        self.logvar_head = nn.Linear(features, self.config.latent_dim)
        self.decoder_input = nn.Linear(self.config.latent_dim, features)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(16, self.config.input_channels, kernel_size=8, stride=4, padding=2),
        )

    def bound_logvar(self, logvar: Tensor) -> Tensor:
        """Apply configured posterior bounds; None preserves legacy behavior."""
        if self.config.logvar_min is None:
            return logvar
        return torch.clamp(logvar, min=self.config.logvar_min, max=self.config.logvar_max)

    def forward(self, x: Tensor, sample: bool | None = None) -> dict[str, Tensor]:
        if x.ndim != 3 or tuple(x.shape[1:]) != (
            self.config.sequence_length, self.config.input_channels
        ):
            raise ValueError(
                "Expected input shape (batch, "
                f"{self.config.sequence_length}, {self.config.input_channels}); "
                f"received {tuple(x.shape)}"
            )
        if x.dtype != torch.float32:
            raise TypeError("Input must be a float32 tensor")
        encoded = self.encoder(x.transpose(1, 2))
        mu = self.mu_head(encoded)
        logvar = self.bound_logvar(self.logvar_head(encoded))
        use_sample = self.training if sample is None else sample
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if use_sample else mu
        decoded = self.decoder_input(z).reshape(x.shape[0], 64, self.encoded_length)
        reconstruction = self.decoder(decoded).transpose(1, 2).contiguous()
        return {"reconstruction": reconstruction, "mu": mu, "logvar": logvar, "z": z}


def loss_function(
    output: dict[str, Tensor], x: Tensor, beta: float = 0.001
) -> dict[str, Tensor]:
    """Mean elementwise MSE plus beta times mean per-window Gaussian KL.

    KL is summed across latent coordinates, then averaged across the batch.
    The losses are differentiable tensors; callers decide when to detach.
    """
    if output["reconstruction"].shape != x.shape:
        raise ValueError("Reconstruction and target shapes must match")
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    reconstruction_loss = F.mse_loss(output["reconstruction"], x, reduction="mean")
    mu, logvar = output["mu"], output["logvar"]
    kl_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=-1).mean()
    return {
        "loss": reconstruction_loss + beta * kl_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
    }
