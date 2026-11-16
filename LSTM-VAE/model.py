"""LSTM variational autoencoder for fixed-length multichannel waveforms.

Inputs and reconstructions use (batch, time, channel) order. Consecutive sample
frames can shorten the recurrent sequence while preserving every input value.
The decoder is driven entirely by the latent code, never the target waveform.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Dataset.Preprocessing.waveform_frames import frame_waveform, unframe_waveform  # noqa: E402


@dataclass
class ModelConfig:
    sequence_length: int = 2048
    input_channels: int = 2
    latent_dim: int = 16
    hidden_size: int = 32
    num_layers: int = 1
    logvar_min: float | None = None
    logvar_max: float | None = None
    frame_size: int = 1

    def __post_init__(self) -> None:
        for name in ("sequence_length", "input_channels", "latent_dim", "hidden_size", "num_layers", "frame_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (self.logvar_min is None) != (self.logvar_max is None):
            raise ValueError("logvar_min and logvar_max must both be set or both be None")
        if self.logvar_min is not None and not (
            math.isfinite(self.logvar_min) and math.isfinite(self.logvar_max)
            and self.logvar_min < self.logvar_max
        ):
            raise ValueError("logvar bounds must be finite and satisfy logvar_min < logvar_max")

    @property
    def recurrent_steps(self) -> int:
        return (self.sequence_length+self.frame_size-1)//self.frame_size


class VAE(nn.Module):
    """Encode the final LSTM state into a Gaussian latent distribution."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.LSTM(
            config.frame_size*config.input_channels, config.hidden_size,
            num_layers=config.num_layers, batch_first=True,
        )
        self.to_mu = nn.Linear(config.hidden_size, config.latent_dim)
        self.to_logvar = nn.Linear(config.hidden_size, config.latent_dim)
        self.latent_to_hidden = nn.Linear(
            config.latent_dim, config.hidden_size * config.num_layers,
        )
        self.latent_to_cell = nn.Linear(
            config.latent_dim, config.hidden_size * config.num_layers,
        )
        self.decoder = nn.LSTM(
            config.latent_dim, config.hidden_size,
            num_layers=config.num_layers, batch_first=True,
        )
        self.to_output = nn.Linear(config.hidden_size, config.frame_size*config.input_channels)

    def bound_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
        """Apply configured posterior bounds; None preserves legacy behavior."""
        if self.config.logvar_min is None:
            return logvar
        return torch.clamp(logvar, min=self.config.logvar_min, max=self.config.logvar_max)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or tuple(x.shape[1:]) != (
            self.config.sequence_length, self.config.input_channels,
        ):
            raise ValueError(
                "Expected input shape (batch, "
                f"{self.config.sequence_length}, {self.config.input_channels})"
            )
        if not x.is_floating_point():
            raise ValueError("Input must contain floating-point, preprocessed samples")
        framed = frame_waveform(x, self.config.frame_size)
        _, (hidden, _) = self.encoder(framed)
        summary = hidden[-1]
        return self.to_mu(summary), self.bound_logvar(self.to_logvar(summary))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        batch = z.shape[0]
        hidden = torch.tanh(self.latent_to_hidden(z)).reshape(
            batch, cfg.num_layers, cfg.hidden_size,
        ).transpose(0, 1).contiguous()
        cell = torch.tanh(self.latent_to_cell(z)).reshape(
            batch, cfg.num_layers, cfg.hidden_size,
        ).transpose(0, 1).contiguous()
        repeated_z = z.unsqueeze(1).expand(-1, cfg.recurrent_steps, -1).contiguous()
        decoded, _ = self.decoder(repeated_z, (hidden, cell))
        framed_output = self.to_output(decoded)
        return unframe_waveform(framed_output, cfg.frame_size, cfg.sequence_length, cfg.input_channels)

    def forward(self, x: torch.Tensor, sample: bool | None = None) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        if sample is None:
            sample = self.training
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return {
            "reconstruction": self.decode(z),
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }


def loss_function(
    output: dict[str, torch.Tensor], target: torch.Tensor, beta: float = 0.001,
) -> dict[str, torch.Tensor]:
    """Mean reconstruction MSE plus beta times batch-mean latent KL sum."""
    reconstruction_loss = F.mse_loss(output["reconstruction"], target)
    mu, logvar = output["mu"], output["logvar"]
    kl_loss = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=-1)).mean()
    return {
        "loss": reconstruction_loss + beta * kl_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
    }
