"""State classifier using the existing LSTM-VAE encoder.

The default three-class head preserves the existing MSM checkpoint layout.
Other tasks supply their class count and retain their own label-name mapping.
"""

import torch
from torch import nn
from torch.nn import functional as F

from model import ModelConfig, VAE


class StateClassifier(nn.Module):
    """Classify posterior means and optionally reconstruct the input window."""

    def __init__(self, config: ModelConfig, num_classes: int = 3):
        super().__init__()
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least two")
        self.config = config
        self.num_classes = num_classes
        self.vae = VAE(config)
        self.head = nn.Sequential(
            nn.Linear(config.latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor, reconstruct: bool = True) -> dict[str, torch.Tensor]:
        if reconstruct:
            output = self.vae(x)
        else:
            mu, logvar = self.vae.encode(x)
            output = {"mu": mu, "logvar": logvar, "z": mu}
        output["logits"] = self.head(output["mu"])
        return output


def classification_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    x: torch.Tensor | None = None,
    reconstruction_weight: float = 0.05,
    beta: float = 0.001,
    class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """CE + reconstruction_weight * mean MSE + beta * batch-mean latent KL.

Reconstruction is optional. If it was requested by the forward pass and has
a nonzero loss weight, its target waveform must be supplied as ``x``.
"""
    ce_loss = F.cross_entropy(output["logits"], target, weight=class_weights)
    reconstruction_loss = ce_loss.new_zeros(())
    if "reconstruction" in output:
        if x is None:
            if reconstruction_weight != 0:
                raise ValueError("Provide x to compute the reconstruction loss, or use reconstruct=False")
        else:
            reconstruction_loss = F.mse_loss(output["reconstruction"], x)
    mu, logvar = output["mu"], output["logvar"]
    kl_loss = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=-1)).mean()
    return {
        "loss": ce_loss + reconstruction_weight * reconstruction_loss + beta * kl_loss,
        "ce_loss": ce_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
    }
