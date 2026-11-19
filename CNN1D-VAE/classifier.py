"""Supervised state classification using the CNN1D-VAE posterior mean.

Inputs are channel-standardized float32 windows with layout (batch, time,
channels). The optional CNC context is a batch of normalized float32 vectors
with layout (batch, context_features). The caller owns sampling rate and labels.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from model import ModelConfig, VAE


class StateClassifier(nn.Module):
    """Attach a supervised classifier to the VAE's deterministic mean code.

    ``reconstruct=False`` skips the decoder entirely. Classification always
    uses ``mu``; sampling affects only the optional reconstruction branch. When
    ``context_dim`` is positive, the context branch is fused with ``mu`` before
    the classification head. ``context_dim=0`` preserves the audio-only model.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        num_classes: int = 3,
        context_dim: int = 0,
        context_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if context_dim < 0:
            raise ValueError("context_dim must be non-negative")
        if context_dim > 0 and context_embedding_dim < 1:
            raise ValueError("context_embedding_dim must be positive when context is enabled")
        self.vae = VAE(config)
        self.config = self.vae.config
        self.num_classes = num_classes
        self.context_dim = context_dim
        self.context_embedding_dim = context_embedding_dim if context_dim else 0
        if context_dim:
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, context_embedding_dim),
                nn.GELU(),
                nn.Linear(context_embedding_dim, context_embedding_dim),
            )
        classifier_input_dim = self.config.latent_dim + self.context_embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_classes),
        )

    def forward(
        self,
        x: Tensor,
        reconstruct: bool = True,
        *,
        context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if reconstruct:
            output = self.vae(x)
        else:
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
            encoded = self.vae.encoder(x.transpose(1, 2))
            mu = self.vae.mu_head(encoded)
            logvar = self.vae.bound_logvar(self.vae.logvar_head(encoded))
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
            output = {"mu": mu, "logvar": logvar, "z": z}
        if self.context_dim:
            if context is None:
                raise ValueError(
                    f"context is required with shape (batch, {self.context_dim})"
                )
            if context.ndim != 2 or tuple(context.shape) != (x.shape[0], self.context_dim):
                raise ValueError(
                    f"Expected context shape ({x.shape[0]}, {self.context_dim}); "
                    f"received {tuple(context.shape)}"
                )
            if context.dtype != torch.float32:
                raise TypeError("Context must be a float32 tensor")
            if context.device != x.device:
                raise ValueError("Context and audio must be on the same device")
            if not torch.isfinite(context).all():
                raise ValueError("Context must contain only finite values")
            context_embedding = self.context_encoder(context)
            classifier_input = torch.cat((output["mu"], context_embedding), dim=-1)
            return {
                **output,
                "context_embedding": context_embedding,
                "logits": self.classifier(classifier_input),
            }
        if context is not None:
            raise ValueError("context was provided, but this classifier has context_dim=0")
        return {**output, "logits": self.classifier(output["mu"])}


def classification_loss(
    output: dict[str, Tensor],
    target: Tensor,
    x: Tensor | None = None,
    reconstruction_weight: float = 0.05,
    beta: float = 0.001,
    class_weights: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return CE + reconstruction_weight*MSE + beta*KL.

    Targets are integer class indices, not anomaly-score thresholds. MSE is
    zero when the forward pass omitted reconstruction. When reconstruction is
    present and its weight is positive, ``x`` must supply the signal target.
    KL sums latent coordinates and then averages windows, matching the VAE.
    """
    if reconstruction_weight < 0 or beta < 0:
        raise ValueError("reconstruction_weight and beta must be nonnegative")
    logits = output["logits"]
    if target.ndim != 1 or target.shape[0] != logits.shape[0] or target.dtype != torch.long:
        raise ValueError("target must be a torch.long vector with one class index per window")
    weight = None if class_weights is None else torch.as_tensor(
        class_weights, dtype=logits.dtype, device=logits.device
    )
    ce_loss = F.cross_entropy(logits, target, weight=weight)
    reconstruction_loss = logits.new_zeros(())
    if "reconstruction" in output:
        if x is None and reconstruction_weight > 0:
            raise ValueError("x is required for a positively weighted reconstruction loss")
        if x is not None:
            if output["reconstruction"].shape != x.shape:
                raise ValueError("Reconstruction and signal target shapes must match")
            reconstruction_loss = F.mse_loss(output["reconstruction"], x, reduction="mean")
    mu, logvar = output["mu"], output["logvar"]
    kl_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=-1).mean()
    return {
        "loss": ce_loss + reconstruction_weight * reconstruction_loss + beta * kl_loss,
        "ce_loss": ce_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
    }
