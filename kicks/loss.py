"""VAE loss: MSE reconstruction + KL divergence."""

import torch
import torch.nn.functional as F


def loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.001,
) -> torch.Tensor:
    """Beta-VAE loss: MSE + beta * KL."""
    mse = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kl
