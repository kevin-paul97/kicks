"""VAE loss: MSE reconstruction + KL divergence."""

import torch
import torch.nn.functional as F


def loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Beta-VAE loss: MSE + beta * KL (sum-reduced, per sample).

    Returns (total_loss, mse, kl) for separate logging.
    """
    batch_size = x.size(0)
    mse = F.mse_loss(recon, x, reduction='sum') / batch_size
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return mse + beta * kl, mse, kl
