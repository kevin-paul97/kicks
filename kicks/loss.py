"""VAE loss: spectral convergence + L1 reconstruction + KL divergence."""

import torch
import torch.nn.functional as F


def spectral_convergence(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spectral convergence loss: Frobenius norm ratio."""
    return torch.norm(target - recon, p="fro") / (torch.norm(target, p="fro") + 1e-8)


def loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Beta-VAE loss: (spectral convergence + L1) + beta * KL.

    Returns (total_loss, recon_loss, kl) for separate logging.
    """
    batch_size = x.size(0)
    sc = spectral_convergence(recon, x)
    lm = F.l1_loss(recon, x)
    recon_loss = sc + lm
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon_loss + beta * kl, recon_loss, kl
