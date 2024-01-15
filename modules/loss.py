import torch.nn.functional as F
import torch

def loss(reconstructed_x, original_x, mu, logvar, beta=1):
    # MSE as the reconstruction loss
    recon_loss = F.mse_loss(reconstructed_x, original_x)

    # KL divergence loss
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_div
