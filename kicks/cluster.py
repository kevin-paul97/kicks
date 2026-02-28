"""Latent space clustering utilities for kick VAE."""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


def extract_latents(
    model, dataloader, device: torch.device
):
    """Extract latent mu vectors and spectrograms from the trained model."""
    model.eval()
    latents = []
    spectrograms = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            latents.append(mu.cpu().numpy())
            spectrograms.append(batch.cpu())
    return np.concatenate(latents), torch.cat(spectrograms)


def select_n_clusters(latents: np.ndarray, max_k: int = 10) -> tuple[int, list[float]]:
    """Select optimal GMM component count via BIC (lower is better)."""
    max_k = min(max_k, max(2, len(latents) // 3))
    k_range = range(2, max_k + 1)
    bics = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=3)
        gmm.fit(latents)
        bics.append(gmm.bic(latents))
    best_k = list(k_range)[int(np.argmin(bics))]
    return best_k, list(bics)


def fit_gmm(
    latents: np.ndarray, n_clusters: int
) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """Fit GMM and return model, hard labels, and soft probabilities."""
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type="full", random_state=42, n_init=3
    )
    gmm.fit(latents)
    labels = gmm.predict(latents)
    probs = gmm.predict_proba(latents)
    return gmm, labels, probs


def compute_descriptors(spec_tensor: torch.Tensor) -> dict[str, float]:
    """Derive perceptual descriptors from a normalized log-mel spectrogram."""
    spec = spec_tensor.squeeze().numpy()

    sub_energy = spec[:10, :].mean()
    mid_energy = spec[20:50, :].mean()
    attack_slope = np.diff(spec.mean(axis=0)[:5]).mean()
    punchiness = mid_energy + attack_slope
    click = spec[80:, :5].mean()
    brightness = spec[64:, :].mean()

    envelope = spec.mean(axis=0)
    threshold = envelope.max() * 0.1
    above = np.where(envelope > threshold)[0]
    n_frames = spec.shape[1]
    decay_length = (above[-1] - above[0]) / n_frames if len(above) > 1 else 0.0

    return {
        "sub": float(sub_energy),
        "punch": float(punchiness),
        "click": float(click),
        "bright": float(brightness),
        "decay": float(decay_length),
    }
