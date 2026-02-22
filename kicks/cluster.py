"""Latent space clustering and TensorBoard visualization for kick VAE."""

import os

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter

from .model import VAE, SAMPLE_RATE


def extract_latents(
    model: VAE, dataloader, device: torch.device
) -> tuple[np.ndarray, torch.Tensor]:
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
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm.fit(latents)
        bics.append(gmm.bic(latents))
    best_k = list(k_range)[int(np.argmin(bics))]
    return best_k, list(bics)


def fit_gmm(
    latents: np.ndarray, n_clusters: int
) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """Fit GMM and return model, hard labels, and soft probabilities."""
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type="full", random_state=42
    )
    gmm.fit(latents)
    labels = gmm.predict(latents)
    probs = gmm.predict_proba(latents)
    return gmm, labels, probs


def compute_descriptors(spec_tensor: torch.Tensor) -> dict[str, float]:
    """Derive perceptual descriptors from a normalized log-mel spectrogram."""
    spec = spec_tensor.squeeze().numpy()

    sub_energy = spec[:5, :].mean()
    mid_energy = spec[10:25, :].mean()
    attack_slope = np.diff(spec.mean(axis=0)[:5]).mean()
    punchiness = mid_energy + attack_slope
    click = spec[40:, :5].mean()
    brightness = spec[32:, :].mean()

    envelope = spec.mean(axis=0)
    threshold = envelope.max() * 0.1
    above = np.where(envelope > threshold)[0]
    decay_length = above[-1] - above[0] if len(above) > 1 else 0

    return {
        "sub": float(sub_energy),
        "punch": float(punchiness),
        "click": float(click),
        "bright": float(brightness),
        "decay": float(decay_length),
    }


def write_embedding(
    latents: np.ndarray,
    spectrograms: torch.Tensor,
    cluster_labels: np.ndarray,
    descriptors: list[dict[str, float]],
    audio_fn,
    log_dir: str = "runs/kick_clusters",
) -> str:
    """Write TensorBoard embedding with metadata, sprite thumbnails, and per-sample audio.

    Embedding and audio use separate subdirectories so the large audio
    events file doesn't interfere with the projector plugin loading.
    Audio is in the Audio tab keyed by sample index, matching the
    'index' column in the projector metadata for cross-referencing.
    """
    embed_dir = os.path.join(log_dir, "embedding")
    audio_dir = os.path.join(log_dir, "audio")

    # ── Embedding projector ────────────────────────────────
    embed_writer = SummaryWriter(embed_dir)

    metadata_header = ["index", "cluster", "sub", "punch", "click", "bright", "decay"]
    metadata = []
    for i, desc in enumerate(descriptors):
        metadata.append([
            str(i),
            str(cluster_labels[i]),
            f"{desc['sub']:.3f}",
            f"{desc['punch']:.3f}",
            f"{desc['click']:.3f}",
            f"{desc['bright']:.3f}",
            f"{desc['decay']:.1f}",
        ])

    # Sprite thumbnails from spectrograms
    thumb_size = 32
    thumbs = []
    for s in spectrograms:
        s = s.squeeze()
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        s = torch.nn.functional.interpolate(
            s.unsqueeze(0).unsqueeze(0),
            size=(thumb_size, thumb_size),
            mode="bilinear",
        ).squeeze()
        thumbs.append(s)
    label_img = torch.stack(thumbs).unsqueeze(1).repeat(1, 3, 1, 1)

    embed_writer.add_embedding(
        mat=torch.tensor(latents, dtype=torch.float32),
        metadata=metadata,
        metadata_header=metadata_header,
        label_img=label_img,
        global_step=0,
        tag="kick_latents",
    )
    embed_writer.close()

    # ── Audio per sample ───────────────────────────────────
    audio_writer = SummaryWriter(audio_dir)
    for i in range(len(spectrograms)):
        audio = audio_fn(spectrograms[i].unsqueeze(0))
        audio = audio / (audio.abs().max() + 1e-8)
        audio_writer.add_audio(f"samples/{i:03d}", audio, sample_rate=SAMPLE_RATE)
    audio_writer.close()

    return log_dir
