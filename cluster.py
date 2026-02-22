"""Cluster kick drum latent space and visualize in TensorBoard."""

import torch
import torchaudio
from sklearn.decomposition import PCA

from kicks import KickDataset, KickDataloader, VAE
from kicks.cluster import (
    extract_latents,
    select_n_clusters,
    fit_gmm,
    compute_descriptors,
    write_embedding,
)
from kicks.model import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS

# ── Dataset ────────────────────────────────────────────────
dataset = KickDataset("data/kicks")
dataloader = KickDataloader(dataset, batch_size=32, shuffle=False)
print(f"Dataset: {len(dataset)} samples")

# ── Load trained model ─────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = VAE(latent_dim=16)
checkpoint = torch.load("models/best.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)
print(f"Loaded checkpoint (epoch {checkpoint['epoch']})")

# ── Extract latents ────────────────────────────────────────
latents, spectrograms = extract_latents(model, dataloader, device)
print(f"Latents: {latents.shape}")

# ── GMM clustering (BIC selection) ─────────────────────────
best_k, bics = select_n_clusters(latents, max_k=10)
print(f"BIC selected k={best_k}")

gmm, cluster_labels, cluster_probs = fit_gmm(latents, best_k)
for k in range(best_k):
    count = (cluster_labels == k).sum()
    print(f"  Cluster {k}: {count} samples")

# ── PCA inspection ─────────────────────────────────────────
pca = PCA(n_components=min(3, latents.shape[1]))
latents_pca = pca.fit_transform(latents)
print(f"PCA variance ratio: {pca.explained_variance_ratio_}")

# ── Audio descriptors ──────────────────────────────────────
descriptors = [compute_descriptors(s) for s in spectrograms]

# ── Griffin-Lim pipeline (spec → audio) ────────────────────
_mel_fb = torchaudio.functional.melscale_fbanks(
    n_freqs=N_FFT // 2 + 1,
    f_min=0.0,
    f_max=SAMPLE_RATE / 2.0,
    n_mels=N_MELS,
    sample_rate=SAMPLE_RATE,
)
_mel_fb_pinv = torch.linalg.pinv(_mel_fb.T)

griffin_lim = torchaudio.transforms.GriffinLim(
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_iter=64,
)


def spec_to_audio(spec_normalized: torch.Tensor) -> torch.Tensor:
    """Convert normalized spectrogram to audio waveform."""
    log_mel = dataset.denormalize(spec_normalized.cpu())
    mel = torch.exp(log_mel)
    mel = mel.squeeze(1)
    linear = torch.clamp(_mel_fb_pinv @ mel, min=0.0)
    waveform = griffin_lim(linear)
    waveform = torchaudio.functional.highpass_biquad(waveform, SAMPLE_RATE, cutoff_freq=30.0)
    return waveform


# ── Write to TensorBoard ──────────────────────────────────
log_dir = write_embedding(
    latents, spectrograms, cluster_labels, descriptors,
    audio_fn=spec_to_audio,
    log_dir="runs/kick_clusters",
)

print(f"\nDone! Run: tensorboard --logdir={log_dir}")
print("  Projector tab → latent space with cluster labels & descriptor metadata")
print("  Audio tab     → listen to each sample (keyed by index)")
