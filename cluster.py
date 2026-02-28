"""Cluster kick drum latent space - optimized for speed."""

import json

import torch
from sklearn.decomposition import PCA

from kicks import KickDataset, KickDataloader, VAE
from kicks.cluster import (
    extract_latents,
    select_n_clusters,
    fit_gmm,
    compute_descriptors,
)
from kicks.model import SAMPLE_RATE
from kicks.vocoder import load_vocoder, spec_to_audio

dataset = KickDataset("data/kicks")
dataloader = KickDataloader(dataset, batch_size=64, shuffle=False)
print(f"Dataset: {len(dataset)} samples")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = VAE(latent_dim=32)
checkpoint = torch.load("models/best.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()
print(f"Loaded checkpoint (epoch {checkpoint['epoch']})")

print("Loading vocoder...")
vocoder = load_vocoder(device)

print("Extracting latents...")
latents, spectrograms = extract_latents(model, dataloader, device)
print(f"Latents: {latents.shape}")

print("Running GMM clustering...")
best_k, _ = select_n_clusters(latents, max_k=10)
print(f"BIC selected k={best_k}")

gmm, cluster_labels, cluster_probs = fit_gmm(latents, best_k)
for k in range(best_k):
    count = (cluster_labels == k).sum()
    print(f"  Cluster {k}: {count} samples")

print("Computing PCA...")
pca = PCA(n_components=3)
latents_pca = pca.fit_transform(latents)
print(f"PCA variance ratio: {pca.explained_variance_ratio_}")

print("Computing descriptors...")
descriptors = [compute_descriptors(s) for s in spectrograms]

print("Generating audio previews...")
audio_data = []
with torch.no_grad():
    for i in range(len(spectrograms)):
        spec = spectrograms[i].unsqueeze(0)
        audio = spec_to_audio(spec, dataset, vocoder, device)
        audio = audio / (audio.abs().max() + 1e-8)
        audio_data.append({
            "sample_idx": i,
            "cluster": int(cluster_labels[i]),
            "pc1": float(latents_pca[i, 0]),
            "pc2": float(latents_pca[i, 1]),
            "pc3": float(latents_pca[i, 2]),
            "descriptors": descriptors[i],
            "probs": cluster_probs[i].tolist(),
        })
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(spectrograms)}")

output = {
    "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
    "n_clusters": best_k,
    "samples": audio_data,
}

with open("output/cluster_analysis.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nDone! Saved to output/cluster_analysis.json")
print(f"  {len(audio_data)} samples, {best_k} clusters, 3 PCs")
