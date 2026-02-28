"""Cluster kick drum latent space - optimized for speed."""

import argparse
import json
import os

import torch
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import Subset

from kicks import KickDataset, KickDataloader, VAE
from kicks.cluster import (
    extract_latents,
    select_n_clusters,
    fit_gmm,
    compute_descriptors,
)


def main():
    parser = argparse.ArgumentParser(description="Cluster kick drum latent space")
    parser.add_argument("-n", "--samples", type=int, default=None, 
                        help="Number of random samples to cluster (default: all)")
    args = parser.parse_args()

    dataset = KickDataset("data/kicks")
    n_samples = args.samples if args.samples else len(dataset)
    n_samples = min(n_samples, len(dataset))
    
    if args.samples and args.samples < len(dataset):
        indices = np.random.choice(len(dataset), n_samples, replace=False).tolist()
        subset = Subset(dataset, indices)
    else:
        subset = dataset
    
    dataloader = KickDataloader(subset, batch_size=64, shuffle=False)
    print(f"Dataset: {len(dataset)} samples, clustering {len(subset)}")

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

    subset_indices = list(subset.indices) if hasattr(subset, 'indices') else list(range(len(subset)))
    
    print("Building output...")
    audio_data = []
    for i, idx in enumerate(subset_indices):
        sample_path = dataset.paths[idx]
        filename = os.path.basename(sample_path)
        
        audio_data.append({
            "sample_idx": i,
            "filename": filename,
            "original_path": sample_path,
            "cluster": int(cluster_labels[i]),
            "pc1": float(latents_pca[i, 0]),
            "pc2": float(latents_pca[i, 1]),
            "pc3": float(latents_pca[i, 2]),
            "descriptors": descriptors[i],
            "probs": cluster_probs[i].tolist(),
        })

    output = {
        "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
        "n_clusters": best_k,
        "samples": audio_data,
    }

    with open("output/cluster_analysis.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone! Saved to output/cluster_analysis.json")
    print(f"  {len(audio_data)} samples, {best_k} clusters, 3 PCs")


if __name__ == "__main__":
    main()
