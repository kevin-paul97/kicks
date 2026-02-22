"""Compare latent dimensions (8, 16, 32) via PCA explained variance."""

import os

import torch
from sklearn.decomposition import PCA
from torch import optim

from kicks import KickDataset, KickDataloader, VAE
from kicks.cluster import extract_latents
from kicks.train import train

os.makedirs("models", exist_ok=True)

# Dataset (loaded once, shared across experiments)
dataset = KickDataset("data/kicks")
print(f"Dataset: {len(dataset)} samples")

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

results = {}

for latent_dim in [8, 16, 32]:
    print(f"\n{'='*60}")
    print(f"Training latent_dim={latent_dim}")
    print(f"{'='*60}")

    model = VAE(latent_dim=latent_dim)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataloader = KickDataloader(dataset, batch_size=32, shuffle=True)

    save_dir = f"models/latent{latent_dim}_"
    train(model, dataloader, optimizer, epochs=500, device=device,
          save_dir=save_dir, beta=4, beta_anneal_epochs=100)

    # Extract latents and run PCA
    eval_loader = KickDataloader(dataset, batch_size=32, shuffle=False)
    latents, _ = extract_latents(model, eval_loader, device)

    pca = PCA(n_components=3)
    pca.fit(latents)
    var_ratio = pca.explained_variance_ratio_
    total_var = sum(var_ratio)
    results[latent_dim] = var_ratio

    print(f"PCA explained variance (3 PCs): {var_ratio}")
    print(f"Total: {total_var:.3f} {'<-- > 60%' if total_var > 0.6 else ''}")

# Summary
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
for ld, vr in results.items():
    total = sum(vr)
    marker = " <-- best" if total > 0.6 else ""
    print(f"  latent_dim={ld:2d}: 3-PC variance = {total:.3f}{marker}")
