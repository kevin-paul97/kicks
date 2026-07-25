"""Training loop for the kick drum VAE."""

import os

import matplotlib.pyplot as plt
import torch
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split

from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from .loss import loss as loss_fn
from .model import VAE


def _corpus_descriptor_stats(dataset) -> tuple["np.ndarray", "np.ndarray"]:  # noqa: F821
    """Per-descriptor mean/std over the training corpus (proxy-score reference)."""
    import numpy as np

    from .cluster import compute_descriptors

    rows = [list(compute_descriptors(dataset[i]).values()) for i in range(len(dataset))]
    X = np.asarray(rows)
    return X.mean(axis=0), X.std(axis=0) + 1e-8


def _eval_proxy_score(
    model: VAE,
    mu_all: torch.Tensor,
    desc_mean,
    desc_std,
    device: torch.device,
    n_samples: int = 32,
) -> float:
    """Spectrogram-domain generative realism proxy (no vocoder needed).

    Samples latents from a Gaussian fit to the val-set posterior means,
    decodes them, and measures how far the decoded kicks' perceptual
    descriptors sit from the corpus distribution (mean |z-score|; lower is
    better). Tracks what `kicks eval` measures well enough for checkpoint
    selection, at a tiny fraction of the cost.
    """
    import numpy as np

    from .cluster import compute_descriptors

    mean = mu_all.mean(dim=0)
    cov = torch.cov(mu_all.T) + 1e-4 * torch.eye(mu_all.shape[1], device=mu_all.device)
    chol = torch.linalg.cholesky(cov)
    eps = torch.randn(n_samples, mu_all.shape[1], device=mu_all.device)
    z = mean + eps @ chol.T
    with torch.no_grad():
        specs = model.decode(z.to(device)).cpu()
    scores = []
    for i in range(n_samples):
        d = np.asarray(list(compute_descriptors(specs[i]).values()))
        scores.append(np.abs((d - desc_mean) / desc_std).mean())
    return float(np.mean(scores))


def train(
    model: VAE,
    dloader: DataLoader,
    optimizer: Optimizer,
    epochs: int = 500,
    device: torch.device | None = None,
    save_dir: str = "models/",
    beta: float = 0.01,
    free_bits: float = 0.5,
    beta_anneal_epochs: int = 0,
    beta_cycles: int = 4,
    val_split: float = 0.1,
    scheduler: LRScheduler | None = None,
    hf_weight: float = 0.5,
    eval_every: int = 5,
) -> dict[str, list[float]]:
    """Train the VAE. Returns per-epoch average losses for loss, recon, kl.

    Beta annealing uses a cyclical schedule: beta ramps linearly from 0 to the
    target value over (beta_anneal_epochs / beta_cycles) epochs, then repeats.
    This prevents posterior collapse while maintaining reconstruction quality.

    Alongside the val-loss best checkpoint (vae_best.pth), every `eval_every`
    epochs a generative eval-proxy score is computed (decode latents sampled
    from the val posterior, measure descriptor realism vs the corpus) and the
    best-scoring model is saved to vae_best_eval.pth — checkpoint selection
    aligned with what `kicks eval` measures rather than pixel loss.
    """
    epoch_loss: list[float] = []
    epoch_recon: list[float] = []
    epoch_kl: list[float] = []
    model.to(device)

    # Train/val split
    dataset = dloader.dataset
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=dloader.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=dloader.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_proxy = float("inf")
    desc_mean, desc_std = _corpus_descriptor_stats(dataset)

    with Progress(
        TextColumn("[bold blue]Epoch {task.fields[epoch]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("Loss: {task.fields[loss]:.4f}  Recon: {task.fields[recon]:.4f}  KL: {task.fields[kl]:.4f}"),
    ) as progress:
        task = progress.add_task("Training", total=epochs, epoch=0, loss=0.0, recon=0.0, kl=0.0)

        for epoch in range(epochs):
            if beta_anneal_epochs > 0 and epoch < beta_anneal_epochs:
                cycle_len = beta_anneal_epochs / beta_cycles
                cycle_pos = (epoch % cycle_len) / cycle_len
                current_beta = beta * min(1.0, cycle_pos * 2)  # ramp up in first half, hold in second
            else:
                current_beta = beta

            # Training
            model.train()
            batch_loss: list[float] = []
            batch_recon: list[float] = []
            batch_kl: list[float] = []

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(data)
                l, recon_l, kl = loss_fn(recon, data, mu, logvar, beta=current_beta, free_bits=free_bits, hf_weight=hf_weight)
                batch_loss.append(l.item())
                batch_recon.append(recon_l.item())
                batch_kl.append(kl.item())
                l.backward()
                optimizer.step()

            avg_loss = sum(batch_loss) / len(batch_loss) if batch_loss else 0.0
            avg_recon = sum(batch_recon) / len(batch_recon) if batch_recon else 0.0
            avg_kl = sum(batch_kl) / len(batch_kl) if batch_kl else 0.0
            epoch_loss.append(avg_loss)
            epoch_recon.append(avg_recon)
            epoch_kl.append(avg_kl)
            if scheduler is not None:
                scheduler.step()
            progress.update(task, advance=1, epoch=epoch + 1, loss=avg_loss, recon=avg_recon, kl=avg_kl)

            # Validation
            model.eval()
            val_losses: list[float] = []
            mus: list[torch.Tensor] = []
            logvars: list[torch.Tensor] = []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    recon, mu, logvar = model(data)
                    vl, _, _ = loss_fn(recon, data, mu, logvar, beta=current_beta, free_bits=free_bits, hf_weight=hf_weight)
                    val_losses.append(vl.item())
                    mus.append(mu)
                    logvars.append(logvar)
            val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

            # Latent diagnostics: per-dim KL over the val set tells us how many
            # latent dimensions are actually carrying information (active) vs.
            # collapsed to the prior, and whether free_bits is masking collapse.
            if mus:
                mu_all = torch.cat(mus, dim=0)
                logvar_all = torch.cat(logvars, dim=0)
                kl_per_dim = -0.5 * (1 + logvar_all - mu_all.pow(2) - logvar_all.exp()).mean(0)
                active_dims = int((kl_per_dim > 0.01).sum().item())
                raw_kl = float(kl_per_dim.sum().item())
                mean_kl = float(kl_per_dim.mean().item())
                progress.console.log(
                    f"epoch {epoch + 1}: val_loss={val_loss:.4f}  beta={current_beta:.4f}  "
                    f"active_dims={active_dims}/{model.latent_dim}  raw_kl={raw_kl:.3f}  mean_kl/dim={mean_kl:.3f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "latent_dim": model.latent_dim,
                }, save_dir + "vae_best.pth")

            # Generative eval proxy: descriptor realism of decoded samples
            if mus and eval_every > 0 and (epoch + 1) % eval_every == 0:
                proxy = _eval_proxy_score(
                    model, mu_all.cpu(), desc_mean, desc_std, device,
                )
                marker = ""
                if proxy < best_proxy:
                    best_proxy = proxy
                    torch.save({
                        "model": model.state_dict(),
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "eval_proxy": proxy,
                        "latent_dim": model.latent_dim,
                    }, save_dir + "vae_best_eval.pth")
                    marker = "  (new best → vae_best_eval.pth)"
                progress.console.log(
                    f"epoch {epoch + 1}: eval_proxy={proxy:.3f}{marker}"
                )

    # Save final checkpoint
    torch.save({
        "model": model.state_dict(),
        "epoch": epochs,
        "loss_history": epoch_loss,
        "latent_dim": model.latent_dim,
    }, save_dir + "vae_checkpoint.pth")

    # Plot loss components
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax1.plot(epoch_loss)
    ax1.set_title("Total Loss")
    ax1.set_xlabel("Epoch")
    ax1.grid()
    ax2.plot(epoch_recon)
    ax2.set_title("Reconstruction (SC + L1)")
    ax2.set_xlabel("Epoch")
    ax2.grid()
    ax3.plot(epoch_kl)
    ax3.set_title("KL Divergence")
    ax3.set_xlabel("Epoch")
    ax3.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"))

    return {"loss": epoch_loss, "recon": epoch_recon, "kl": epoch_kl}
