"""Training loop for the kick drum VAE."""

import matplotlib.pyplot as plt
import torch
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .loss import loss as loss_fn
from .model import VAE


def train(
    model: VAE,
    dloader: DataLoader,
    optimizer: Optimizer,
    epochs: int = 500,
    device: torch.device | None = None,
    save_dir: str = "models/",
    beta: float = 0.001,
) -> list[float]:
    """Train the VAE. Returns per-epoch average losses."""
    epoch_loss: list[float] = []
    model.train()
    model.to(device)

    with Progress(
        TextColumn("[bold blue]Epoch {task.fields[epoch]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("Loss: {task.fields[loss]:.6f}"),
    ) as progress:
        task = progress.add_task("Training", total=epochs, epoch=0, loss=0.0)

        for epoch in range(epochs):
            batch_loss: list[float] = []

            for data in dloader:
                data = data.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(data)
                l = loss_fn(recon, data, mu, logvar, beta=beta)
                batch_loss.append(l.item())
                l.backward()
                optimizer.step()

            avg_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(avg_loss)
            progress.update(task, advance=1, epoch=epoch + 1, loss=avg_loss)

    torch.save({
        "model": model.state_dict(),
        "epoch": epochs,
        "loss_history": epoch_loss,
    }, save_dir + "checkpoint.pth")

    plt.plot(epoch_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.grid()
    plt.show()

    return epoch_loss
