from .loss import loss as ls
import matplotlib.pyplot as plt
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
import torch

def train(model, dloader, optimizer, epochs=3, lr=0.001, loss=ls, device=None, save_dir="models/"):
    epoch_loss = []
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
            batch_loss = []

            for batch_idx, data in enumerate(dloader):

                data = data.to(device)

                for param in model.parameters():
                    param.grad = None

                output_batch, mu, logvar = model(data)

                recon_loss = ls(output_batch, data, mu, logvar)
                batch_loss.append(recon_loss.item())

                recon_loss.backward()
                optimizer.step()

            avg_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(avg_loss)
            progress.update(task, advance=1, epoch=epoch + 1, loss=avg_loss)

    torch.save(model, save_dir + "model.pth")

    plt.plot(epoch_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.text(1, 1, "Hello")
    plt.grid()
    plt.show()
