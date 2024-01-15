from .loss import loss as ls
import matplotlib.pyplot as plt
from rich.progress import track
import torch

def train(model, dloader, optimizer, epochs=3, lr=0.001, loss=ls, device="cpu", save_dir="models/"):
    epoch_loss = []

    model.to(device)
    model.train()

    for epoch in track(range(epochs), total=epochs):
        batch_loss = []

        for batch_idx, data in enumerate(dloader):

            data.to(device)

            for param in model.parameters():
                param.grad = None

            output_batch, mu, logvar = model(data)

            recon_loss = ls(output_batch, data, mu, logvar)
            batch_loss.append(recon_loss.item())

            recon_loss.backward()
            optimizer.step()
            
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    torch.save(model, save_dir + "model.pth")

    plt.plot(epoch_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.text(1, 1, "Hello")
    plt.grid()
    plt.show()
