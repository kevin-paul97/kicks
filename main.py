from torch import optim
from modules import KickDataset, KickDataloader, VAE
from modules.info import info
from modules.train import train
from torchaudio import save # type: ignore
import torch

dataset = KickDataset("data/kicks")
dataloader = KickDataloader(dataset, batch_size=10)

try:
    model = torch.load("models/model.pth")

    print("Model loaded succesfully... ")
except:
    model = VAE(dataset.max_length, 400, 100, 100)
    print("Model not found, instanciated new one... ")

info(model, dataset, dataloader)

device = torch.device("mps")

optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, dataloader, optimizer, epochs=200, device=device)

with torch.no_grad():
    model.eval()
    for i in range(10):
        z = torch.randn(1, 100).to(device)
        output = model.decode(z).detach().cpu()
        if output.ndim == 1:
            output = output.unsqueeze(0)
        save(f"output/kick_{i+1}.wav", output, 44100, channels_first=True)
        print(f"Saved output/kick_{i+1}.wav")
    print("Done! Generated 10 kick samples.")
