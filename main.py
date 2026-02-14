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
train(model, dataloader, optimizer, epochs=3, device=device)

with torch.no_grad():
    data_iter = iter(dataset)
    batch = next(data_iter).to(device)
    output = model(batch)
    output = output[0].detach().cpu()
    if output.ndim == 1:
        output = output.unsqueeze(0)
    save("output/out.wav", output, 44100, channels_first=True)
