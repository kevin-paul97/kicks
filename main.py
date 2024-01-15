from torch import optim
from modules import KickDataset, KickDataloader, VAE
from modules.info import info
from modules.train import train
from torchaudio import save
import torch
dataset = KickDataset("data/kicks")
dataloader = KickDataloader(dataset, batch_size=10)

model = torch.load("models/model.pth")
# model = VAE(dataset.max_length, 400, 100, 100)
# info(model, dataset, dataloader)

optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, dataloader, optimizer, epochs=1)

with torch.no_grad():
    data_iter = iter(dataset)
    batch = next(data_iter)
    output = model(batch)
    output = output[0].detach().cpu()
    if output.ndim == 1:
        output = output.unsqueeze(0)
    save("output/out.wav", output, 44100, channels_first=True)
