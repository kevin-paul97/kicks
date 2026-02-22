import os

import torch
import torchaudio
from torch import optim

from kicks import KickDataset, KickDataloader, VAE
from kicks.model import SAMPLE_RATE
from kicks.train import train
from kicks.vocoder import load_vocoder, spec_to_audio

os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Dataset
dataset = KickDataset("data/kicks")
dataloader = KickDataloader(dataset, batch_size=32, shuffle=True)
print(f"Dataset: {len(dataset)} samples")

# Model
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = VAE(latent_dim=16)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

param_count = sum(p.numel() for p in model.parameters())
print(f"Model: {param_count:,} parameters, latent_dim=16, device={device}")

# Train
train(model, dataloader, optimizer, epochs=500, device=device, beta=4, beta_anneal_epochs=100)

# BigVGAN vocoder for spectrogram → audio
vocoder = load_vocoder(device)

# Generate outputs
with torch.no_grad():
    model.eval()

    # Reconstructions
    for i in range(min(20, len(dataset))):
        original = dataset[i].unsqueeze(0).to(device)
        recon, _, _ = model(original)
        audio = spec_to_audio(recon, dataset, vocoder, device)
        torchaudio.save(f"output/recon_{i+1}.wav", audio, SAMPLE_RATE)
        print(f"Saved output/recon_{i+1}.wav")

    # Generated from latent space
    for i in range(10):
        z = torch.randn(1, 16).to(device)
        spec = model.decode(z)
        audio = spec_to_audio(spec, dataset, vocoder, device)
        torchaudio.save(f"output/gen_{i+1}.wav", audio, SAMPLE_RATE)
        print(f"Saved output/gen_{i+1}.wav")

print("Done!")
