import os

import torch
import torchaudio
from torch import optim

from kicks import KickDataset, KickDataloader, VAE
from kicks.model import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS
from kicks.train import train

os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Dataset
dataset = KickDataset("data/kicks")
dataloader = KickDataloader(dataset, batch_size=32, shuffle=True)
print(f"Dataset: {len(dataset)} samples")

# Model
device = torch.device("mps")
model = VAE(latent_dim=32)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

param_count = sum(p.numel() for p in model.parameters())
print(f"Model: {param_count:,} parameters, latent_dim=32")

# Train
train(model, dataloader, optimizer, epochs=200, device=device, beta=0.001)

# Griffin-Lim pipeline for spectrogram → audio
# Build mel filterbank and compute pseudo-inverse for mel → linear conversion
_mel_fb = torchaudio.functional.melscale_fbanks(
    n_freqs=N_FFT // 2 + 1,
    f_min=0.0,
    f_max=SAMPLE_RATE / 2.0,
    n_mels=N_MELS,
    sample_rate=SAMPLE_RATE,
)  # (n_stft, n_mels)
_mel_fb_pinv = torch.linalg.pinv(_mel_fb.T)  # (n_stft, n_mels)

griffin_lim = torchaudio.transforms.GriffinLim(
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
)


def spec_to_audio(spec_normalized: torch.Tensor) -> torch.Tensor:
    """Convert normalized spectrogram (B, 1, 128, 64) to audio waveform."""
    log_mel = dataset.denormalize(spec_normalized.cpu())
    mel = torch.exp(log_mel)  # undo log
    mel = mel.squeeze(1)  # (B, N_MELS, T)
    # mel → linear via pseudo-inverse of mel filterbank
    linear = torch.clamp(_mel_fb_pinv @ mel, min=0.0)  # (B, n_stft, T)
    waveform = griffin_lim(linear)  # (B, T')
    return waveform


# Generate outputs
with torch.no_grad():
    model.eval()

    # Reconstructions
    for i in range(min(10, len(dataset))):
        original = dataset[i].unsqueeze(0).to(device)
        recon, _, _ = model(original)
        audio = spec_to_audio(recon)
        torchaudio.save(f"output/recon_{i+1}.wav", audio, SAMPLE_RATE)
        print(f"Saved output/recon_{i+1}.wav")

    # Generated from latent space
    for i in range(10):
        z = torch.randn(1, 32).to(device)
        spec = model.decode(z)
        audio = spec_to_audio(spec)
        torchaudio.save(f"output/gen_{i+1}.wav", audio, SAMPLE_RATE)
        print(f"Saved output/gen_{i+1}.wav")

print("Done!")
