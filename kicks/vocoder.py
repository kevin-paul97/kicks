"""BigVGAN neural vocoder for mel spectrogram to audio conversion."""

import os

import torch
import torchaudio
import bigvgan

from .model import SAMPLE_RATE

BIGVGAN_MODEL = "nvidia/bigvgan_v2_44khz_128band_256x"
FINETUNED_PATH = "models/vocoder/best.pth"


def load_vocoder(device: torch.device) -> bigvgan.BigVGAN:
    """Load BigVGAN vocoder. Uses fine-tuned weights if available, otherwise pretrained."""
    model = bigvgan.BigVGAN.from_pretrained(BIGVGAN_MODEL, use_cuda_kernel=False)

    if os.path.exists(FINETUNED_PATH):
        checkpoint = torch.load(FINETUNED_PATH, map_location=device)
        model.load_state_dict(checkpoint["generator"])
        print(f"Loaded fine-tuned vocoder (epoch {checkpoint['epoch']})")
    else:
        print("Using pretrained BigVGAN (run finetune_vocoder.py to improve)")

    model.remove_weight_norm()
    return model.eval().to(device)


def spec_to_audio(
    spec_normalized: torch.Tensor,
    dataset,
    vocoder: bigvgan.BigVGAN,
    device: torch.device,
) -> torch.Tensor:
    """Convert normalized spectrogram to audio waveform via BigVGAN.

    Args:
        spec_normalized: VAE output, shape (B, 1, 128, 256), values in [0, 1].
        dataset: KickDataset with denormalize() method.
        vocoder: Loaded BigVGAN model.
        device: Torch device.

    Returns:
        Waveform tensor, shape (B, T).
    """
    log_mel = dataset.denormalize(spec_normalized.cpu())
    log_mel = log_mel.squeeze(1)  # (B, 128, 256)
    with torch.no_grad():
        waveform = vocoder(log_mel.to(device))  # (B, 1, T)
    waveform = waveform.squeeze(1).cpu()  # (B, T)
    waveform = torchaudio.functional.highpass_biquad(waveform, SAMPLE_RATE, cutoff_freq=25.0)
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform
