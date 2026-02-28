"""Dataset for loading kick drum audio as normalized log-mel spectrograms."""

import os

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
from torch.utils.data import Dataset

from .model import SAMPLE_RATE, AUDIO_LENGTH, N_FFT, HOP_LENGTH, N_MELS

# Fixed dB bounds for normalization — independent of dataset content.
# These stay constant so checkpoints remain valid across dataset changes.
LOG_MEL_MIN = -80.0  # dB floor (silence)
LOG_MEL_MAX = 0.0    # dB ceiling

# Target integrated loudness for LUFS normalization
TARGET_LUFS = -14.0


class KickDataset(Dataset):
    """Loads .wav kick samples and converts to normalized log-mel spectrograms.

    Pre-processing: LUFS loudness normalization of input audio.
    Normalization: fixed dB bounds [-80, 0] mapped to [0, 1].
    Returns tensors of shape (1, 128, 256).
    """

    def __init__(self, dir: str) -> None:
        self.dir = dir
        self.tensors: list[torch.Tensor] = []
        self.paths: list[str] = []

        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )

        self._lufs_meter = pyln.Meter(SAMPLE_RATE)

        for file in sorted(os.listdir(dir)):
            if not file.endswith(".wav"):
                continue
            path = os.path.join(dir, file)
            audio, sr = torchaudio.load(path)  # type: ignore

            # Convert to mono
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Resample to target sample rate
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                audio = resampler(audio)

            # Pad/truncate to fixed length
            length = audio.shape[-1]
            if length > AUDIO_LENGTH:
                audio = audio[:, :AUDIO_LENGTH]
            elif length < AUDIO_LENGTH:
                audio = torch.nn.functional.pad(audio, (0, AUDIO_LENGTH - length))

            # LUFS loudness normalization
            audio_np = audio.squeeze(0).numpy()
            loudness = self._lufs_meter.integrated_loudness(audio_np)
            if np.isfinite(loudness):
                audio_np = pyln.normalize.loudness(audio_np, loudness, TARGET_LUFS)
                audio_np = np.clip(audio_np, -1.0, 1.0)
                audio = torch.from_numpy(audio_np).unsqueeze(0).float()

            # Compute log-mel spectrogram and truncate/pad to 256 frames
            mel = self._mel_transform(audio)  # (1, N_MELS, T)
            if mel.shape[-1] > 256:
                mel = mel[:, :, :256]
            elif mel.shape[-1] < 256:
                mel = torch.nn.functional.pad(mel, (0, 256 - mel.shape[-1]))
            log_mel = torch.log(mel + 1e-7)

            # Normalize to [0, 1] using fixed dB bounds
            log_mel = torch.clamp(log_mel, min=LOG_MEL_MIN, max=LOG_MEL_MAX)
            normalized = (log_mel - LOG_MEL_MIN) / (LOG_MEL_MAX - LOG_MEL_MIN)

            self.tensors.append(normalized)
            self.paths.append(path)

        print(f"Loaded {len(self.tensors)} samples, spectrogram shape: {self.tensors[0].shape}")

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensors[idx]

    @staticmethod
    def denormalize(spec: torch.Tensor) -> torch.Tensor:
        """Convert [0,1] normalized spectrogram back to log-mel scale."""
        return spec * (LOG_MEL_MAX - LOG_MEL_MIN) + LOG_MEL_MIN
