"""Dataset for loading kick drum audio as normalized log-mel spectrograms."""

import os

import torch
import torchaudio
from torch.utils.data import Dataset

from .model import SAMPLE_RATE, AUDIO_LENGTH, N_FFT, HOP_LENGTH, N_MELS


class KickDataset(Dataset):
    """Loads .wav kick samples and converts to normalized log-mel spectrograms.

    Two-pass loading: first compute all spectrograms to find global min/max,
    then normalize to [0, 1]. Returns tensors of shape (1, 128, 256).
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

        self._global_min: float = float("inf")
        self._global_max: float = float("-inf")

        # Pass 1: load audio, compute log-mel spectrograms
        raw_specs: list[torch.Tensor] = []
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

            # Compute log-mel spectrogram and truncate/pad to 256 frames
            mel = self._mel_transform(audio)  # (1, N_MELS, T)
            if mel.shape[-1] > 256:
                mel = mel[:, :, :256]
            elif mel.shape[-1] < 256:
                mel = torch.nn.functional.pad(mel, (0, 256 - mel.shape[-1]))
            log_mel = torch.log(mel + 1e-7)

            self._global_min = min(self._global_min, log_mel.min().item())
            self._global_max = max(self._global_max, log_mel.max().item())

            raw_specs.append(log_mel)
            self.paths.append(path)

        # Pass 2: normalize to [0, 1]
        for spec in raw_specs:
            normalized = (spec - self._global_min) / (self._global_max - self._global_min)
            self.tensors.append(normalized)

        print(f"Loaded {len(self.tensors)} samples, spectrogram shape: {self.tensors[0].shape}")

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensors[idx]

    def denormalize(self, spec: torch.Tensor) -> torch.Tensor:
        """Convert [0,1] normalized spectrogram back to log-mel scale."""
        return spec * (self._global_max - self._global_min) + self._global_min
