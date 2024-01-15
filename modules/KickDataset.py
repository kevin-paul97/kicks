import torchaudio
import torch
from torch.utils.data import Dataset
import os

class KickDataset(Dataset):
    def __init__(self, dir) -> None:
        self.dir = dir
        self.tensors = []
        self.paths = []
        self.max_length = 0
        self.max_length_path = None
        self._process_folder(dir)
        self.tensors = self.pad()
    

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]

    def _process_folder(self, dir):
        for file in os.listdir(dir):
            if file.endswith(".wav"):
                path = os.path.join(self.dir, file)
                audio, sr = torchaudio.load(path) # type: ignore
                if sr != 44100:
                    continue
                if audio.ndim > 1:
                    audio = torch.mean(audio, dim=0)  # Convert to mono
                self.paths.append(path)
                self.tensors.append(audio)
                if audio.shape[0] > self.max_length:
                    self.max_length = audio.shape[0]
                    self.max_length_path = path


    def pad(self):
        if self.max_length % 2 != 0:
            return [torch.nn.functional.pad(audio, (0, self.max_length - audio.shape[0])) for audio in self.tensors]
        else:
            return [torch.nn.functional.pad(audio, (0, self.max_length - audio.shape[0] + (self.max_length - audio.shape[0]) % 2)) for audio in self.tensors]

    def get_path(self, idx):
        return self.paths[idx] 
