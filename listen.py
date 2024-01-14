from IPython.display import Audio, display
from torch.nn.functional import normalize
from modules import KickDataset
import random


dataset = KickDataset("data/kicks/")
audio = Audio(dataset.get_path(random.randint(0, len(dataset))), rate=44100, autoplay=True)

display(audio)
