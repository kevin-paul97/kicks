from modules import KickDataset
from modules import KickDataloader
from modules import VAE
from torchinfo import summary

dataset = KickDataset("data/kicks")
dataloader = KickDataloader(dataset, batch_size=1)

data = iter(dataloader)
batch = next(data)

print("max lenght: ", dataset.max_length)
print("longest sample: ", dataset.max_length_path)
model = VAE(dataset.max_length, dataset.max_length // 2, dataset.max_length // 3)

