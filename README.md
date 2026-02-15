# kicks

A Variational Autoencoder (VAE) for synthesizing kick drum sounds using log-mel spectrograms.

Treats spectrograms as small images (128x64) and uses a 2D Conv VAE. Audio is reconstructed via Griffin-Lim.

## Setup

Requires Python 3.10+ and an MPS-capable Mac (Apple Silicon).

```bash
pip install torch torchaudio rich matplotlib
```

Place `.wav` kick samples in `data/kicks/`.

## Usage

```bash
python main.py
```

Trains for 500 epochs, saves checkpoint to `models/`, generates 10 reconstructions and 10 kicks in `output/`.

## Project Structure

```
kicks/
├── main.py              # Entry point: train + generate (Griffin-Lim)
├── listen.py            # Jupyter utility to preview samples
├── kicks/               # Core package
│   ├── model.py         # 2D Conv VAE (~500K params)
│   ├── dataset.py       # Load audio → normalized log-mel spectrograms
│   ├── dataloader.py    # DataLoader wrapper
│   ├── train.py         # Training loop
│   └── loss.py          # MSE + KL loss
├── data/kicks/          # Input samples (not tracked)
├── models/              # Saved checkpoints (not tracked)
└── output/              # Generated kicks (not tracked)
```
