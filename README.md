# kicks

A Variational Autoencoder (VAE) for synthesizing kick drum sounds using log-mel spectrograms.

Treats spectrograms as small images (128x128) and uses a 2D Conv VAE. Audio is reconstructed via Griffin-Lim.

## Setup

Requires Python 3.10+ and an MPS-capable Mac (Apple Silicon).

```bash
pip install torch torchaudio rich matplotlib scikit-learn tensorboard
```

Place `.wav` kick samples in `data/kicks/`.

## Usage

### Train + generate

```bash
python main.py
```

Trains for 500 epochs, saves checkpoint to `models/`, generates 10 reconstructions and 10 latent samples in `output/`.

### Cluster + visualize

```bash
python cluster.py
```

Loads a trained checkpoint, clusters the latent space with a GMM (component count selected via BIC), and writes results to TensorBoard:

```bash
tensorboard --logdir=runs/kick_clusters
```

- **Projector** tab — latent space visualization (color by cluster, sub, punch, click, brightness, decay)
- **Audio** tab — listen to each sample, keyed by index

## Project Structure

```
kicks/
├── main.py              # Entry point: train + generate (Griffin-Lim)
├── cluster.py           # Entry point: GMM clustering + TensorBoard projector
├── listen.py            # Jupyter utility to preview samples
├── kicks/               # Core package
│   ├── model.py         # 2D Conv VAE (~500K params)
│   ├── dataset.py       # Load audio → normalized log-mel spectrograms
│   ├── dataloader.py    # DataLoader wrapper
│   ├── train.py         # Training loop
│   ├── loss.py          # MSE + KL loss
│   └── cluster.py       # GMM clustering, descriptors, TensorBoard embedding
├── data/kicks/          # Input samples (not tracked)
├── models/              # Saved checkpoints (not tracked)
└── output/              # Generated kicks (not tracked)
```
