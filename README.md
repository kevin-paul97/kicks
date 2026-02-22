# kicks

A Variational Autoencoder (VAE) for synthesizing kick drum sounds using log-mel spectrograms.

Treats spectrograms as small images (128x128) and uses a 2D Conv VAE with beta-annealing. Audio is reconstructed via Griffin-Lim.

## Setup

Requires Python 3.10+. Runs on Apple Silicon (MPS), CUDA, or CPU.

```bash
pip install torch torchaudio rich matplotlib scikit-learn tensorboard
```

Place `.wav` kick samples in `data/kicks/`.

## Usage

### Train + generate

```bash
python main.py
```

Trains for 500 epochs with beta-annealing (beta ramps 0 to 4 over 100 epochs). Monitors reconstruction loss (MSE) and KL divergence separately. Saves `models/best.pth` (best validation loss) and `models/checkpoint.pth` (final). Generates 20 reconstructions and 10 latent samples in `output/`.

### Cluster + visualize

```bash
python cluster.py
```

Loads the best checkpoint, clusters the latent space with a GMM (component count selected via BIC), and writes results to TensorBoard:

```bash
tensorboard --logdir=runs/kick_clusters
```

- **Projector** tab -- latent space visualization (color by cluster, sub, punch, click, brightness, decay)
- **Audio** tab -- listen to each sample, keyed by index

### Latent dimension experiment

```bash
python experiment_latent.py
```

Trains three models with latent_dim 8, 16, and 32. Runs PCA on each and reports explained variance ratio for the top 3 components. Pick the dimension where 3 PCs explain 60%+ variance.

## Model

2D Convolutional VAE (~596K parameters at latent_dim=16).

- **Encoder**: 4 conv layers (1 -> 16 -> 32 -> 64 -> 128, stride 2, BatchNorm + ReLU), flatten, FC to mu/logvar
- **Decoder**: FC from latent, reshape, 4 transposed conv layers (mirror of encoder), Sigmoid output
- **Loss**: MSE reconstruction + beta * KL divergence (sum-reduced, per sample)
- **Training**: 10% validation split, best checkpoint saved by validation loss

## Project structure

```
kicks/
├── main.py              # Entry point: train + generate (Griffin-Lim)
├── cluster.py           # Entry point: GMM clustering + TensorBoard projector
├── experiment_latent.py # Entry point: latent dim comparison (8, 16, 32)
├── listen.py            # Jupyter utility to preview samples
├── kicks/               # Core package
│   ├── model.py         # 2D Conv VAE
│   ├── dataset.py       # Load audio -> normalized log-mel spectrograms
│   ├── dataloader.py    # DataLoader wrapper
│   ├── train.py         # Training loop with val split + best checkpoint
│   ├── loss.py          # MSE + beta * KL loss (returns components separately)
│   └── cluster.py       # GMM clustering, descriptors, TensorBoard embedding
├── data/kicks/          # Input samples (not tracked)
├── models/              # Saved checkpoints (not tracked)
├── output/              # Generated kicks (not tracked)
└── runs/                # TensorBoard logs (not tracked)
```

SUB X
DECAY Y