# kicks

A VAE-powered kick drum synthesizer. Train a convolutional VAE on your kick samples, then generate new kicks in real-time by moving sliders in a web UI.

## How it works

```
Kick samples (.wav)
  -> Log-mel spectrograms (128x256)
  -> Beta-VAE training (beta annealed 0 -> 4)
  -> Latent vectors (16-dim)
  -> PCA -> 4 principal components
  -> Slider UI (Decay, Brightness, Subby, Click)
  -> PCA inverse -> z vector
  -> VAE decoder -> spectrogram
  -> BigVGAN vocoder -> audio
```

## Setup

### Requirements

- Python 3.10+
- Node.js 18+
- Apple Silicon (MPS), CUDA GPU, or CPU

### Install Python dependencies

```bash
pip install torch torchaudio rich matplotlib scikit-learn tensorboard flask flask-cors bigvgan
```

### Install frontend dependencies

```bash
cd web && npm install
```

### Add training data

Place `.wav` kick drum samples in `data/kicks/`.

## Usage

### 1. Train

```bash
python main.py
```

Trains for 500 epochs with beta-annealing (0 -> 4 over 100 epochs). Monitors MSE and KL divergence separately. Saves `models/best.pth` (best validation loss) and `models/checkpoint.pth` (final). Generates 20 reconstructions and 10 latent samples in `output/`.

### 2. Run the web synth

Start the backend and frontend in two terminals:

```bash
# Terminal 1 -- API backend
python app.py
```

```bash
# Terminal 2 -- Next.js frontend
cd web && npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Move the sliders to generate kick drums in real-time.

### 3. Cluster + visualize

```bash
python cluster.py
tensorboard --logdir=runs/kick_clusters
```

Clusters the latent space with a GMM (component count via BIC). TensorBoard shows:

- **Projector** tab -- latent space colored by cluster and audio descriptors
- **Audio** tab -- listen to each sample

### 4. Latent dimension experiment

```bash
python experiment_latent.py
```

Trains three models (latent_dim 8, 16, 32), runs PCA, reports explained variance. Pick the dim where 3 PCs explain 60%+ variance.

## Model

2D Convolutional VAE (~998K parameters at latent_dim=16).

| Setting | Value |
|---------|-------|
| Input | Log-mel spectrogram (1, 128, 256) |
| Sample rate | 44100 Hz |
| Audio length | ~1.49s (65536 samples) |
| N_FFT | 1024 |
| HOP_LENGTH | 256 |
| N_MELS | 128 |
| Latent dim | 16 |
| Beta | 4 (annealed from 0 over 100 epochs) |
| Vocoder | BigVGAN v2 (MIT, pretrained at 44kHz) |

- **Encoder**: 4 conv layers (1->16->32->64->128, stride 2, BatchNorm+ReLU), flatten, FC to mu/logvar
- **Decoder**: FC, reshape, 4 transposed conv layers (mirror), Sigmoid output
- **Loss**: MSE + beta * KL (sum-reduced, per sample)
- **Training**: 10% validation split, best checkpoint saved by val loss

## Project structure

```
kicks/
├── app.py                  # API backend (Flask, port 8080)
├── main.py                 # Train + generate audio samples
├── cluster.py              # GMM clustering + TensorBoard
├── experiment_latent.py    # Latent dim comparison (8, 16, 32)
├── listen.py               # Jupyter utility to preview samples
├── kicks/                  # Core Python package
│   ├── model.py            # 2D Conv VAE + audio constants
│   ├── dataset.py          # Load audio -> normalized log-mel spectrograms
│   ├── dataloader.py       # DataLoader wrapper
│   ├── train.py            # Training loop with val split + best checkpoint
│   ├── loss.py             # MSE + beta * KL loss
│   ├── vocoder.py          # BigVGAN vocoder (spec -> audio)
│   └── cluster.py          # GMM, PCA, descriptors, TensorBoard
├── web/                    # Next.js + shadcn/ui frontend
│   ├── app/page.tsx        # Main page with 4 sliders
│   └── components/ui/      # shadcn components (Slider, Card)
├── data/kicks/             # Input .wav samples (not tracked)
├── models/                 # Saved checkpoints (not tracked)
├── output/                 # Generated audio (not tracked)
└── runs/                   # TensorBoard logs (not tracked)
```

## License

MIT
