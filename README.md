# kicks

A VAE-powered kick drum synthesizer. Train a convolutional VAE on your kick samples, then generate new kicks in real-time by moving sliders in a web UI.

## How it works

```
Kick samples (.wav)
  -> LUFS loudness normalisation (-14 LUFS)
  -> Log-mel spectrograms (128x256)
  -> Fixed dB normalisation [-80, 0] -> [0, 1]
  -> Beta-VAE training (beta=1.0, cyclical annealing)
  -> Latent vectors (32-dim)
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
pip install torch torchaudio rich matplotlib scikit-learn flask flask-cors bigvgan pyloudnorm
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

Trains for 200 epochs with cyclical beta annealing (4 cycles, beta ramping 0 -> 1.0 per cycle). Monitors reconstruction loss (spectral convergence + L1) and KL divergence separately. Saves `models/best.pth` (best validation loss) and `models/checkpoint.pth` (final). Generates 20 reconstructions and 10 latent samples in `output/`.

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

Open [http://localhost:3000](http://localhost:3000). Move the sliders to generate kick drums in real-time. Use **Randomise** to explore the latent space and **Download WAV** to save kicks.

### 3. Cluster + visualize

```bash
python cluster.py
cd web && npm run dev
```

Runs GMM clustering with BIC-selected k, PCA to 3 components, and saves analysis to `output/cluster_analysis.json`. Open [http://localhost:3000/cluster](http://localhost:3000/cluster) to view the 3D PCA visualization.

## Model

2D Convolutional VAE (~998K parameters at latent_dim=32).

| Setting | Value |
|---------|-------|
| Input | Log-mel spectrogram (1, 128, 256) |
| Sample rate | 44100 Hz |
| Audio length | ~1.49s (65536 samples) |
| N_FFT | 2048 |
| HOP_LENGTH | 256 |
| N_MELS | 128 |
| Latent dim | 32 |
| Beta | 1.0 (cyclical annealing, 4 cycles) |
| Vocoder | BigVGAN v2 (MIT, pretrained at 44kHz) |

- **Encoder**: 4 conv layers (1->32->64->128->256, stride 2, BatchNorm+ReLU), flatten, FC to mu/logvar
- **Decoder**: FC, reshape, 4 transposed conv layers (mirror), Sigmoid output
- **Loss**: Spectral convergence + L1 + beta * KL
- **Pre-processing**: LUFS loudness normalisation to -14 LUFS, fixed dB normalisation [-80, 0] -> [0, 1]
- **Training**: 10% validation split, best checkpoint saved by val loss

## Project structure

```
kicks/
├── app.py                  # API backend (Flask, port 8080)
├── main.py                 # Train + generate audio samples
├── cluster.py              # GMM clustering + TensorBoard
├── kicks/                  # Core Python package
│   ├── model.py            # 2D Conv VAE + audio constants
│   ├── dataset.py          # Load audio -> LUFS norm -> log-mel -> fixed dB norm
│   ├── dataloader.py       # DataLoader wrapper
│   ├── train.py            # Training loop with cyclical beta annealing
│   ├── loss.py             # Spectral convergence + L1 + beta * KL
│   ├── vocoder.py          # BigVGAN vocoder (spec -> audio)
│   └── cluster.py          # GMM, PCA, descriptors, TensorBoard
├── web/                    # Next.js + shadcn/ui frontend
│   ├── app/page.tsx        # Main page with sliders, randomise, download
│   └── components/ui/      # shadcn components (Slider, Card)
├── data/kicks/             # Input .wav samples (not tracked)
├── models/                 # Saved checkpoints (not tracked)
├── output/                 # Generated audio (not tracked)
└── runs/                   # TensorBoard logs (not tracked)
```

## License

MIT
