"""Kick drum synthesizer API backend."""

import io

import numpy as np
import torch
import torchaudio
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from sklearn.decomposition import PCA

from kicks import KickDataset, KickDataloader, VAE
from kicks.cluster import extract_latents
from kicks.model import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS

N_PCS = 5

# ── Startup ───────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dataset = KickDataset("data/kicks")
dataloader = KickDataloader(dataset, batch_size=32, shuffle=False)

model = VAE(latent_dim=16)
checkpoint = torch.load("models/best.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

latents, _ = extract_latents(model, dataloader, device)
pca = PCA(n_components=N_PCS)
pca.fit(latents)

pc_projected = pca.transform(latents)  # (N, 5)
PC_MINS = [float(pc_projected[:, i].min()) for i in range(N_PCS)]
PC_MAXS = [float(pc_projected[:, i].max()) for i in range(N_PCS)]

print(f"PCA variance explained: {pca.explained_variance_ratio_}")

# Griffin-Lim pipeline
_mel_fb = torchaudio.functional.melscale_fbanks(
    n_freqs=N_FFT // 2 + 1, f_min=0.0,
    f_max=SAMPLE_RATE / 2.0, n_mels=N_MELS, sample_rate=SAMPLE_RATE,
)
_mel_fb_pinv = torch.linalg.pinv(_mel_fb.T)
griffin_lim = torchaudio.transforms.GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=64)


def spec_to_audio(spec_normalized: torch.Tensor) -> torch.Tensor:
    log_mel = dataset.denormalize(spec_normalized.cpu())
    mel = torch.exp(log_mel)
    mel = mel.squeeze(1)
    linear = torch.clamp(_mel_fb_pinv @ mel, min=0.0)
    waveform = griffin_lim(linear)
    waveform = torchaudio.functional.highpass_biquad(waveform, SAMPLE_RATE, cutoff_freq=30.0)
    return waveform


# ── Flask app ─────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


@app.route("/config")
def config():
    sliders = []
    for i in range(N_PCS):
        sliders.append({
            "id": i + 1,
            "min": PC_MINS[i],
            "max": PC_MAXS[i],
            "default": (PC_MINS[i] + PC_MAXS[i]) / 2,
            "step": (PC_MAXS[i] - PC_MINS[i]) / 100,
        })
    return jsonify({"sliders": sliders})


@app.route("/generate")
def generate():
    pc_values = []
    for i in range(N_PCS):
        val = float(request.args.get(f"pc{i + 1}", 0.0))
        val = max(PC_MINS[i], min(PC_MAXS[i], val))
        pc_values.append(val)

    z_np = pca.inverse_transform([pc_values])
    z = torch.tensor(z_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        spec = model.decode(z)
        waveform = spec_to_audio(spec)

    waveform = waveform / (waveform.abs().max() + 1e-8)

    buf = io.BytesIO()
    torchaudio.save(buf, waveform, SAMPLE_RATE, format="wav")
    buf.seek(0)

    return send_file(buf, mimetype="audio/wav")


if __name__ == "__main__":
    for i in range(N_PCS):
        print(f"PC{i+1} range: [{PC_MINS[i]:.3f}, {PC_MAXS[i]:.3f}]")
    print(f"API running at http://localhost:8080")
    app.run(debug=False, port=8080)
