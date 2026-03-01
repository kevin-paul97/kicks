"""Kick drum synthesizer API backend."""

import io

import numpy as np
import torch
import torchaudio
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from sklearn.decomposition import PCA

from kicks import KickDataset, KickDataloader, VAE
from kicks.cluster import extract_latents, compute_descriptors
from kicks.model import SAMPLE_RATE
from kicks.vocoder import load_vocoder, spec_to_audio

N_PCS = 4

# ── Startup ───────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dataset = KickDataset("data/kicks")
dataloader = KickDataloader(dataset, batch_size=32, shuffle=False)

model = VAE(latent_dim=32)
checkpoint = torch.load("models/best.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

vocoder = load_vocoder(device)

latents, spectrograms = extract_latents(model, dataloader, device)

# Compute perceptual descriptors for all samples
print("Computing descriptors for PC naming...")
descriptors = [compute_descriptors(s) for s in spectrograms]
desc_keys = ["sub", "punch", "click", "bright", "decay"]
desc_name_map = {"sub": "Sub", "punch": "Punch", "click": "Click", "bright": "Bright", "decay": "Decay"}
desc_arrays = {k: np.array([d[k] for d in descriptors]) for k in desc_keys}

# Fit PCA
pca = PCA(n_components=N_PCS)
pc_projected = pca.fit_transform(latents)

# Auto-name PCs from highest descriptor correlations and flip negative axes
PC_NAMES = []
used = set()
for i in range(N_PCS):
    pc_vals = pc_projected[:, i]
    pc_mean, pc_std = pc_vals.mean(), pc_vals.std()
    best_desc, best_corr = None, 0.0
    for dk in desc_keys:
        if dk in used:
            continue
        dv = desc_arrays[dk]
        d_mean, d_std = dv.mean(), dv.std()
        if pc_std > 0 and d_std > 0:
            corr = float(((pc_vals - pc_mean) * (dv - d_mean)).mean() / (pc_std * d_std))
        else:
            corr = 0.0
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_desc = dk
    if best_desc and abs(best_corr) >= 0.15:
        used.add(best_desc)
        PC_NAMES.append(desc_name_map.get(best_desc, best_desc.capitalize()))
        # Flip axis if negative correlation so slider right = more of that characteristic
        if best_corr < 0:
            pca.components_[i] *= -1
            pc_projected[:, i] *= -1
            print(f"  PC{i+1} -> {PC_NAMES[-1]} (r={best_corr:.2f}, flipped)")
        else:
            print(f"  PC{i+1} -> {PC_NAMES[-1]} (r={best_corr:.2f})")
    else:
        PC_NAMES.append(f"PC{i+1}")
        print(f"  PC{i+1} -> PC{i+1} (no strong correlation)")

PC_MINS = [float(pc_projected[:, i].min()) for i in range(N_PCS)]
PC_MAXS = [float(pc_projected[:, i].max()) for i in range(N_PCS)]

print(f"PCA variance explained: {pca.explained_variance_ratio_}")

# ── Flask app ─────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


@app.route("/config")
def config():
    sliders = []
    for i in range(N_PCS):
        sliders.append({
            "id": i + 1,
            "name": PC_NAMES[i],
            "min": 0,
            "max": 1,
            "default": 0.5,
            "step": 0.01,
        })
    return jsonify({"sliders": sliders})


@app.route("/generate")
def generate():
    pc_values = []
    for i in range(N_PCS):
        normalized = float(request.args.get(f"pc{i + 1}", 0.5))
        val = PC_MINS[i] + normalized * (PC_MAXS[i] - PC_MINS[i])
        val = max(PC_MINS[i], min(PC_MAXS[i], val))
        pc_values.append(val)

    z_np = pca.inverse_transform([pc_values])
    z = torch.tensor(z_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        spec = model.decode(z)
        waveform = spec_to_audio(spec, dataset, vocoder, device)

    buf = io.BytesIO()
    torchaudio.save(buf, waveform, SAMPLE_RATE, format="wav")
    buf.seek(0)

    return send_file(buf, mimetype="audio/wav")


if __name__ == "__main__":
    for i in range(N_PCS):
        print(f"{PC_NAMES[i]} range: [{PC_MINS[i]:.3f}, {PC_MAXS[i]:.3f}]")
    print(f"API running at http://localhost:8080")
    app.run(debug=False, port=8080)
