"""Generation with on-manifold latent sampling and eval-guided selection.

Two fixes over naive `z ~ N(0, 1)` decoding:

1. Random standard-normal latents land off the data manifold — that is where
   the blob/multi-onset generations come from. Instead, a GMM is fitted to the
   aggregate posterior (corpus latent mu vectors) and z is sampled from it.
   The fitted prior is cached in ``models/latent_prior.npz``.
2. Best-of-k: each output slot decodes several candidates, scores them with
   the perceptual metrics from ``kicks.eval`` against the corpus reference,
   and keeps the best-scoring one.
"""

import os
import random

import numpy as np
import torch

from .config import BEST_CHECKPOINT, MODEL_DIR, get_device, load_vae_from_checkpoint
from .eval import build_reference, score_sample
from .model import SAMPLE_RATE

LATENT_PRIOR_CACHE = os.path.join(MODEL_DIR, "latent_prior.npz")


def fit_latent_prior(
    model,
    device: torch.device,
    data_dir: str,
    n_fit: int = 512,
    n_components: int = 8,
    cache_path: str = LATENT_PRIOR_CACHE,
    refresh: bool = False,
):
    """Fit (or load cached) GMM over corpus latent mu vectors.

    Returns a fitted sklearn GaussianMixture whose ``sample()`` draws latents
    from the aggregate posterior instead of the standard-normal prior.
    """
    from sklearn.mixture import GaussianMixture

    latent_dim = model.latent_dim

    if not refresh and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached["means"].shape[1] == latent_dim:
            gmm = GaussianMixture(
                n_components=len(cached["weights"]), covariance_type="full"
            )
            gmm.weights_ = cached["weights"]
            gmm.means_ = cached["means"]
            gmm.covariances_ = cached["covariances"]
            gmm.precisions_cholesky_ = cached["precisions_cholesky"]
            return gmm

    from .dataset import KickDataset

    files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".wav")
    )
    if len(files) > n_fit:
        rng = random.Random(42)
        files = sorted(rng.sample(files, n_fit))

    print(f"Encoding {len(files)} corpus samples to fit the latent prior...")
    import pyloudnorm as pyln

    meter = pyln.Meter(SAMPLE_RATE)
    latents = []
    model.eval()
    with torch.no_grad():
        batch: list[torch.Tensor] = []
        for i, f in enumerate(files):
            spec = KickDataset.process_file(os.path.join(data_dir, f), meter)
            if spec is None:
                continue
            batch.append(spec)
            if len(batch) == 64 or i == len(files) - 1:
                mu, _ = model.encode(torch.stack(batch).to(device))
                latents.append(mu.cpu().numpy())
                batch = []
    X = np.concatenate(latents).astype(np.float64)

    n_components = min(n_components, max(1, len(X) // 20))
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full",
        random_state=42, n_init=3, reg_covar=1e-4,
    )
    gmm.fit(X)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(
        cache_path,
        weights=gmm.weights_, means=gmm.means_,
        covariances=gmm.covariances_,
        precisions_cholesky=gmm.precisions_cholesky_,
    )
    print(f"Latent prior: {n_components}-component GMM on {len(X)} latents "
          f"(cached to {cache_path})")
    return gmm


def generate_kicks(
    count: int = 10,
    best_of: int = 4,
    data_dir: str = "data/kicks",
    out_dir: str = "output",
    checkpoint: str = BEST_CHECKPOINT,
    seed: int | None = None,
    vocoder_type: str = "bigvgan",
    refresh_prior: bool = False,
) -> list[str]:
    """Generate kicks: GMM-prior sampling + best-of-k perceptual selection."""
    import soundfile as sf

    from .eval import analyze_kick
    from .vocoder import load_vocoder, spec_to_audio

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = get_device()
    model, _ = load_vae_from_checkpoint(checkpoint, device)
    gmm = fit_latent_prior(model, device, data_dir, refresh=refresh_prior)
    vocoder = load_vocoder(device, vocoder_type)
    ref = build_reference(data_dir)

    n_candidates = count * max(1, best_of)
    z, _ = gmm.sample(n_candidates)
    z = torch.from_numpy(np.random.default_rng(seed).permutation(z)).float()

    print(f"Decoding {n_candidates} candidates ({count} slots × best-of-{best_of})...")
    waves = []
    with torch.no_grad():
        for i in range(0, n_candidates, 8):
            spec = model.decode(z[i: i + 8].to(device))
            audio = spec_to_audio(spec, None, vocoder, device)
            waves.append(audio.cpu())
    waves = torch.cat(waves).numpy()

    scored = []
    for i in range(n_candidates):
        m = analyze_kick(waves[i].astype(np.float32))
        s = score_sample(f"candidate_{i}", m, ref).score if m is not None else 0.0
        scored.append((s, i))

    # Each output slot picks the best of its own k candidates, so the output
    # keeps the diversity of `count` independent draws.
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for slot in range(count):
        group = scored[slot * best_of: (slot + 1) * best_of]
        best_score, best_i = max(group)
        path = os.path.join(out_dir, f"gen_{slot + 1}.wav")
        sf.write(path, waves[best_i], SAMPLE_RATE)
        rest = ", ".join(f"{s:.0f}" for s, _ in sorted(group, reverse=True)[1:])
        print(f"  {path}: score {best_score:.0f} (rejected: {rest})")
        paths.append(path)
    return paths
