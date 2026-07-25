"""Shared PCA analysis: fit, name PCs, compute decorrelation.

Used by both server.py (FastAPI lifespan) and tui.py (Textual on_mount)
to avoid duplicating ~80 lines of identical PCA/descriptor logic.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA

from .cluster import compute_descriptors
from .config import N_PCS

DESC_KEYS = ["sub", "punch", "click", "bright", "decay"]
DESC_NAME_MAP = {
    "sub": "Sub", "punch": "Punch", "click": "Click",
    "bright": "Bright", "decay": "Decay",
}


@dataclass
class PCAnalysis:
    pca: "PCA | DescriptorBasis"      # anything with inverse_transform()
    pc_projected: np.ndarray          # (n_samples, N_PCS)
    pc_names: list[str]               # e.g. ["Sub", "Punch", ...]
    pc_mins: list[float]              # 2nd percentile per PC
    pc_maxs: list[float]              # 98th percentile per PC
    decay_idx: int | None = None      # index of the "Decay" PC
    decay_compensation: np.ndarray | None = None  # per-PC ratios


class DescriptorBasis:
    """Slider basis with direct perceptual-descriptor control.

    Fits a linear map descriptors ≈ (z - mean_z) @ W over the corpus, then
    uses the minimum-norm pseudo-inverse directions S = W (WᵀW)⁻¹ as slider
    axes: moving slider j changes descriptor j by the requested amount while
    (to first order) leaving the other descriptors untouched — unlike raw PCA
    components, which each drag several descriptors at once.

    Duck-types PCA's `inverse_transform` so PCAnalysis consumers (server,
    TUI) work unchanged; "PC values" are simply target descriptor values.
    """

    def __init__(self, latents: np.ndarray, desc_matrix: np.ndarray):
        self.mean_z = latents.mean(axis=0)
        self.d_means = desc_matrix.mean(axis=0)
        Zc = latents - self.mean_z
        Dc = desc_matrix - self.d_means
        W, *_ = np.linalg.lstsq(Zc, Dc, rcond=None)     # (latent_dim, 5)
        self.W = W
        self.axes = W @ np.linalg.inv(W.T @ W + 1e-6 * np.eye(W.shape[1]))
        # Fit quality per descriptor (R²) — how linearly controllable each is
        pred = Zc @ W
        ss_res = ((Dc - pred) ** 2).sum(axis=0)
        ss_tot = (Dc ** 2).sum(axis=0) + 1e-12
        self.r2 = 1.0 - ss_res / ss_tot

    def inverse_transform(self, X) -> np.ndarray:
        """Descriptor targets (B, 5) -> latent vectors (B, latent_dim)."""
        X = np.asarray(X, dtype=np.float64)
        return self.mean_z + (X - self.d_means) @ self.axes.T

    def solve(self, targets, measure_fn=None, n_iter: int = 2) -> np.ndarray:
        """Latent (1, latent_dim) hitting the descriptor targets.

        With a `measure_fn` (latents (1, dim) -> measured descriptor vector of
        the *decoded* spectrogram), runs closed-loop Newton correction: decode,
        measure, step along the axes toward the residual. Measured on the
        Phase 2 model this doubles slider authority vs the open-loop map
        (~30% -> ~64% of corpus descriptor span) because the decoder's
        response is nonlinear away from the corpus mean.
        """
        t = np.asarray(targets, dtype=np.float64)
        z = self.inverse_transform(t[None])
        if measure_fn is None:
            return z
        for _ in range(n_iter):
            d = np.asarray(measure_fn(z), dtype=np.float64)
            z = z + ((t - d) @ self.axes.T)[None, :]
        return z


def analyze_latent_space(
    latents: np.ndarray,            # (n_samples, latent_dim)
    spectrograms: np.ndarray,       # (n_samples, 1, 128, 256)
    n_pcs: int = N_PCS,
    verbose: bool = True,
    basis: str = "pca",
    model=None,
) -> PCAnalysis:
    """Fit the slider basis and compute everything needed for slider mapping.

    basis="pca" (default): unsupervised PCA components, auto-named by
    descriptor correlation, with decay cross-talk compensation.
    basis="descriptor": supervised DescriptorBasis — each slider directly
    targets one perceptual descriptor (sub/punch/click/bright/decay) with
    first-order cross-talk cancellation built into the axes. When `model` is
    given, the axes are fit on descriptors of *decoded* spectrograms (the
    decoder's actual response — the correct Jacobian for closed-loop solve);
    otherwise on the corpus spectrograms.
    """
    if basis == "descriptor":
        if model is not None:
            import torch

            rng = np.random.default_rng(0)
            idx = rng.choice(len(latents), min(1500, len(latents)), replace=False)
            Z = latents[idx]
            rows = []
            with torch.no_grad():
                for i in range(0, len(Z), 64):
                    batch = torch.tensor(Z[i:i + 64], dtype=torch.float32)
                    dec = model.decode(batch.to(next(model.parameters()).device)).cpu()
                    rows.extend(
                        [compute_descriptors(dec[b])[k] for k in DESC_KEYS]
                        for b in range(len(dec))
                    )
            D = np.array(rows)
        else:
            Z = latents
            descriptors = [compute_descriptors(s) for s in spectrograms]
            D = np.array([[d[k] for k in DESC_KEYS] for d in descriptors])
        db = DescriptorBasis(Z, D)
        if verbose:
            r2s = ", ".join(f"{k}={r:.2f}" for k, r in zip(DESC_KEYS, db.r2))
            print(f"Descriptor basis fit R²: {r2s}")
        return PCAnalysis(
            pca=db,
            pc_projected=D,
            pc_names=[DESC_NAME_MAP[k] for k in DESC_KEYS],
            pc_mins=[float(np.percentile(D[:, i], 2)) for i in range(D.shape[1])],
            pc_maxs=[float(np.percentile(D[:, i], 98)) for i in range(D.shape[1])],
            decay_idx=None,
            decay_compensation=None,
        )

    descriptors = [compute_descriptors(s) for s in spectrograms]
    desc_arrays = {k: np.array([d[k] for d in descriptors]) for k in DESC_KEYS}

    # Fit PCA
    pca = PCA(n_components=n_pcs)
    pc_projected = pca.fit_transform(latents)

    # Auto-name PCs from highest descriptor correlations and flip negative axes
    pc_names: list[str] = []
    used: set[str] = set()
    for i in range(n_pcs):
        pc_vals = pc_projected[:, i]
        pc_mean, pc_std = pc_vals.mean(), pc_vals.std()
        best_desc, best_corr = None, 0.0
        for dk in DESC_KEYS:
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
            pc_names.append(DESC_NAME_MAP.get(best_desc, best_desc.capitalize()))
            if best_corr < 0:
                pca.components_[i] *= -1
                pc_projected[:, i] *= -1
                if verbose:
                    print(f"  PC{i + 1} -> {pc_names[-1]} (r={best_corr:.2f}, flipped)")
            elif verbose:
                print(f"  PC{i + 1} -> {pc_names[-1]} (r={best_corr:.2f})")
        else:
            pc_names.append(f"PC{i + 1}")
            if verbose:
                print(f"  PC{i + 1} -> PC{i + 1} (no strong correlation)")

    pc_mins = [float(np.percentile(pc_projected[:, i], 2)) for i in range(n_pcs)]
    pc_maxs = [float(np.percentile(pc_projected[:, i], 98)) for i in range(n_pcs)]

    # Decay decorrelation
    decay_idx: int | None = None
    decay_compensation: np.ndarray | None = None
    for i, name in enumerate(pc_names):
        if name == "Decay":
            decay_idx = i
            break

    if decay_idx is not None:
        decay_vals = desc_arrays["decay"]
        d_mean, d_std = decay_vals.mean(), decay_vals.std()
        betas = np.zeros(n_pcs)
        for i in range(n_pcs):
            pc_vals = pc_projected[:, i]
            pc_std = pc_vals.std()
            if pc_std > 0 and d_std > 0:
                r = float(np.corrcoef(pc_vals, decay_vals)[0, 1])
                betas[i] = r * d_std / pc_std
        decay_beta = betas[decay_idx]
        if abs(decay_beta) > 1e-8:
            ratios = betas / decay_beta
            ratios[decay_idx] = 0.0
            decay_compensation = ratios
            if verbose:
                print(f"Decay decorrelation enabled (ratios: {ratios.round(3)})")
        elif verbose:
            print("Decay decorrelation skipped (Decay PC has negligible decay correlation)")

    if verbose:
        print(f"PCA variance explained: {pca.explained_variance_ratio_}")

    return PCAnalysis(
        pca=pca,
        pc_projected=pc_projected,
        pc_names=pc_names,
        pc_mins=pc_mins,
        pc_maxs=pc_maxs,
        decay_idx=decay_idx,
        decay_compensation=decay_compensation,
    )
