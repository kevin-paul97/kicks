# Kick Quality Improvement Plan

Grounded in measurements from `kicks eval` (2026-07-25). The eval scores every
kick against the real-kick corpus and translates metrics into perceptual
verdicts — see "Measurement" below.

## Baseline (measured)

| Set | Mean score | Pass rate (≥70) | Fréchet distance to corpus |
|---|---|---|---|
| Real kicks (control, 66 files) | 81.8 | 80% | 2.7 |
| VAE reconstructions (`recon_*.wav`) | 69.3 | 30% | 14.0 |
| Random generations (`gen_*.wav`) | 60.1 | 30% | 14.3 |

Failure modes across the 10 generated kicks, most frequent first:

1. **Vocoder hiss — 10/10 files.** Tail noise floor at −93 dBFS; real kicks
   decay into digital silence (−120 dBFS). Present despite the fine-tuned
   BigVGAN checkpoint.
2. **Multiple onsets — 7/10 files** (2–5 hits per file). Root cause is the
   training data, not the model: **57.8% of the corpus is multi-onset**
   (29% have ≥3 hits) — loops and multi-hits that survived `kicks strip`.
   The VAE reproduces what it was trained on.
3. **HF smear — ~4/10 files.** High frequencies ring for 150–226 ms vs the
   corpus median of 67 ms; one file has *more* HF energy after 80 ms than
   before (hf_tail_ratio +2.7 dB, 98th percentile). This is the
   VAE-over-smoothing / vocoder problem the recent loss work targets.

## Phase 1 — Quick wins, no retraining ✅ DONE (2026-07-25)

Implemented and measured (same eval reference for both rows):

| Set | Mean score | Pass rate (≥70) | Fréchet distance |
|---|---|---|---|
| Baseline `N(0,1)` sampling (`output/gen_baseline/`) | 57.9 | 3/10 | 14.8 |
| **After Phase 1** (`kicks generate`) | **98.5** | **10/10** | 19.7 |

All three baseline failure modes fixed: tail noise −93 → −120 dBFS (true
silence), onsets 2.4 → 1.0 per file, HF decay 114 → ~55 ms (corpus median 67).

1. **Tail gate in vocoder post-processing** (`vocoder.py::_gate_tail`).
   Envelope-following gate: below −70 dBFS the tail fades to digital silence
   over 80 ms. Applied in both BigVGAN and Griffin-LIM paths.
2. **Aggregate-posterior sampling** (`generate.py::fit_latent_prior`).
   8-component GMM fitted on 512 corpus latent µ vectors (cached in
   `models/latent_prior.npz`), sampled instead of `N(0,1)`.
3. **Best-of-k generation** (`kicks generate --best-of 4`). Each output slot
   decodes k candidates and keeps the best `kicks eval` score. Rejected
   candidates still mostly score 80–100, so the GMM prior does most of the
   work — selection is a safety floor, not a crutch.

Caveats discovered:

- **Fréchet distance rose** (14.8 → 19.7) while per-sample quality soared:
  best-of-k narrows diversity (short punchy kicks win), and the gate pushes
  HF-tail energy below the corpus range. If set-diversity matters, use
  `--best-of 1`; per-sample quality stays high (GMM prior alone).
- **Remaining artefact — 9/10 files:** negative brightness contour (spectrum
  starts dark, gets brighter). The VAE still doesn't decode a proper HF click
  transient — this is the over-smoothing problem, fixable only in Phase 3.

## Phase 2 — Data cleaning 🔄 IN PROGRESS (2026-07-25)

**Correction to the baseline analysis:** the "57.8% multi-onset corpus"
finding was mostly a measurement artifact. A 25–30 Hz sub fundamental has a
33–40 ms period, and the original 11 ms RMS envelope window tracked individual
sub cycles as separate "hits". `detect_onsets` now uses a ~93 ms smoothed
envelope with a 150 ms minimum peak distance (validated: deep-sub kicks → 1
onset; real loops from `data/kicks_loops/` → hits at tempo spacing). The
real contamination rate is **3.2%**.

Done via the new `kicks clean` command (dry-run by default, `--apply` moves
files to `data/kicks_quarantine/<reason>/`, manifest in
`output/clean_manifest.json`, fully reversible):

- 18 loops (verified: onset spacing matches BPM filename tags)
- 27 double-hits (second onset >150 ms after the first)
- 80 perceptual outliers (bottom 2% Mahalanobis kick-likeness)
- 2 unreadable files
- **kept 3,892 of 4,019 files (96.8%)**

**Measured results** (all sets scored against the same cleaned-corpus
reference and the fixed onset detector; Phase 1/2 rows use the full
GMM-prior + gate + best-of-4 pipeline, models differ only in training data):

| Set | Mean score | ⚠/✗ flags (10 files) | Fréchet distance |
|---|---|---|---|
| Old `N(0,1)` + no gate (baseline wavs) | 87.1 | 17 | 13.2 |
| Phase 1 model (dirty corpus) | 99.1 | 7 | 16.3 |
| **Phase 2 model (cleaned corpus)** | **99.2** | **5** | **14.1** |

Note: the baseline's previously reported 57.9 was measured with the buggy
onset detector; 87.1 is the honest number (its real failures — hiss in
10/10, HF smear — remain). Cleaning's gains are where predicted: brightness
contour −2.2 → −0.6 oct (corpus median +2.1), `centroid_drop` flags 7 → 4,
HF click −35.8 → −33.8 dB (corpus median −33.6), FD 16.3 → 14.1. Phase 2
model best: epoch 151, val_loss 0.3367. Backed up as
`models/vae_best_phase2.pth`.

## Phase 3 — HF fidelity (training changes) ✅ MEASURED (2026-07-25)

**Outcome: the transient-HF loss (weight 0.5) did not beat Phase 2** — the
Phase 2 model remains production (`models/vae_best.pth`):

| Model | Mean score | Flags | Fréchet |
|---|---|---|---|
| **Phase 2 (production)** | **99.2** | **5** | **14.06** |
| Phase 3, val-loss checkpoint | 98.6 | 9 | 16.08 |
| Phase 3, eval-proxy checkpoint | 99.0 | 6 | 15.27 |

The HF term bought a slightly stronger click on the val-loss checkpoint
(−32.5 vs −33.8 dB) but worsened the brightness contour (−2.0 vs −0.6 oct)
and the set-level distribution. **The eval-proxy checkpoint selection did
validate itself** — it beat val-loss selection within the same run on every
aggregate (15.27 vs 16.08 FD, 6 vs 9 flags). Next attempts if HF fidelity is
revisited: lower `--hf-weight` (0.1–0.25), or click-region-only term without
the tail penalty (the tail is already handled by the vocoder gate).
Checkpoints kept: `vae_best_phase3.pth`, `vae_best_eval.pth`.

1. **Transient-HF loss term** — implemented (`loss.py::transient_hf_loss`,
   CLI `--hf-weight`, default 0.5) and measured: no net win at 0.5 (see
   table above).
2. **Model-selection by eval score** — implemented (`train.py`) and
   validated: the proxy-selected checkpoint beat the val-loss checkpoint on
   every set-level metric within the Phase 3 run.
3. **Continue vocoder fine-tuning on the cleaned corpus** — still open. The
   current `checkpoint_100.pth` hisses (mitigated by the tail gate); more
   epochs on clean kicks plus a waveform-domain silence loss on padded
   regions should push the noise floor down at the source.
4. **If smoothing persists:** lower `--hf-weight` (0.1–0.25) or click-only
   loss; lower KL pressure on HF-heavy dims; or a mild adversarial/
   perceptual refinement stage on the decoder.

## REST API generation + evaluation ✅ (2026-07-25)

- `GET /evaluate` (server): same params as `/generate` (sliders + optional
  attack/decay/drive/filter), returns eval score/grade/verdicts, waveform
  metrics, and the 5 spectrogram descriptors. Scores exactly the audio
  `/generate` returns (shared `_synthesize`).
- `kicks sweep`: Latin-hypercube sample of slider space against a running
  server; saves WAVs + `output/sweep_report.json`, prints per-kick verdicts
  and slider↔score correlations.
- Measured (Phase 2 model, `--control descriptor`, 20 points): **mean 98.1,
  100% pass, worst 79.4** (extreme corner: Decay≈0 + high Sub/Punch →
  double-hit flag). Only Decay correlates with score (r = +0.43, short-decay
  corner is the risky region); the rest of the slider cube is uniformly
  high-quality.

## PCA-controlled generation ✅ MEASURED + UPGRADED (2026-07-25)

Sweep harness (Phase 2 model, spectrogram-domain): each slider swept 0→1
with others at 0.5, correlating slider position against all five descriptors
and measuring effect size as % of the corpus descriptor span (2nd–98th pct).

**Raw PCA sliders (shipped behavior):** direction works — every named slider
moves its descriptor monotonically (r ≈ +1.0) — but authority is weak
(~34% of corpus span) and cross-talk pervasive (~23%: each slider drags most
other descriptors nearly as hard). Root causes: 5 PCs explain only ~40% of
latent variance, and PC↔descriptor naming correlations are weak (0.15–0.34).

**Fix: supervised descriptor axes** (`pca_analysis.py::DescriptorBasis`,
`kicks serve --control descriptor`). Linear probe z→descriptors has R² 0.97–
0.99, so descriptors are almost perfectly linearly *readable* — the axes are
the probe's pseudo-inverse (first-order zero cross-talk). Open-loop that
still underdelivers (~30% authority) because the *decoder's* response is
nonlinear away from the mean; adding closed-loop Newton correction (decode →
measure descriptors → correct, 2 iterations ≈ 2 extra VAE decodes per
generation) gives:

| Basis | Authority (named slider) | Cross-talk | Monotonicity |
|---|---|---|---|
| PCA (current default) | 34% | 23% | ~+1.0 |
| Descriptor axes, open-loop | 29% | 18% | +0.86 |
| **Descriptor axes, closed-loop** | **64%** | 23% | **+0.98** |

Decay reaches 95% span, bright 63%, click 56%. Remaining gap: punch is
promiscuous (co-moves with sub/bright — physically entangled in kicks), and
authority is capped by what the decoder can express — expect the Phase 3
HF-loss model to widen the click/bright axes. Re-measure after Phase 3.

## Phase 4 — Continuous verification

- Run `kicks eval --json runs/eval_<date>.json` automatically after every
  `kicks train` / `kicks fine-tune`; append the set summary to a history file.
- Regression gate: fail the run if mean score or Fréchet distance regresses
  vs the previous run.
- Optional upgrade: CLAP-embedding FAD as a learned second opinion (heavier
  dependency; only if the DSP metrics stop discriminating).

## Measurement

`kicks eval` (in `kicks/eval.py`) — waveform-domain metrics, corpus-referenced:

- Per-metric verdicts: attack/decay envelope, crest, onset count, sub
  fundamental, downward pitch glide, spectral-centroid drop, **HF click
  presence, HF decay time, HF tail-energy ratio, tail spectral flatness**
  (hiss vs metallic ringing), tail noise floor.
- Numbers → words: each metric's robust z-score / percentile vs the real
  corpus maps to ✓/⚠/✗ plus a plain-English phrase; overall 0–100 score with
  grades ("sounds like a real kick" ≥ 85 … "does not pass as a kick" < 50).
- Set-level: Fréchet distance in standardized metric space (lightweight FAD).
- Usage: `uv run kicks eval` (generated), `--pattern "recon_*.wav"`,
  `-g data/kicks --pattern "..."` for controls, `--refresh-ref` after any
  corpus change, `--json` for machine-readable output.
