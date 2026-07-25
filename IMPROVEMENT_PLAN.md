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

VAE retrain on the cleaned corpus is running (latent_dim 64, 200 epochs,
same CLI defaults as the current checkpoint; previous model backed up to
`models/vae_best_phase1.pth`). After training: `kicks generate
--refresh-prior`, `kicks eval --refresh-ref`, A/B against Phase 1.

## Phase 3 — HF fidelity (training changes) 🔄 CODE DONE (2026-07-25), unmeasured

1. **Transient-HF loss term** ✅ implemented (`loss.py::transient_hf_loss`,
   CLI `--hf-weight`, default 0.5). L1 on the HF click region (mel bands 50+,
   first ~35 ms) + one-sided penalty on excess HF in the tail (after ~80 ms) —
   directly optimizes the failing eval metrics (`hf_click_db`,
   `hf_tail_ratio_db`). *Not active in the current Phase 2 training run*
   (started before the change) — takes effect next run.
2. **Model-selection by eval score** ✅ implemented (`train.py`). Every 5
   epochs: sample latents from the val-posterior Gaussian, decode, score
   descriptor realism vs corpus stats (spectrogram-domain proxy, no vocoder
   needed), save best to `models/vae_best_eval.pth`. Smoke-tested end-to-end.
3. **Continue vocoder fine-tuning on the cleaned corpus** — deferred until the
   GPU is free (VAE retrain running). The current `checkpoint_100.pth` still
   hisses (mitigated by the tail gate); more epochs on clean kicks plus a
   waveform-domain silence loss on padded regions should push the noise floor
   down at the source.
4. **If smoothing persists:** lower KL pressure further on HF-heavy dims, or
   swap the decoder for a mild adversarial/perceptual refinement stage.
   Bigger effort — only after 1–3 are measured.

Next training run should use: `uv run kicks train` (HF term on by default),
then compare `vae_best.pth` vs `vae_best_eval.pth` with `kicks generate` +
`kicks eval`.

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
