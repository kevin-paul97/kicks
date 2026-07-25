"""Automatic perceptual evaluation of generated kick drums.

Scores generated .wav files against the reference corpus and translates the
numbers into plain-English verdicts ("sounds like a real kick" / "high-end
tail is smeared"). All metrics are computed on the final waveform (i.e.
post-vocoder), because that is what the listener hears and where vocoder
artefacts (HF smear, hiss, metallic ringing) actually live.

Approach: "good" is defined statistically — a generated kick passes when each
perceptual metric falls inside the distribution of the same metric measured
over the real-kick corpus. Per-metric robust z-scores are mapped to verdicts,
a Mahalanobis distance in metric space gives an overall kick-likeness
percentile, and a Fréchet distance between the generated set and the corpus
gives a single set-level quality number (lower = closer to real kicks).

No torch/bigvgan imports — numpy/scipy only, so `kicks eval` starts fast.
"""

import glob
import hashlib
import json
import os
import random
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
import scipy.signal
import soundfile as sf

SAMPLE_RATE = 44100
AUDIO_LENGTH = 65536  # ~1.49 s, matches training pipeline

_EPS = 1e-12

# Analysis band edges (Hz)
HF_LO, HF_HI = 2000.0, 16000.0
SUB_LO, SUB_HI = 25.0, 150.0

REF_CACHE = os.path.join("output", "eval_reference.json")


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: str) -> np.ndarray | None:
    """Load a wav as mono float32 at 44.1 kHz, peak-normalized, fixed length.

    Returns None for unreadable or silent files.
    """
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return None
    x = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = np.gcd(int(sr), SAMPLE_RATE)
        x = scipy.signal.resample_poly(x, SAMPLE_RATE // g, sr // g)
    if len(x) > AUDIO_LENGTH:
        x = x[:AUDIO_LENGTH]
    elif len(x) < AUDIO_LENGTH:
        x = np.pad(x, (0, AUDIO_LENGTH - len(x)))
    peak = np.abs(x).max()
    if peak < 1e-4:
        return None
    return (x / peak).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def _rms_envelope(x: np.ndarray, win: int = 512) -> np.ndarray:
    """Sample-resolution RMS envelope via Hann-windowed moving average of x²."""
    w = scipy.signal.windows.hann(win)
    w /= w.sum()
    return np.sqrt(np.convolve(x ** 2, w, mode="same") + _EPS)

def detect_onsets(x: np.ndarray) -> np.ndarray:
    """Distinct hit times (ms) in a waveform.

    Uses a heavily smoothed (~93 ms) envelope: a 25-30 Hz sub fundamental has
    a 33-40 ms period and ripples straight through a short RMS window, making
    every sub cycle look like a separate "hit". The wide window suppresses
    that ripple, and the 150 ms minimum peak distance means anything closer
    (beater bounce, envelope wobble) counts as one hit. Real loops place hits
    a tempo apart (≥250 ms at 240 BPM), comfortably above both limits.
    """
    env = _rms_envelope(x, win=4096)
    peak = env.max()
    peaks, _ = scipy.signal.find_peaks(
        env, height=0.25 * peak, prominence=0.2 * peak,
        distance=int(0.150 * SAMPLE_RATE),
    )
    return peaks / SAMPLE_RATE * 1000.0


def _band_power_per_frame(x: np.ndarray, lo: float, hi: float,
                          n_fft: int = 1024, hop: int = 256):
    """(times_s, power) of band [lo, hi] Hz per STFT frame."""
    f, t, Z = scipy.signal.stft(
        x, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft - hop,
        boundary=None, padded=False,
    )
    P = np.abs(Z) ** 2
    band = (f >= lo) & (f < hi)
    return t, P[band].sum(axis=0), f, P


def analyze_kick(x: np.ndarray) -> dict[str, float] | None:
    """Compute perceptual metrics for a single kick waveform.

    Returns a dict of scalar metrics (times in ms, ratios in dB) or None when
    the file does not contain a usable transient.
    """
    env = _rms_envelope(x)
    peak = env.max()
    peak_i = int(env.argmax())

    # Onset: first time envelope exceeds 10% of peak
    above = np.flatnonzero(env > 0.1 * peak)
    if len(above) == 0:
        return None
    start_i = int(above[0])

    # Attack: onset -> envelope peak
    attack_ms = max(0.0, (peak_i - start_i) / SAMPLE_RATE * 1000.0)

    # Decay: peak -> last time envelope is above -30 dBFS-of-peak
    thresh = peak * 10 ** (-30 / 20)
    tail_idx = np.flatnonzero(env[peak_i:] > thresh)
    decay_ms = (tail_idx[-1] if len(tail_idx) else 0) / SAMPLE_RATE * 1000.0

    # Onset count: distinct hits => flutter / double-trigger check
    n_onsets = max(1, len(detect_onsets(x)))

    # Crest factor over the active region
    active = x[start_i: start_i + max(1, int(decay_ms / 1000 * SAMPLE_RATE) + 1)]
    rms = np.sqrt((active ** 2).mean() + _EPS)
    crest_db = 20 * np.log10(np.abs(x).max() / (rms + _EPS))

    # Tail noise floor: last 300 ms relative to peak (vocoder hiss check)
    tail = x[-int(0.3 * SAMPLE_RATE):]
    noise_floor_db = 20 * np.log10(np.sqrt((tail ** 2).mean() + _EPS) + _EPS)

    # --- High-frequency behaviour (2-16 kHz) ---------------------------------
    t, hf_pow, f, P = _band_power_per_frame(x, HF_LO, HF_HI)
    t_ms = t * 1000.0
    hf_total = hf_pow.sum() + _EPS

    # HF click presence: how much of the first 25 ms is high-frequency energy
    early = t_ms < 25.0
    frame_pow = P.sum(axis=0) + _EPS
    hf_click_db = 10 * np.log10(
        (hf_pow[early].sum() + _EPS) / (frame_pow[early].sum() + _EPS)
    )

    # HF decay: time for the HF band envelope to fall 30 dB below its own peak
    hf_peak_i = int(hf_pow.argmax())
    hf_thresh = hf_pow[hf_peak_i] * 1e-3  # -30 dB in power
    hf_above = np.flatnonzero(hf_pow[hf_peak_i:] > hf_thresh)
    hf_decay_ms = (
        t_ms[hf_peak_i + hf_above[-1]] - t_ms[hf_peak_i]
        if len(hf_above) else 0.0
    )

    # HF tail ratio: energy after 80 ms vs before (real kicks: strongly negative)
    split = t_ms >= 80.0
    hf_tail_ratio_db = 10 * np.log10(
        (hf_pow[split].sum() + _EPS) / (hf_pow[~split].sum() + _EPS)
    )

    # Tail character in the HF band (100-400 ms): spectral flatness.
    # High flatness = hiss; sustained HF with LOW flatness = metallic ringing.
    hf_bins = (f >= HF_LO) & (f < HF_HI)
    tail_frames = (t_ms >= 100.0) & (t_ms <= 400.0)
    Ptail = P[np.ix_(hf_bins, tail_frames)]
    fw = Ptail.sum(axis=0)  # weight frames by their energy
    # Only judge the tail's character when it carries audible energy: below
    # -40 dB of the total HF energy the "spectrum" is just fade/dither residue
    # and its flatness is meaningless.
    if fw.sum() > 1e-4 * hf_total:
        gmean = np.exp(np.log(Ptail + _EPS).mean(axis=0))
        amean = Ptail.mean(axis=0) + _EPS
        tail_flatness = float((gmean / amean * fw).sum() / fw.sum())
    else:
        tail_flatness = 0.5  # no tail energy -> neutral

    # --- Low end -------------------------------------------------------------
    # Sub fundamental: spectral peak 25-150 Hz over the body (20-250 ms)
    body = x[int(0.020 * SAMPLE_RATE): int(0.250 * SAMPLE_RATE)]
    w = scipy.signal.windows.hann(len(body))
    spec = np.abs(np.fft.rfft(body * w, n=1 << 16))
    freqs = np.fft.rfftfreq(1 << 16, 1 / SAMPLE_RATE)
    band = (freqs >= SUB_LO) & (freqs <= SUB_HI)
    bi = int(spec[band].argmax())
    sub_hz = float(freqs[band][bi])

    # Pitch glide: kick fundamentals sweep downward. f0 early vs late via
    # a high-resolution STFT restricted to 25-250 Hz.
    fg, tg, Zg = scipy.signal.stft(
        x, fs=SAMPLE_RATE, nperseg=4096, noverlap=4096 - 512,
        boundary=None, padded=False,
    )
    Pg = np.abs(Zg) ** 2
    lo_bins = (fg >= 25) & (fg <= 250)
    f0 = fg[lo_bins][Pg[lo_bins].argmax(axis=0)]
    tg_ms = tg * 1000.0
    e_sel = (tg_ms >= 10) & (tg_ms <= 60)
    l_sel = (tg_ms >= 100) & (tg_ms <= 250)
    if e_sel.any() and l_sel.any():
        f0_early = float(np.median(f0[e_sel]))
        f0_late = float(np.median(f0[l_sel]))
        pitch_glide = f0_early / (f0_late + _EPS)
    else:
        pitch_glide = 1.0

    # Centroid drop: spectral centroid transient vs body (clicks -> sub sweep)
    centroid = (f[:, None] * P).sum(axis=0) / (P.sum(axis=0) + _EPS)
    c_early = centroid[t_ms < 25.0]
    c_late = centroid[(t_ms >= 75.0) & (t_ms <= 200.0)]
    centroid_drop = float(
        np.log2((c_early.mean() + _EPS) / (c_late.mean() + _EPS))
    ) if len(c_early) and len(c_late) else 0.0

    return {
        "attack_ms": float(attack_ms),
        "decay_ms": float(decay_ms),
        "n_onsets": float(n_onsets),
        "crest_db": float(crest_db),
        "noise_floor_db": float(noise_floor_db),
        "hf_click_db": float(hf_click_db),
        "hf_decay_ms": float(hf_decay_ms),
        "hf_tail_ratio_db": float(hf_tail_ratio_db),
        "tail_flatness": float(tail_flatness),
        "sub_hz": sub_hz,
        "pitch_glide": float(pitch_glide),
        "centroid_drop": centroid_drop,
    }


# ---------------------------------------------------------------------------
# Metric specifications: scoring weights + verdict phrasing
# ---------------------------------------------------------------------------

@dataclass
class MetricSpec:
    key: str
    label: str
    fmt: str          # value formatting, e.g. "{:.0f} ms"
    weight: float
    log: bool         # score in log-space (for skewed ms-scale metrics)
    good: str         # phrasing when inside corpus distribution
    low: str          # phrasing when far below the corpus
    high: str         # phrasing when far above the corpus


METRIC_SPECS: list[MetricSpec] = [
    MetricSpec("attack_ms", "attack", "{:.1f} ms", 1.0, True,
               "punchy attack",
               "attack is instantaneous — may click unnaturally",
               "attack too slow — hit feels soft / swallowed"),
    MetricSpec("decay_ms", "decay", "{:.0f} ms", 1.0, True,
               "natural decay length",
               "dies off abruptly — sounds truncated",
               "rings far longer than real kicks — boomy / unresolved tail"),
    MetricSpec("crest_db", "punch (crest)", "{:.1f} dB", 0.5, False,
               "healthy transient-to-body dynamics",
               "over-compressed — transient is flattened",
               "thin body — all click, no weight"),
    MetricSpec("noise_floor_db", "tail silence", "{:.0f} dBFS", 1.5, False,
               "tail decays into silence",
               "",  # a lower noise floor than corpus is fine
               "audible noise floor in the tail — vocoder hiss"),
    MetricSpec("hf_click_db", "HF click", "{:.1f} dB", 1.0, False,
               "crisp high-frequency click in the transient",
               "no high-frequency click — sounds dull / muffled",
               "transient is all high end — sounds like a tick, not a kick"),
    MetricSpec("hf_decay_ms", "HF decay", "{:.0f} ms", 2.0, True,
               "high end decays as fast as real kicks",
               "high end vanishes instantly — click sounds detached",
               "high frequencies ring on — smeared / metallic top end"),
    MetricSpec("hf_tail_ratio_db", "HF tail energy", "{:.1f} dB", 2.0, False,
               "high-end energy correctly concentrated in the transient",
               "",  # less HF tail than corpus is not an artefact
               "too much HF energy after 80 ms — hissy / smeared tail"),
    MetricSpec("tail_flatness", "tail character", "{:.2f}", 1.0, False,
               "tail spectrum looks like a real kick",
               "tonal ringing in the tail — metallic vocoder artefact",
               "noise-like tail — hiss instead of pitch"),
    MetricSpec("sub_hz", "sub fundamental", "{:.0f} Hz", 1.0, False,
               "fundamental in the kick sweet spot",
               "fundamental below typical kicks — may just rumble",
               "fundamental too high — sounds like a tom, not a kick"),
    MetricSpec("pitch_glide", "pitch glide", "{:.2f}×", 1.0, False,
               "downward pitch sweep like a real kick",
               "pitch rises over time — unnatural for a kick",
               "extreme pitch drop — laser-like sweep"),
    MetricSpec("centroid_drop", "brightness contour", "{:.1f} oct", 1.0, False,
               "brightness falls from click to sub as expected",
               "spectrum stays static — sounds like a filtered blob",
               "unusually steep spectral collapse"),
]

_SPEC_BY_KEY = {s.key: s for s in METRIC_SPECS}


# ---------------------------------------------------------------------------
# Reference corpus statistics
# ---------------------------------------------------------------------------

@dataclass
class Reference:
    """Corpus metric distributions used to judge generated kicks."""
    values: dict[str, np.ndarray]        # sorted per-metric corpus values
    median: dict[str, float]
    sigma: dict[str, float]              # robust sigma (1.4826 * MAD)
    p5: dict[str, float]
    p95: dict[str, float]
    mean_vec: np.ndarray                 # standardized-space Gaussian fit
    cov: np.ndarray
    cov_inv: np.ndarray
    mahal_ref: np.ndarray                # corpus samples' own Mahalanobis d²
    n_files: int
    keys: list[str] = field(default_factory=list)

    def transform(self, m: dict[str, float]) -> np.ndarray:
        """Metric dict -> standardized vector (log-space where specified).

        Z-scores are clipped to ±6 so a single degenerate metric (e.g. the
        corpus tail being digital silence) cannot dominate the multivariate
        Mahalanobis / Fréchet statistics.
        """
        out = []
        for k in self.keys:
            v = _to_score_space(k, m[k])
            out.append(np.clip((v - self.median[k]) / self.sigma[k], -6.0, 6.0))
        return np.asarray(out)


def _to_score_space(key: str, v: float) -> float:
    spec = _SPEC_BY_KEY[key]
    return float(np.log1p(max(v, 0.0))) if spec.log else float(v)


def build_reference(ref_dir: str, n_samples: int = 400,
                    cache_path: str = REF_CACHE, refresh: bool = False,
                    progress=None) -> Reference:
    """Compute (or load cached) corpus metric distributions."""
    files = sorted(glob.glob(os.path.join(ref_dir, "*.wav")))
    if not files:
        raise RuntimeError(f"No .wav files in {ref_dir}")
    if n_samples and len(files) > n_samples:
        rng = random.Random(42)
        files = sorted(rng.sample(files, n_samples))

    fingerprint = hashlib.sha1(
        json.dumps([ref_dir, len(files), files[:5], files[-5:]]).encode()
    ).hexdigest()[:12]

    if not refresh and os.path.exists(cache_path):
        with open(cache_path) as fh:
            cached = json.load(fh)
        if cached.get("fingerprint") == fingerprint:
            return _reference_from_rows(
                {k: np.array(v) for k, v in cached["metrics"].items()},
                cached["n_files"],
            )

    rows: dict[str, list[float]] = {}
    n_ok = 0
    for i, path in enumerate(files):
        x = load_audio(path)
        m = analyze_kick(x) if x is not None else None
        if m is None:
            continue
        n_ok += 1
        for k, v in m.items():
            rows.setdefault(k, []).append(v)
        if progress and (i + 1) % 50 == 0:
            progress(i + 1, len(files))

    metrics = {k: np.array(v) for k, v in rows.items()}
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump(
            {"fingerprint": fingerprint, "n_files": n_ok,
             "metrics": {k: v.tolist() for k, v in metrics.items()}},
            fh,
        )
    return _reference_from_rows(metrics, n_ok)


def _reference_from_rows(metrics: dict[str, np.ndarray], n_files: int) -> Reference:
    stat_keys = [s.key for s in METRIC_SPECS if s.key != "n_onsets"]
    # noise_floor_db is excluded from the multivariate stats: the corpus tail is
    # digital silence, so any vocoder output saturates that dimension and would
    # mask all other differences. It is still reported as a standalone verdict.
    keys = [k for k in stat_keys if k != "noise_floor_db"]
    median, sigma, p5, p95, values = {}, {}, {}, {}, {}
    for k in stat_keys:
        v = metrics[k]
        s = np.array([_to_score_space(k, x) for x in v])
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med)))
        # Robust sigma with a std-based floor: when most of the corpus sits on
        # one exact value (e.g. tail = digital silence), MAD collapses to 0 and
        # would turn any deviation into an infinite z-score.
        median[k] = med
        sigma[k] = max(1.4826 * mad, 0.25 * float(s.std()), 1e-3)
        values[k] = np.sort(v)
        p5[k] = float(np.percentile(v, 5))
        p95[k] = float(np.percentile(v, 95))

    # Gaussian fit in standardized space for Mahalanobis / Fréchet
    X = np.stack([
        np.clip(
            (np.array([_to_score_space(k, x) for x in metrics[k]]) - median[k]) / sigma[k],
            -6.0, 6.0,
        )
        for k in keys
    ], axis=1)
    mean_vec = X.mean(axis=0)
    cov = np.cov(X, rowvar=False) + 1e-3 * np.eye(len(keys))
    cov_inv = np.linalg.inv(cov)
    d = X - mean_vec
    mahal_ref = np.sort(np.einsum("ij,jk,ik->i", d, cov_inv, d))

    return Reference(values=values, median=median, sigma=sigma, p5=p5, p95=p95,
                     mean_vec=mean_vec, cov=cov, cov_inv=cov_inv,
                     mahal_ref=mahal_ref, n_files=n_files, keys=keys)


# ---------------------------------------------------------------------------
# Scoring & verdict translation
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    key: str
    symbol: str      # "✓" | "⚠" | "✗"
    text: str
    value: float
    percentile: float
    z: float


@dataclass
class SampleReport:
    path: str
    score: float
    grade: str
    kick_likeness_pct: float   # % of corpus more atypical than this sample
    verdicts: list[Verdict]
    metrics: dict[str, float]


def _ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _grade(score: float) -> str:
    if score >= 85:
        return "sounds like a real kick"
    if score >= 70:
        return "kick-like, minor artefacts"
    if score >= 50:
        return "recognizably a kick, audible problems"
    return "does not pass as a kick"


def score_sample(path: str, metrics: dict[str, float], ref: Reference) -> SampleReport:
    """Score one sample against the corpus and phrase the verdicts."""
    verdicts: list[Verdict] = []
    total_w, total = 0.0, 0.0

    for spec in METRIC_SPECS:
        if spec.key == "n_onsets":
            continue
        v = metrics[spec.key]
        sv = _to_score_space(spec.key, v)
        z = (sv - ref.median[spec.key]) / ref.sigma[spec.key]
        pct = 100.0 * np.searchsorted(ref.values[spec.key], v) / len(ref.values[spec.key])

        # One-sided leniency: an empty `low`/`high` hint means that direction
        # is not an artefact (e.g. a quieter noise floor than the corpus).
        z_eff = z
        if z < 0 and not spec.low:
            z_eff = 0.0
        if z > 0 and not spec.high:
            z_eff = 0.0

        # Extreme percentiles escalate even when the corpus spread is wide:
        # sitting past the 98th percentile of real kicks is never a clean pass.
        if z_eff > 0 and pct >= 98.0 and spec.high:
            z_eff = max(z_eff, 2.5)
        if z_eff < 0 and pct <= 2.0 and spec.low:
            z_eff = min(z_eff, -2.5)

        sub = float(np.clip(1.0 - max(0.0, abs(z_eff) - 2.0) / 3.0, 0.0, 1.0))
        total += spec.weight * sub
        total_w += spec.weight

        if abs(z_eff) <= 2.0:
            sym, text = "✓", spec.good
        else:
            sym = "⚠" if abs(z_eff) <= 3.5 else "✗"
            text = spec.low if z_eff < 0 else spec.high
        detail = (
            f"{spec.label}: {spec.fmt.format(v)} "
            f"(corpus median {spec.fmt.format(ref.values[spec.key][len(ref.values[spec.key]) // 2])}, "
            f"{_ordinal(pct)} pct)"
        )
        verdicts.append(Verdict(spec.key, sym, f"{text} — {detail}", v, pct, float(z)))

    score = 100.0 * total / total_w

    # Hard gate: multiple onsets is always an artefact
    n_onsets = int(metrics["n_onsets"])
    if n_onsets > 1:
        score -= 20.0 * (n_onsets - 1)
        verdicts.append(Verdict(
            "n_onsets", "✗",
            f"{n_onsets} separate hits detected — should be a single hit "
            "(flutter / double-trigger artefact)",
            float(n_onsets), 100.0, 10.0,
        ))
    score = float(np.clip(score, 0.0, 100.0))

    # Overall kick-likeness: how typical is this sample vs the corpus itself
    vec = ref.transform(metrics)
    d = vec - ref.mean_vec
    d2 = float(d @ ref.cov_inv @ d)
    likeness = 100.0 * (1.0 - np.searchsorted(ref.mahal_ref, d2) / len(ref.mahal_ref))

    # Sort worst problems first, passes last
    order = {"✗": 0, "⚠": 1, "✓": 2}
    verdicts.sort(key=lambda v: (order[v.symbol], -abs(v.z)))

    return SampleReport(path=path, score=score, grade=_grade(score),
                        kick_likeness_pct=float(likeness),
                        verdicts=verdicts, metrics=metrics)


def frechet_distance(reports: list[SampleReport], ref: Reference) -> float:
    """Fréchet distance between generated set and corpus in metric space.

    Both sets are standardized by corpus statistics; lower is better, and
    ~0 means the generated distribution is indistinguishable from real kicks.
    """
    if len(reports) < 2:
        return float("nan")
    X = np.stack([ref.transform(r.metrics) for r in reports])
    mu_g, cov_g = X.mean(axis=0), np.cov(X, rowvar=False) + 1e-3 * np.eye(X.shape[1])
    diff = mu_g - ref.mean_vec
    covmean = scipy.linalg.sqrtm(cov_g @ ref.cov)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_g + ref.cov - 2.0 * covmean))


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def run_eval(generated: str, reference: str, pattern: str = "*.wav",
             ref_samples: int = 400, refresh_ref: bool = False,
             json_out: str | None = None, verbose: bool = True) -> dict:
    """Evaluate generated kicks against the reference corpus."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print(f"[dim]Building reference from {reference} "
                  f"(up to {ref_samples} samples)...[/dim]")
    ref = build_reference(
        reference, n_samples=ref_samples, refresh=refresh_ref,
        progress=lambda i, n: console.print(f"[dim]  {i}/{n}[/dim]"),
    )
    console.print(f"[dim]Reference: {ref.n_files} corpus kicks analyzed[/dim]\n")

    paths = sorted(glob.glob(os.path.join(generated, pattern)))
    if not paths:
        raise RuntimeError(f"No files matching {pattern} in {generated}")

    reports: list[SampleReport] = []
    for path in paths:
        x = load_audio(path)
        m = analyze_kick(x) if x is not None else None
        if m is None:
            console.print(f"[red]✗ {os.path.basename(path)}: unreadable or silent[/red]")
            continue
        reports.append(score_sample(path, m, ref))

    # Per-file output
    for r in reports:
        color = "green" if r.score >= 70 else ("yellow" if r.score >= 50 else "red")
        console.print(
            f"[bold {color}]{os.path.basename(r.path)}  "
            f"{r.score:.0f}/100 — {r.grade}[/bold {color}]  "
            f"[dim](more typical than {r.kick_likeness_pct:.0f}% of corpus)[/dim]"
        )
        if verbose:
            for v in r.verdicts:
                style = {"✓": "green", "⚠": "yellow", "✗": "red"}[v.symbol]
                console.print(f"  [{style}]{v.symbol}[/{style}] {v.text}")
        console.print()

    # Set-level summary
    fd = frechet_distance(reports, ref)
    mean_score = float(np.mean([r.score for r in reports])) if reports else 0.0
    n_pass = sum(1 for r in reports if r.score >= 70)

    table = Table(title="Set summary", show_header=False)
    table.add_row("Files evaluated", str(len(reports)))
    table.add_row("Mean score", f"{mean_score:.1f}/100")
    table.add_row("Pass rate (≥70)", f"{n_pass}/{len(reports)}")
    table.add_row("Fréchet distance to corpus", f"{fd:.2f}  (lower = more kick-like)")
    console.print(table)

    result = {
        "mean_score": mean_score,
        "pass_rate": n_pass / max(1, len(reports)),
        "frechet_distance": fd,
        "files": [
            {
                "path": r.path,
                "score": r.score,
                "grade": r.grade,
                "kick_likeness_pct": r.kick_likeness_pct,
                "metrics": r.metrics,
                "verdicts": [
                    {"metric": v.key, "symbol": v.symbol, "text": v.text,
                     "percentile": v.percentile, "z": v.z}
                    for v in r.verdicts
                ],
            }
            for r in reports
        ],
    }
    if json_out:
        with open(json_out, "w") as fh:
            json.dump(result, fh, indent=2)
        console.print(f"[dim]Wrote {json_out}[/dim]")
    return result
