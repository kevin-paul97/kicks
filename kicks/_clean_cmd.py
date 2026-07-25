"""Corpus cleaning: quarantine loops, double-hits, and perceptual outliers.

The eval scan showed ~58% of the corpus is multi-onset — loops and multi-hit
files that survived `kicks strip`, which the VAE then faithfully learns to
reproduce. This command scans every file with the perceptual analyzer from
``kicks.eval`` and moves offenders into a quarantine directory (reversible —
files are moved, never deleted).

Rules:
- ``loop``:       ≥3 detected onsets — almost certainly a loop / roll.
- ``double_hit``: exactly 2 onsets spaced >150 ms apart — a genuine second
                  hit. Two onsets within 150 ms (beater bounce / envelope
                  wobble) are kept: the detector over-counts those.
- ``outlier``:    bottom N% by Mahalanobis kick-likeness among the surviving
                  files — mislabeled percussion, FX, broken files.
"""

import json
import os
import shutil

import numpy as np

from .eval import (
    _reference_from_rows,
    analyze_kick,
    detect_onsets,
    load_audio,
)

MANIFEST = os.path.join("output", "clean_manifest.json")
DOUBLE_HIT_MS = 150.0


def run_clean(
    data: str = "data/kicks",
    quarantine_dir: str = "data/kicks_quarantine",
    outlier_pct: float = 2.0,
    apply: bool = False,
) -> dict:
    """Scan the corpus and quarantine non-kick files. Dry-run unless apply."""
    from rich.console import Console

    console = Console()
    files = sorted(f for f in os.listdir(data) if f.endswith(".wav"))
    console.print(f"Scanning {len(files)} files in {data}...")

    decisions: dict[str, str] = {}   # filename -> reason
    rows: dict[str, list[float]] = {}
    kept_metrics: list[tuple[str, dict[str, float]]] = []

    for i, f in enumerate(files):
        path = os.path.join(data, f)
        x = load_audio(path)
        if x is None:
            decisions[f] = "unreadable"
            continue
        onsets = detect_onsets(x)
        if len(onsets) >= 3:
            decisions[f] = "loop"
            continue
        if len(onsets) == 2 and (onsets[1] - onsets[0]) > DOUBLE_HIT_MS:
            decisions[f] = "double_hit"
            continue
        m = analyze_kick(x)
        if m is None:
            decisions[f] = "unreadable"
            continue
        kept_metrics.append((f, m))
        for k, v in m.items():
            rows.setdefault(k, []).append(v)
        if (i + 1) % 250 == 0:
            console.print(f"[dim]  {i + 1}/{len(files)}[/dim]")

    # Outliers: Mahalanobis distance among the surviving files
    ref = _reference_from_rows({k: np.array(v) for k, v in rows.items()},
                               len(kept_metrics))
    d2 = np.array([
        float((ref.transform(m) - ref.mean_vec) @ ref.cov_inv
              @ (ref.transform(m) - ref.mean_vec))
        for _, m in kept_metrics
    ])
    cutoff = np.percentile(d2, 100.0 - outlier_pct)
    for (f, _), d in zip(kept_metrics, d2):
        if d > cutoff:
            decisions[f] = "outlier"

    counts: dict[str, int] = {}
    for reason in decisions.values():
        counts[reason] = counts.get(reason, 0) + 1
    n_keep = len(files) - len(decisions)

    console.print()
    for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        console.print(f"  {reason:12s} {n:5d} files")
    console.print(f"  {'keep':12s} {n_keep:5d} files "
                  f"({100.0 * n_keep / len(files):.1f}% of corpus)")

    if apply:
        for f, reason in decisions.items():
            dest_dir = os.path.join(quarantine_dir, reason)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(os.path.join(data, f), os.path.join(dest_dir, f))
        os.makedirs(os.path.dirname(MANIFEST) or ".", exist_ok=True)
        with open(MANIFEST, "w") as fh:
            json.dump({"data_dir": data, "quarantine_dir": quarantine_dir,
                       "decisions": decisions}, fh, indent=2)
        console.print(f"\nMoved {len(decisions)} files to {quarantine_dir}/ "
                      f"(manifest: {MANIFEST})")
    else:
        console.print("\n[yellow]Dry run — no files moved. "
                      "Pass --apply to quarantine.[/yellow]")

    return {"counts": counts, "keep": n_keep, "decisions": decisions}
