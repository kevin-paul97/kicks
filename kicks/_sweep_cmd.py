"""Sweep the REST API's slider space and evaluate every generated kick.

Samples diverse slider combinations (Latin hypercube over the PC sliders, plus
the center point), calls the running server's /generate and /evaluate for
each, saves the WAVs, and reports per-kick verdicts plus which sliders
correlate with eval quality. Requires `kicks serve` to be running.
"""

import json
import os
import urllib.parse
import urllib.request

import numpy as np


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def _get_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _latin_hypercube(n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    """n stratified samples in [0,1]^dims — spreads kicks across slider space."""
    u = (rng.random((n, dims)) + np.arange(n)[:, None]) / n
    for d in range(dims):
        u[:, d] = u[rng.permutation(n), d]
    return u


def run_sweep(
    server: str = "http://localhost:8080",
    count: int = 20,
    out_dir: str = "output/sweep",
    seed: int = 42,
    json_out: str = "output/sweep_report.json",
) -> dict:
    """Generate `count` diverse kicks via the API and evaluate each."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    server = server.rstrip("/")

    cfg = _get_json(f"{server}/config")
    sliders = cfg["sliders"]
    names = [s["name"] for s in sliders]
    console.print(f"Server ready — sliders: {', '.join(names)} "
                  f"(vocoder: {cfg.get('vocoder', '?')})")

    rng = np.random.default_rng(seed)
    grid = _latin_hypercube(count - 1, len(sliders), rng) if count > 1 else np.empty((0, len(sliders)))
    points = np.vstack([np.full((1, len(sliders)), 0.5), grid])  # center first

    os.makedirs(out_dir, exist_ok=True)
    results = []
    for i, point in enumerate(points):
        params = {f"pc{j + 1}": f"{v:.3f}" for j, v in enumerate(point)}
        query = urllib.parse.urlencode(params)
        report = _get_json(f"{server}/evaluate?{query}")
        wav_path = os.path.join(out_dir, f"sweep_{i + 1:02d}.wav")
        with open(wav_path, "wb") as fh:
            fh.write(_get_bytes(f"{server}/generate?{query}"))
        report["sliders"] = {n: float(v) for n, v in zip(names, point)}
        report["wav"] = wav_path
        results.append(report)

        score = report.get("score", 0.0)
        color = "green" if score >= 70 else ("yellow" if score >= 50 else "red")
        worst = next((v["text"].split(" — ")[0] for v in report.get("verdicts", [])
                      if v["symbol"] == "✗"), "")
        slider_str = " ".join(f"{n}={v:.2f}" for n, v in report["sliders"].items())
        console.print(f"[{color}]{i + 1:2d}. {score:5.1f}  {slider_str}"
                      f"{'  ✗ ' + worst if worst else ''}[/{color}]")

    scored = [r for r in results if "score" in r]
    scores = np.array([r["score"] for r in scored])

    # Which sliders drive quality? Pearson r of slider position vs score.
    table = Table(title="Slider ↔ eval-score correlation", show_header=True)
    table.add_column("Slider")
    table.add_column("r", justify="right")
    for j, n in enumerate(names):
        vals = np.array([list(r["sliders"].values())[j] for r in scored])
        r_val = float(np.corrcoef(vals, scores)[0, 1]) if len(scored) > 2 and vals.std() > 0 else float("nan")
        table.add_row(n, f"{r_val:+.2f}")
    console.print(table)

    summary = {
        "mean_score": float(scores.mean()) if len(scores) else 0.0,
        "pass_rate": float((scores >= 70).mean()) if len(scores) else 0.0,
        "min": float(scores.min()) if len(scores) else 0.0,
        "max": float(scores.max()) if len(scores) else 0.0,
    }
    console.print(
        f"\nSweep: mean {summary['mean_score']:.1f}/100, "
        f"pass rate {100 * summary['pass_rate']:.0f}%, "
        f"range [{summary['min']:.0f}, {summary['max']:.0f}] "
        f"({len(scored)}/{len(results)} evaluated)"
    )

    payload = {"summary": summary, "results": results}
    if json_out:
        with open(json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        console.print(f"[dim]Wrote {json_out}[/dim]")
    return payload
