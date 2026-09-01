"""Two-curve figure: the same-base signature flip (owner 2026-09-01).

Left: EDL/label-token vs n — pre-elicit parent (evt-ts1b-mix-nl-*, read
from run manifests) strictly monotone vs blank TS (fig2ts noinst, read from
results/dataset_size_sweep_ts.parquet if present, else its manifests) with
the teaching hump. Right: 0-shot EM (G5) vs n. CPU-only.

Usage:
    python3 plot_mix_sweep.py [--out ../analysis/figures/fig_mix_flip.png]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
STORE = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
SIZES = [100, 316, 1000, 3162, 10000, 31623, 100000, 316228, 1000000]


def from_manifests(prefix: str) -> dict[int, tuple[float, float | None]]:
    out = {}
    for n in SIZES + [1468, 2154, 4642, 6813, 14678, 21544, 46416, 68129,
                      146780, 215443, 464159, 681292]:
        p = STORE / "runs" / f"{prefix}{n}" / "manifest.json"
        if not p.is_file():
            continue
        m = json.loads(p.read_text())
        tr = m.get("experiment", {}).get("target_result") or m.get("target_result") or {}
        gates = m.get("experiment", {}).get("gates", {})
        em = None
        g5 = gates.get("G5")
        if isinstance(g5, dict):
            em = g5.get("zero_shot_exact_match", g5.get("exact_match"))
        edl = tr.get("edl_per_label_token_nats")
        if edl is not None:
            out[n] = (float(edl), em if em is None else float(em))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis/figures/fig_mix_flip.png"))
    args = ap.parse_args()

    mix = from_manifests("evt-ts1b-mix-nl-n")
    blank = from_manifests("evt-ts1b-fig2ts-noinst-n")
    if not mix:
        raise SystemExit("[plot] no mix-nl manifests found — wrong GEODE_STORE?")
    print(f"[plot] mix rungs: {sorted(mix)}")
    print(f"[plot] blank rungs: {sorted(blank)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for data, label, color in ((blank, "blank TS (teach)", "#2c6b74"),
                               (mix, "pre-elicit parent", "#9a6316")):
        ns = sorted(data)
        ax1.plot(ns, [data[n][0] for n in ns], "o-", color=color, label=label)
        ems = [(n, data[n][1]) for n in ns if data[n][1] is not None]
        if ems:
            ax2.plot([n for n, _ in ems], [e for _, e in ems], "o-",
                     color=color, label=label)
    ax1.set_xscale("log")
    ax1.set_xlabel("n (fine-tuning examples)")
    ax1.set_ylabel("EDL / label token (nats)")
    ax1.set_title("Signature flip: hump vs monotone")
    ax2.set_xscale("log")
    ax2.set_xlabel("n")
    ax2.set_ylabel("0-shot exact match (G5)")
    ax2.set_title("Capability threshold shift")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"[plot] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
