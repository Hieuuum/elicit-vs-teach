"""Figure for unlearning_wmdp.tex: the EDL sweep (M19), EDL/D against n under the OCV floor.

Data: edl_sweep_points.csv, the 105 sweep points (5 models x 3 seeds x 7 sizes) transcribed from the
stage-5 log at 365a0b3 (relearn.py --sweep). Every row satisfies MDL/D - floor = EDL/D under both
floors, and the seed means match edl_sweep.py's [edl] lines to rounding (decisions.md 2026-09-29).
Nats in the file; bits (/ ln 2) in the figure, the paper's unit.

Usage: python3 fig_edl_sweep.py   (writes fig_edl_sweep.png next to this file)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = {"orig": ("original", "black"), "rmu": ("RMU", "#1f77b4"), "elm": ("ELM", "#2ca02c"),
          "npo": ("NPO", "#d62728"), "simnpo": ("SimNPO", "#9467bd")}


def main() -> None:
    rows = list(csv.DictReader((HERE / "edl_sweep_points.csv").open()))
    ns = sorted({int(r["n"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for m, (label, color) in MODELS.items():
        seeds = sorted({r["seed"] for r in rows if r["model"] == m})
        curves = [[float(next(r for r in rows if r["model"] == m and r["seed"] == s and int(r["n"]) == n)
                          ["edl_ocv_per_token_nats"]) / math.log(2) for n in ns] for s in seeds]
        for c in curves:
            ax.plot(ns, c, color=color, lw=0.6, alpha=0.35)
        ax.plot(ns, [sum(col) / len(col) for col in zip(*curves)], color=color, marker="o", ms=3.5,
                lw=2.0 if m == "orig" else 1.5, label=label)
    ax.set_xscale("log")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yscale("symlog", linthresh=1.0, linscale=1.5)   # linear within +-1 bit: no cliffs at 0
    ax.set_ylim(-0.8, 600)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("relearned facts $n$ (of half A)")
    ax.set_ylabel("EDL / D (bits per label token)")
    ax.set_title("WMDP-bio relearning: every curve falls (elicitation)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(HERE / "fig_edl_sweep.png", dpi=200)


if __name__ == "__main__":
    main()
