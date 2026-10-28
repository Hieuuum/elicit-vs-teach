"""Figure for the CMAP run (HANDOFF-cmap.md): recovery vs circuit size, one panel per route.

  python3 plot_cmap.py --results results_cmap --out figures/cmap_recovery.png

Lines: the child's circuit (a), the same minus edges into logits (b, the main read), and on
the elicit route the op-form parent circuit (d, nologits) and the child's nologits circuit
split into edges shared with (d) and the rest. Bands (p5-p95 of 20 draws): random sets
drawn outside (b) with its edge-type/layer mix ("typed disjoint"), and uniform random sets
minus logits edges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLOR = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
         "red": "#e34948"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "baseline": "#c3c2b7", "surface": "#fcfcfb"}
LINES = [  # (mask label, legend, color, style, width)
    ("child", "child circuit (a)", INK["muted"], ":", 1.5),
    ("child_nologits", "child circuit minus logits edges (b)", COLOR["blue"], "-", 2.5),
    ("ep_sym_nologits", "op-form parent circuit, minus logits (d)", COLOR["aqua"], "--", 1.8),
    ("child_and_ep_sym_nologits", "(b) ∩ (d)", COLOR["yellow"], "-", 1.5),
    ("child_minus_ep_sym_nologits", "(b) minus (d)", COLOR["red"], "-", 1.5),
]


def size_label(frac: float) -> str:
    return f"{frac * 100:g}%"


def plot(results: Path, out: Path) -> None:
    routes = [r for r in ("elicit", "teach") if (results / f"{r}.json").is_file()]
    fig, axes = plt.subplots(1, len(routes), figsize=(6.2 * len(routes), 4.6), squeeze=False)
    for ax, route in zip(axes[0], routes):
        r = json.loads((results / f"{route}.json").read_text())
        sizes = sorted(r["sizes"], key=float)  # every size present, ascending
        x = np.array([float(s) for s in sizes])
        for key, lab, color in (("random_typed_disjoint", "type/layer-matched random, outside (b)",
                                 COLOR["blue"]),
                                ("random_nologits", "uniform random minus logits", INK["muted"])):
            if key not in r["sizes"][sizes[0]]:
                continue
            band = np.array([r["sizes"][s][key]["recovery_band"] for s in sizes])
            ax.fill_between(x, band[:, 0], band[:, 2], color=color, alpha=0.13, linewidth=0,
                            label=f"{lab}, p5–p95")
        for mask, lab, color, ls, lw in LINES:
            if mask not in r["sizes"][sizes[0]]["masks"]:
                continue
            y = [r["sizes"][s]["masks"][mask]["recovery"] for s in sizes]
            ax.plot(x, y, color=color, linestyle=ls, linewidth=lw, marker="o", markersize=3.5,
                    label=lab)
        ax.axhline(0, color=INK["baseline"], linewidth=1)
        ax.axhline(r["all"]["recovery"], color=INK["baseline"], linewidth=1, linestyle="--")
        ax.text(x[0], r["all"]["recovery"], " all edges patched", fontsize=8,
                color=INK["secondary"], va="bottom")
        ax.set_xscale("log")
        ax.set_xticks(x, [size_label(v) for v in x], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("circuit size (fraction of 195,865 edges)")
        ax.set_ylabel("recovery = 1 − KL(child‖patched) / KL(child‖parent)")
        b = r["baseline"]
        ax.set_title(f"{route}: child → parent\nKL(child‖parent) {b['kl_child_parent_mean']:.1f} "
                     f"nats; 2-token acc parent {b['acc2_parent']:.2f}, child "
                     f"{b['acc2_child']:.2f}", fontsize=10)
        ax.grid(True, color=INK["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5, loc="center right")
    fig.suptitle("Edge-level CMAP: the parent runs with the child's outputs on a set of edges "
                 "(256 NL prompts, single seed)", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=INK["surface"])
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    p.add_argument("--results", default=str(here / "results_cmap"))
    p.add_argument("--out", default=str(here / "figures" / "cmap_recovery.png"))
    a = p.parse_args(argv)
    plot(Path(a.results), Path(a.out))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
