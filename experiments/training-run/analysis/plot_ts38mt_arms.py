"""plot_ts38mt_arms.py — one log-log figure, the ts38mt three-arm family:
base (teach reference, `evt-run1-base-v3-ext` theta0), pp (elicit theta0 =
`evt-ts38pp-parent`, 4M op full-FT pre-teach), fmt (teach theta0 =
`evt-ts38mt-fmt-parent`, 21,544-example permuted-label format pre-teach).

Reads the single committed `edl_converged_val_floor_ts38mt.csv`
(`analysis/edl_converged_val_floor.py --family ts38mt`), selecting rows by
its `condition` column (`base`/`pp`/`fmt` — NOT the noinst/inst pair the
other ts38 families use, since ts38mt is a 3-arm not 2-arm family; see
`analysis/ts38mt_mech_summary.py` and decisions.md 2026-08-21 (night) "ts38mt
pre-registration"). OCV floor only (edl_per_token_bits), matching
`plot_ts38_all_arms.py`'s convention (feedback-edl-floor-is-converged-
val-per-run.md).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ANALYSIS = Path(__file__).resolve().parent
FIGURES = ANALYSIS / "figures"
CSV = ANALYSIS / "edl_converged_val_floor_ts38mt.csv"

# Fixed categorical order (dataviz skill palette, slots 1-3).
ARMS = [
    # (condition value, color, legend label)
    ("base", "#2a78d6", "base (teach reference)"),
    ("fmt", "#eda100", "pre-teach format (teach)"),
    ("pp", "#e87ba4", "pre-teach 4M op full-FT (elicit)"),
]


def _plot_arm(ax, series, color) -> None:
    ax.plot(series["n"], series["edl_per_token_bits"], color=color, lw=2.0, alpha=0.9, zorder=2)
    ax.plot(
        series["n"],
        series["edl_per_token_bits"],
        ls="none",
        marker="o",
        ms=7,
        color=color,
        mec="white",
        mew=1.4,
        zorder=3,
    )


def main() -> None:
    if not CSV.is_file():
        raise SystemExit(f"{CSV} not found — run edl_converged_val_floor.py --family ts38mt first")
    df = pd.read_csv(CSV)

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for condition, color, label in ARMS:
        series = df[df["condition"] == condition].sort_values("n")
        if series.empty:
            print(f"[evt] condition={condition!r} not found — {label} arm skipped")
            continue
        _plot_arm(ax, series, color)
        ax.plot([], [], color=color, lw=2.0, marker="o", ms=7, mec="white", mew=1.4, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.18, lw=0.6)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xlabel("training examples $n$ (log scale)")
    ax.set_ylabel("EDL/D — bits per label token")
    ax.set_title(
        "EDL per label token vs. dataset size — OCV floor\n"
        "ts38mt: base vs. teach (format pre-teach) vs. elicit (op pre-teach)",
        fontsize=10.5,
    )
    ax.legend(fontsize=9, frameon=False, loc="upper right")

    fig.text(
        0.5,
        0.005,
        r"OCV floor: EDL$(n)$ = MDL$_{\rm epoch1}(n) - D(n)\cdot L^{\rm val}_{\rm conv}(n)$ "
        r"(each run's own $\theta_T$ val loss). No floor is shared between dataset sizes.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
        linespacing=1.5,
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.85, bottom=0.14)
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "ts38mt_arms_loglog.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[evt] wrote {out}")


if __name__ == "__main__":
    main()
