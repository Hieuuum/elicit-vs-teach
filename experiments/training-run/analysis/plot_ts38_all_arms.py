"""plot_ts38_all_arms.py — one log-log figure, the primary TinyStories 38.7M
arms: base (teach), pre-teach-format (ts38pf), and pre-teach 4M full-FT
(ts38pp).

Narrowed 2026-08-21 evening (owner request) from a 5-arm chart that also
carried multiwrap pre-taught (ts38mw) and format-install dose i=1000
(ts38fs) — dropped the same way ts38's own "pre-taught (elicit)" arm was
dropped 2026-08-17: an explicit ARMS trim, not a parameterized option.
mw/fs still have their own committed CSVs and figures
(edl_converged_val_floor_ts38mw.csv, edl_converged_val_floor_ts38fs.csv +
ts38fs_dose_curve.py) if a future chart wants them back.

Reads the committed per-family CSVs
(edl_converged_val_floor_{ts38,ts38pf,ts38pp}.csv) rather than
re-collecting from run dirs — the base (noinst) row is byte-identical
across all three (reused run, not retrained; verified 2026-08-16), so it
is read once from ts38's own CSV, which ts38dense (2026-08-21) grew to the
full 10-point grid via `edl_converged_val_floor.py --family ts38` (base's
evt-ts38-base-n<N> ids match that family's regex at every size, new or
shipped). ts38pp is read at 10 points (ts38dense densified it); pf stays
at its original 5. Any missing CSV is a graceful skip (printed note), not
an error; the other arms still plot.

Single log-log axis: dropping mw removed the only arm whose EDL/D
collapsed below 0.1 bits (the reason an earlier version of this figure
used a broken two-panel y-axis) — the three arms here span 0.31-4.48
bits, well inside one log decade band, so a plain axis reads cleanly.

OCV floor only (edl_per_token_bits) — the owner-default floor
(feedback-edl-floor-is-converged-val-per-run.md). Does not include ts38grid's
small-n bracket (n=128/256/512): that family was never launched (owner:
"confirm results with me first", 2026-08-16).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ANALYSIS = Path(__file__).resolve().parent
FIGURES = ANALYSIS / "figures"


def _condition(value):
    """Select rows by the standard noinst/inst "condition" column."""
    return lambda df: df[df["condition"] == value]


# Fixed categorical order (dataviz skill palette, slots 1-3; validated
# colorblind-safe on the "adjacent" pairlist used for line charts).
ARMS = [
    # (csv stem, row-selector(df) -> filtered df, color, legend label)
    ("edl_converged_val_floor_ts38.csv", _condition("noinst"), "#2a78d6", "base (teach)"),
    ("edl_converged_val_floor_ts38pf.csv", _condition("inst"), "#eda100", "pre-teach-format"),
    (
        "edl_converged_val_floor_ts38pp.csv",
        _condition("inst"),
        "#e87ba4",
        "pre-teach 4M full-FT (ts38pp)",
    ),
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
    fig, ax = plt.subplots(figsize=(8.4, 5.8))

    n_plotted = 0
    for stem, select, color, label in ARMS:
        csv_path = ANALYSIS / stem
        if not csv_path.is_file():
            print(f"[evt] {stem} not found — {label} arm skipped (not built yet)")
            continue
        n_plotted += 1
        df = pd.read_csv(csv_path)
        series = select(df).sort_values("n")
        _plot_arm(ax, series, color)
        ax.plot([], [], color=color, lw=2.0, marker="o", ms=7, mec="white", mew=1.4, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.18, lw=0.6)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xlabel("training examples $n$ (log scale)")
    ax.set_ylabel("EDL/D — bits per label token")
    arms_tag = (
        f"all {n_plotted} arms" if n_plotted == len(ARMS) else f"{n_plotted}/{len(ARMS)} arms"
    )
    ax.set_title(
        "EDL per label token vs. dataset size — OCV floor\n"
        f"TinyStories 38.7M, D_algo_bare, r128 LoRA — {arms_tag}",
        fontsize=10.5,
    )
    ax.legend(fontsize=9, frameon=False, loc="upper right")

    fig.text(
        0.5,
        0.005,
        r"OCV floor: EDL$(n)$ = MDL$_{\rm epoch1}(n) - D(n)\cdot L^{\rm val}_{\rm conv}(n)$ "
        r"(each run's own $\theta_T$ val loss). No floor is shared between dataset sizes."
        "\nExcludes ts38grid's small-n bracket (n=128/256/512) — never launched.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
        linespacing=1.5,
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.85, bottom=0.16)
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "ts38_all_arms_loglog.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[evt] wrote {out}")


if __name__ == "__main__":
    main()
