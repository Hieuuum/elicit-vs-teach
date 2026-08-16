"""plot_ts38_all_arms.py — one log-log figure, all TinyStories 38.7M arms.

Overlays base (teach), pre-taught (elicit, ts38), multiwrap pre-taught
(ts38mw), pre-teach-format (ts38pf), and pre-teach 4M full-FT (ts38pp) on a
single EDL/D-vs-n plot, x-axis log-scaled. Reads the committed per-family
CSVs (edl_converged_val_floor_{ts38,ts38mw,ts38pf,ts38pp}.csv) rather than
re-collecting from run dirs — the base (noinst) row is byte-identical across
all of them (reused run, not retrained; verified 2026-08-16), so it is read
once from ts38's own CSV. ts38pp's CSV is read only if present — that family
was still building when the ts38pp arm was added here, so a missing CSV is
a graceful skip (printed note), not an error; the other arms still plot.

Broken y-axis (owner request 2026-08-16): mw's EDL/D collapses to 0.10 and
0.03 bits at n=100000/316228 (base at those sizes: 1.73/0.84) while every
other point across all four arms sits in [0.6, 5.5] bits. A single log y-axis
gives those two outliers 1.5 of the plot's ~2.3 total decades, squeezing the
cluster where the arms are actually comparable. Two stacked axes (4:1 height)
put most of the figure on the [0.55, 6.5] band and compress mw's dive into a
thin bottom strip — present, not hidden, just not competing for space.

OCV floor only (edl_per_token_bits) — the owner-default floor
(feedback-edl-floor-is-converged-val-per-run.md). Does not include ts38grid's
small-n bracket (n=128/256/512): that family was still provisioning when this
was written.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ANALYSIS = Path(__file__).resolve().parent
FIGURES = ANALYSIS / "figures"

# Fixed categorical order (dataviz skill palette, slots 1-5; validated
# colorblind-safe on the "adjacent" pairlist used for line charts).
ARMS = [
    # (csv stem, condition column value, color, legend label)
    ("edl_converged_val_floor_ts38.csv", "noinst", "#2a78d6", "base (teach)"),
    ("edl_converged_val_floor_ts38.csv", "inst", "#eb6834", "pre-taught (elicit)"),
    ("edl_converged_val_floor_ts38mw.csv", "inst", "#1baf7a", "multiwrap pre-taught (elicit)"),
    ("edl_converged_val_floor_ts38pf.csv", "inst", "#eda100", "pre-teach-format"),
    ("edl_converged_val_floor_ts38pp.csv", "inst", "#e87ba4", "pre-teach 4M full-FT (ts38pp)"),
]

# Short annotation labels for RELIEF_COLORS arms, keyed by a substring of the
# legend label (checked in order) — extend alongside ARMS/RELIEF_COLORS when
# a new low-contrast arm is added.
RELIEF_SHORT_LABELS = (("multiwrap", "mw"), ("pre-teach-format", "pf"), ("ts38pp", "pp"))

# Split point between the two panels, in bits. Every arm's value sits above
# this except mw's n=100000/316228 pair (0.102, 0.034) — see module docstring.
SPLIT = 0.55
TOP_YLIM = (SPLIT, 6.5)
BOT_YLIM = (0.025, SPLIT)
RELIEF_COLORS = (
    "#1baf7a",
    "#eda100",
    "#e87ba4",
)  # below 3:1 contrast vs. surface (dataviz validator WARN)


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
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.8, 6.4), sharex=True, height_ratios=(4, 1), gridspec_kw={"hspace": 0.06}
    )

    n_plotted = 0
    for stem, condition, color, label in ARMS:
        csv_path = ANALYSIS / stem
        if not csv_path.is_file():
            print(f"[evt] {stem} not found — {label} arm skipped (not built yet)")
            continue
        n_plotted += 1
        df = pd.read_csv(csv_path)
        series = df[df["condition"] == condition].sort_values("n")
        _plot_arm(ax_top, series, color)
        _plot_arm(ax_bot, series, color)
        # Legend proxy on whichever panel holds this arm's highest point.
        ax_top.plot([], [], color=color, lw=2.0, marker="o", ms=7, mec="white", mew=1.4, label=label)

        if color in RELIEF_COLORS:
            last = series.iloc[-1]
            target_ax = ax_top if last["edl_per_token_bits"] >= SPLIT else ax_bot
            short_label = next(s for key, s in RELIEF_SHORT_LABELS if key in label)
            target_ax.annotate(
                short_label,
                xy=(last["n"], last["edl_per_token_bits"]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                color=color,
                va="center",
                fontweight="bold",
            )

    ax_top.set_ylim(*TOP_YLIM)
    ax_bot.set_ylim(*BOT_YLIM)
    for ax in (ax_top, ax_bot):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.18, lw=0.6)
        ax.spines["right"].set_visible(False)

    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)
    ax_bot.spines["top"].set_visible(False)

    # Diagonal break marks at the panel seam.
    d = 0.5
    break_kwargs = dict(
        marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color="black", mec="black", mew=1, clip_on=False
    )
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **break_kwargs)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **break_kwargs)

    ax_bot.set_xlabel("training examples $n$ (log scale)")
    fig.text(0.02, 0.5, "EDL/D — bits per label token", va="center", rotation="vertical", fontsize=10)
    arms_tag = (
        f"all {n_plotted} arms" if n_plotted == len(ARMS) else f"{n_plotted}/{len(ARMS)} arms"
    )
    ax_top.set_title(
        "EDL per label token vs. dataset size — OCV floor\n"
        f"TinyStories 38.7M, D_algo_bare, r128 LoRA — {arms_tag}\n"
        "(broken y-axis: bottom strip is where multiwrap collapses, most other data is above the break)",
        fontsize=10.5,
    )
    ax_top.legend(fontsize=9, frameon=False, loc="upper right")

    fig.text(
        0.5,
        0.005,
        r"OCV floor: EDL$(n)$ = MDL$_{\rm epoch1}(n) - D(n)\cdot L^{\rm val}_{\rm conv}(n)$ "
        r"(each run's own $\theta_T$ val loss). No floor is shared between dataset sizes."
        "\nExcludes ts38grid's small-n bracket (n=128/256/512) — not yet converged.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
        linespacing=1.5,
    )
    fig.subplots_adjust(left=0.1, right=0.78, top=0.82, bottom=0.14, hspace=0.06)
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "ts38_all_arms_loglog.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[evt] wrote {out}")


if __name__ == "__main__":
    main()
