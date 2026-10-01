"""plot_probe1b.py -- probe1b phase 1 figures (Llama-3.2-1B vs evt-ts1b-base
per-digit addition probes, no training). Plan `docs/plan-probe1b-digits-
phase1.md` sec8; pinned CSV schema `probe1b_contract.md`.

Reads ``probe_rows.csv`` from ``--results`` (one row per model/format/layer/
placement/head, ``head`` in {d1,d2,d3,d4,agg}; a missing file is a hard
error) and writes three PNGs to ``--out-dir``:

1. ``probe1b_grid_{model}.png`` -- one per model present in the CSV: 2x2
   grid (rows = format op/nl, cols = placement B/C) of per-layer accuracy
   curves, one line per digit head (d1-d3: ``top1_acc_affected``; d4 (no
   affected subset): ``top1_acc_all``, dotted), a dashed per-head cheat
   baseline (``cheat_acc_all``), a grey band spanning min..max over heads of
   the shuffled-label control (``shuffled_top1_affected``, falling back to
   ``shuffled_top1_all`` for d4), and the mid-layer window (hidden-state
   indices 4-13, plan sec2) lightly shaded. Placements B and C share the
   d1-d3 fits (pinned contract): the two columns differ only in the d4 line.
2. ``probe1b_overlay.png`` -- Llama vs ts1b overlay of the per-layer
   aggregate ``mean_logprob_all`` (the ``head == "agg"`` rows, nats), one
   panel per placement, line style by format.
3. ``probe1b_d4_gap.png`` -- per layer, grouped bar chart of the d4 B-vs-C
   gap ``g = acc(d4 @ C/pos2) - acc(d4 @ B/pos1)`` (``top1_acc_all``),
   grouped per (model, format) -- the R-P3 read.

``lens_rows.csv`` (logit-lens rows, also under ``--results``) is NOT read by
this script: its ``mean_logprob_nats`` is a chunk-level log-prob over the
full ~128k vocab (chance floor ~= -11.8 nats), not comparable on one axis to
``probe_rows.csv``'s per-digit 10-way ``mean_logprob_all`` (chance floor
~= -2.30 nats) used in figure 2 -- plotting both together would manufacture
a misleading comparison. None of the three figures above need it.

Degrades gracefully: a model, format, or placement combination missing from
``probe_rows.csv`` leaves the corresponding panel blank (noted on stderr)
rather than crashing; only a missing ``probe_rows.csv`` itself is a hard
error.

    python3 plot_probe1b.py [--results results/probe1b_phase1] [--out-dir analysis/figures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = Path(__file__).resolve().parent
FIGURES = ANALYSIS / "figures"

MID_LAYER_LO = 4
MID_LAYER_HI = 13

DIGIT_HEADS = ("d1", "d2", "d3", "d4")
HEAD_COLORS = {
    "d1": "#2a78d6",
    "d2": "#eda100",
    "d3": "#e87ba4",
    "d4": "#3a3a3a",
}
FORMATS = ("op", "nl")
PLACEMENTS = ("B", "C")
MODEL_COLORS = {"llama": "#2a78d6", "ts1b": "#eda100"}
FORMAT_STYLES = {"op": "-", "nl": "--"}


def plot_grid(df: pd.DataFrame, model: str, out_dir: Path) -> Path:
    """2x2 (format x placement) grid of per-layer digit-head accuracy
    curves for one model. See module docstring item 1 for the metric
    choices and the shared-fit note. Missing (format, placement) panels are
    left blank with a note on stderr and in the panel itself."""
    mdf = df[(df["model"] == model) & (df["head"].isin(DIGIT_HEADS))]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    legend_handles: list = []
    legend_labels: list[str] = []
    for r, fmt in enumerate(FORMATS):
        for c, placement in enumerate(PLACEMENTS):
            ax = axes[r][c]
            sub = mdf[(mdf["format"] == fmt) & (mdf["placement"] == placement)]
            ax.set_title(f"{fmt} / placement {placement}")
            if sub.empty:
                print(
                    f"[evt] {model}: no rows for format={fmt} placement={placement}; "
                    "panel left blank",
                    file=sys.stderr,
                )
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue
            layers = np.sort(sub["layer"].unique())
            lo, hi = float(layers.min()), float(layers.max())
            if hi >= MID_LAYER_LO and lo <= MID_LAYER_HI:
                ax.axvspan(
                    max(MID_LAYER_LO, lo), min(MID_LAYER_HI, hi), color="0.5", alpha=0.08, lw=0
                )
            shuf = sub["shuffled_top1_affected"].fillna(sub["shuffled_top1_all"])
            band = sub.assign(_shuf=shuf).groupby("layer")["_shuf"].agg(["min", "max"])
            ax.fill_between(
                band.index,
                band["min"],
                band["max"],
                color="0.7",
                alpha=0.3,
                lw=0,
                label="shuffled band",
            )
            for head in DIGIT_HEADS:
                hd = sub[sub["head"] == head].sort_values("layer")
                if hd.empty:
                    continue
                y = hd["top1_acc_all"] if head == "d4" else hd["top1_acc_affected"]
                ax.plot(
                    hd["layer"],
                    y,
                    color=HEAD_COLORS[head],
                    ls=":" if head == "d4" else "-",
                    marker="o",
                    ms=3.5,
                    lw=1.6,
                    label=head,
                )
                cheat_val = hd["cheat_acc_all"].mean()
                if pd.notna(cheat_val):
                    ax.axhline(cheat_val, color=HEAD_COLORS[head], lw=0.8, ls="--", alpha=0.6)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.2)
            if not legend_handles:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
    for ax in axes[-1]:
        ax.set_xlabel("layer (hidden-state index)")
    for ax in axes[:, 0]:
        ax.set_ylabel("top-1 acc (d1-d3: affected subset; d4: all rows)")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            fontsize=8,
            bbox_to_anchor=(0.5, 1.0),
        )
    fig.suptitle(
        f"probe1b per-digit probe accuracy -- {model}\n"
        "(placement B/C share the d1-d3 fits; only d4 differs between columns)",
        y=1.08,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    out_path = out_dir / f"probe1b_grid_{model}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_overlay(df: pd.DataFrame, out_path: Path) -> Path | None:
    """Llama vs ts1b overlay of the per-layer aggregate mean_logprob_all
    (``head == "agg"`` rows), one panel per placement present in the data,
    line style by format. Returns ``None`` (and warns on stderr) if there
    are no agg rows at all -- the figure is skipped, not an error."""
    agg = df[df["head"] == "agg"]
    if agg.empty:
        print(
            "[evt] no head=='agg' rows in probe_rows.csv; skipping probe1b_overlay.png",
            file=sys.stderr,
        )
        return None
    placements = [p for p in PLACEMENTS if p in set(agg["placement"])]
    if not placements:
        placements = sorted(agg["placement"].unique())
    models = sorted(agg["model"].unique())
    fig, axes = plt.subplots(
        1, len(placements), figsize=(6.5 * len(placements), 5.5), squeeze=False
    )
    axes = axes[0]
    for ax, placement in zip(axes, placements):
        sub_p = agg[agg["placement"] == placement]
        for model in models:
            for fmt in FORMATS:
                s = sub_p[(sub_p["model"] == model) & (sub_p["format"] == fmt)].sort_values("layer")
                if s.empty:
                    print(
                        f"[evt] overlay: no agg rows for model={model} format={fmt} "
                        f"placement={placement}; line omitted",
                        file=sys.stderr,
                    )
                    continue
                ax.plot(
                    s["layer"],
                    s["mean_logprob_all"],
                    color=MODEL_COLORS.get(model, "#888888"),
                    ls=FORMAT_STYLES.get(fmt, "-."),
                    marker="o",
                    ms=4,
                    lw=1.8,
                    label=f"{model} / {fmt}",
                )
        ax.set_xlabel("layer (hidden-state index)")
        ax.set_ylabel("mean log-prob, true digit (nats)")
        ax.set_title(f"placement {placement}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.suptitle("probe1b aggregate per-digit log-prob -- Llama vs ts1b")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def compute_d4_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, format, layer): ``gap = top1_acc_all(d4, C) -
    top1_acc_all(d4, B)`` (R-P3, plan sec2). Rows with no matching B/C pair
    at a given (model, format, layer) are dropped (inner join)."""
    d4 = df[df["head"] == "d4"]
    b = d4[d4["placement"] == "B"][["model", "format", "layer", "top1_acc_all"]].rename(
        columns={"top1_acc_all": "acc_b"}
    )
    c = d4[d4["placement"] == "C"][["model", "format", "layer", "top1_acc_all"]].rename(
        columns={"top1_acc_all": "acc_c"}
    )
    merged = pd.merge(b, c, on=["model", "format", "layer"], how="inner")
    merged["gap"] = merged["acc_c"] - merged["acc_b"]
    return merged


def plot_d4_gap(df: pd.DataFrame, out_path: Path) -> Path | None:
    """Grouped bar chart of ``compute_d4_gap`` per layer, grouped by
    (model, format) -- color by model, hatch by format. Returns ``None``
    (and warns on stderr) if there is no B/C pair for d4 anywhere."""
    gap_df = compute_d4_gap(df)
    if gap_df.empty:
        print(
            "[evt] no d4 B/C pairs in probe_rows.csv; skipping probe1b_d4_gap.png",
            file=sys.stderr,
        )
        return None
    layers = sorted(gap_df["layer"].unique())
    combos = sorted({(m, f) for m, f in zip(gap_df["model"], gap_df["format"])})
    n_combos = len(combos)
    width = 0.8 / n_combos
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * len(layers) * n_combos), 5.5))
    for i, (model, fmt) in enumerate(combos):
        sub = gap_df[(gap_df["model"] == model) & (gap_df["format"] == fmt)].set_index("layer")
        sub = sub.reindex(layers)
        offset = (i - (n_combos - 1) / 2) * width
        ax.bar(
            x + offset,
            sub["gap"].to_numpy(),
            width=width * 0.9,
            color=MODEL_COLORS.get(model, "#888888"),
            hatch="" if fmt == "op" else "//",
            edgecolor="black",
            linewidth=0.4,
            label=f"{model} / {fmt}",
        )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(layer) for layer in layers])
    ax.set_xlabel("layer (hidden-state index)")
    ax.set_ylabel("acc(d4 @ C/pos2) - acc(d4 @ B/pos1)")
    ax.set_title("d4 plan-ahead (B) vs teacher-forced (C) gap (R-P3)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    fig.suptitle("probe1b d4 B-vs-C gap")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results",
        type=Path,
        default=Path("results/probe1b_phase1"),
        help="dir with probe_rows.csv (default: results/probe1b_phase1)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=FIGURES,
        help="dir to write PNGs (default: analysis/figures next to this script)",
    )
    args = ap.parse_args()

    probe_path = args.results / "probe_rows.csv"
    if not probe_path.is_file():
        raise SystemExit(f"[evt] probe_rows.csv not found at {probe_path} (hard error)")
    df = pd.read_csv(probe_path)

    lens_path = args.results / "lens_rows.csv"
    if not lens_path.is_file():
        print(
            f"[evt] {lens_path.name}: not found under {args.results} "
            "(not consumed by these figures, see module docstring)",
            file=sys.stderr,
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for model in sorted(df["model"].unique()):
        written.append(plot_grid(df, model, out_dir))

    overlay_path = plot_overlay(df, out_dir / "probe1b_overlay.png")
    if overlay_path is not None:
        written.append(overlay_path)

    gap_path = plot_d4_gap(df, out_dir / "probe1b_d4_gap.png")
    if gap_path is not None:
        written.append(gap_path)

    for p in written:
        print(f"[evt] wrote {p}")


if __name__ == "__main__":
    main()
