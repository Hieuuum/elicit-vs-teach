"""plot_ts38mt_train_val.py -- train loss (last ~5% of steps, averaged) vs.
converged val loss, per arm, across the ts38mt grid: base / pp / fmt at
each of the ten dataset sizes (1,000-316,228).

Val loss: `l_val_converged_nats` from the committed
`edl_converged_val_floor_ts38mt.csv` (the OCV floor -- decisions.md
feedback-edl-floor-is-converged-val-per-run.md -- same number used by
`plot_ts38mt_arms.py`).

Train loss: not a pre-aggregated field anywhere in this repo. Derived here
from each run's raw `train_log.jsonl` (`geode-store/runs/<run_id>/`, one
JSON object per optimizer step with `train_loss_nats`, single-batch and
noisy) as the mean over the last 5% of logged steps (min 5 steps) -- a
smoothed "converged" train loss, parallel to how the val number is already
a floor/average rather than one raw reading.

Linear (not log) y-axis, auto-ranged to the data: across the grid, loss
spans roughly 0.02 nats (n=316,228) to 4+ nats (n=1,000), so a fixed 0-1
range would clip most of the small-n side, where the arms differ most.
x-axis (dataset size) is log-spaced.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ANALYSIS = Path(__file__).resolve().parent
FIGURES = ANALYSIS / "figures"
VAL_CSV = ANALYSIS / "edl_converged_val_floor_ts38mt.csv"
GEODE_STORE = ANALYSIS.parents[2] / "geode-store"
RUN_PREFIX = "evt-ts38mt"

# Fixed categorical order + colors (dataviz skill palette, slots 1-3),
# matching plot_ts38mt_arms.py.
ARMS = (
    ("base", "#2a78d6", "base"),
    ("fmt", "#eda100", "pre-teach format"),
    ("pp", "#e87ba4", "pre-teach 4M op full-FT"),
)

TRAIN_TAIL_FRAC = 0.05
TRAIN_TAIL_MIN_STEPS = 5


def run_id(arm: str, n: int) -> str:
    return f"{RUN_PREFIX}-{arm}-n{n}"


def final_train_loss_nats(run_id_: str) -> float:
    """Mean `train_loss_nats` over the last `TRAIN_TAIL_FRAC` of logged
    steps (at least `TRAIN_TAIL_MIN_STEPS`) -- smooths the single-batch
    noise in the raw per-step log into one representative "converged"
    train loss per run."""
    path = GEODE_STORE / "runs" / run_id_ / "train_log.jsonl"
    losses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            losses.append(json.loads(line)["train_loss_nats"])
    if not losses:
        raise ValueError(f"{path}: no train_log rows")
    tail_n = max(TRAIN_TAIL_MIN_STEPS, round(len(losses) * TRAIN_TAIL_FRAC))
    tail_n = min(tail_n, len(losses))
    return sum(losses[-tail_n:]) / tail_n


def main() -> None:
    if not VAL_CSV.is_file():
        raise SystemExit(f"{VAL_CSV} not found -- run edl_converged_val_floor.py --family ts38mt first")
    val_df = pd.read_csv(VAL_CSV)

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    for arm, color, label in ARMS:
        sub = val_df[val_df["condition"] == arm].sort_values("n")
        if sub.empty:
            print(f"[evt] {arm}: no rows in {VAL_CSV.name}, skipped")
            continue
        ns = sub["n"].tolist()
        val_losses = sub["l_val_converged_nats"].tolist()
        train_losses = [final_train_loss_nats(run_id(arm, n)) for n in ns]

        ax.plot(
            ns, train_losses, color=color, lw=2.0, ls="-", marker="o", ms=6,
            mec="white", mew=1.2, label=f"{label} -- train", zorder=3,
        )
        ax.plot(
            ns, val_losses, color=color, lw=2.0, ls="--", marker="s", ms=6,
            mec="white", mew=1.2, label=f"{label} -- val (converged)", zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel("dataset size n (log scale)")
    ax.set_ylabel("loss (nats/token, linear scale)")
    ax.set_title("ts38mt: train loss (last 5% of steps) vs. converged val loss")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "ts38mt_train_val.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[evt] wrote {out}")


if __name__ == "__main__":
    main()
