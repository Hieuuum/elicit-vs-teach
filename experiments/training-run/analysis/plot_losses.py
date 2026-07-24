"""Overlay train vs. validation loss curves for a set of runs.

Plotting aid (owner 2026-07-24): compare per-step training loss
(train_log.jsonl) against the validation curve (eval_log.jsonl — stopping
evals every eval_every plus the log-spaced curve evals) for runs 7, 8,
and 10 on one log-log axes. Laptop-side, CPU-only; pull logs first
(from analysis/):

    python3 ../scripts/hf_checkpoint.py pull --run-id <rid> --no-weights
    python3 plot_losses.py [--run-id <rid> ...] [--out figures/losses.png]

Losses are plotted in nats, exactly as stored (CLAUDE.md convention:
bits conversion only at reporting boundaries — label the axis instead).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = (
    "evt-run7-armA-target-1m",
    "evt-run8-armB-target-1m",
    "evt-run10-llama1b-target",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean; output aligns to x[window-1:]."""
    if window <= 1 or len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help=f"repeatable; default: {', '.join(DEFAULT_RUNS)}",
    )
    ap.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")),
        help="artifact store root (default: $GEODE_STORE, else <repo>/geode-store)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "losses.png",
    )
    ap.add_argument(
        "--window", type=int, default=100, help="rolling-mean window for the per-step train loss"
    )
    args = ap.parse_args()
    run_ids = args.run_ids or list(DEFAULT_RUNS)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plotted = 0
    for i, rid in enumerate(run_ids):
        run_dir = args.store / "runs" / rid
        train_f, eval_f = run_dir / "train_log.jsonl", run_dir / "eval_log.jsonl"
        if not (train_f.is_file() and eval_f.is_file()):
            print(
                f"[plot] {rid}: logs missing under {run_dir} — skipped "
                f"(python3 hf_checkpoint.py pull --run-id {rid} --no-weights)"
            )
            continue
        color = colors[i % len(colors)]
        train = read_jsonl(train_f)
        t_steps = np.array([r["step"] for r in train])
        t_loss = rolling_mean(np.array([r["train_loss_nats"] for r in train]), args.window)
        ax.plot(t_steps[len(t_steps) - len(t_loss) :], t_loss, color=color, alpha=0.35, lw=1.0)
        evals = sorted(read_jsonl(eval_f), key=lambda r: r["step"])
        e_steps = np.array([r["step"] for r in evals])
        e_loss = np.array([r["val_loss_nats"] for r in evals])
        ax.plot(
            e_steps,
            e_loss,
            color=color,
            lw=1.8,
            marker=".",
            ms=3,
            label=f"{rid}  (min val {e_loss.min():.4f})",
        )
        plotted += 1
    if not plotted:
        raise SystemExit("plot_losses: no runs had logs — nothing to plot")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (nats)")
    ax.set_title(f"train (faint, rolling mean w={args.window}) vs. validation loss")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out} ({plotted} runs)")


if __name__ == "__main__":
    main()
