"""plot_probe_traj.py -- probe trajectory at ONE dataset size (default
n=46416, `scripts/run_probe_traj.sh`): carry-subset accuracy and cross-format
transfer (`probe_routing_control.py`) across snapshots of the three ts38mt
arms (base/pp/fmt) plus the k7 positive control, from theta0 to the final
snapshot. Pre-registration: decisions.md 2026-08-22 "probe trajectory";
EXPERIMENTS.md §6.24.

Reads, from `--results-dir` (default `$GEODE_STORE/results/ts38mt_probe_traj`):

- `probe_traj_{base,pp,fmt,k7}.csv` -- routing rows (one per set/layer/step)
  from a `--run-id` snapshot sweep.
- `probe_traj_{arm}_transfer.csv` -- the sweep's transfer rows (one per
  direction/layer/step), `probe_routing_control.py`'s own `--transfer-out`
  default.
- `probe_traj_theta0.csv` / `probe_traj_theta0_transfer.csv` -- the four
  parents' `checkpoint_step == -1` rows (one `--model` run per arm, all in
  one file, `model` column names the arm).

A missing file is skipped with a printed note, never an error; the script
only fails if NOTHING was found for any arm. `checkpoint_step == -1`
(theta0) is treated as step 0 for the plot's log-x axis (`step + 1`, with
-1 clipped to 0 first, so theta0 lands at x=1).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ANALYSIS = Path(__file__).resolve().parent
FIGURES = ANALYSIS / "figures"
DEFAULT_N = 46416

# (arm, color) -- dataviz skill palette, matching plot_ts38mt_arms.py's base/
# fmt/pp slots plus a fourth (k7, the ts38tr positive control).
ARMS = (
    ("base", "#2a78d6"),
    ("fmt", "#eda100"),
    ("pp", "#e87ba4"),
    ("k7", "#3a3a3a"),
)


TRAJECTORY_COLUMNS = (
    "model",
    "checkpoint_step",
    "acc_affected_best",
    "best_layer",
    "acc_affected_l8",
    "majority_affected_acc",
)
TRANSFER_COLUMNS = (
    "model",
    "checkpoint_step",
    "acc_affected_best",
    "best_layer",
    "majority_affected_acc",
)


def trajectory_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, checkpoint_step) of routing-probe rows:
    ``acc_affected_best`` (max over layers), the layer it came from
    (``best_layer``), ``acc_affected_l8`` (``acc_affected`` at the largest
    layer present in that group -- the final-layer reading), and
    ``majority_affected_acc`` (constant within a group, taken from the first
    row). Always carries ``TRAJECTORY_COLUMNS``, even on an empty/no-match
    input -- a caller (``main``'s merge, ``_plot_traj``'s ``["model"]``
    lookup) must not see a columnless frame."""
    rows = []
    for (model, step), sub in df.groupby(["model", "checkpoint_step"], sort=False):
        best = sub.loc[sub["acc_affected"].idxmax()]
        last_layer = sub.loc[sub["layer"].idxmax()]
        rows.append(
            {
                "model": model,
                "checkpoint_step": step,
                "acc_affected_best": float(best["acc_affected"]),
                "best_layer": int(best["layer"]),
                "acc_affected_l8": float(last_layer["acc_affected"]),
                "majority_affected_acc": float(sub["majority_affected_acc"].iloc[0]),
            }
        )
    return pd.DataFrame(rows, columns=list(TRAJECTORY_COLUMNS))


def transfer_table(df: pd.DataFrame, direction: str = "op_to_task") -> pd.DataFrame:
    """Like ``trajectory_table``, restricted to one ``direction`` of the
    transfer rows: one row per (model, checkpoint_step) with
    ``acc_affected_best`` (max over layers), ``best_layer``, and
    ``majority_affected_acc``. Always carries ``TRANSFER_COLUMNS`` (module
    docstring of ``trajectory_table`` explains why), including when
    ``direction`` matches nothing in ``df`` -- e.g. a killed-mid-write sweep
    file that has ``op_to_task`` rows but no ``task_to_task`` rows yet."""
    sub_dir = df[df["direction"] == direction]
    rows = []
    for (model, step), sub in sub_dir.groupby(["model", "checkpoint_step"], sort=False):
        best = sub.loc[sub["acc_affected"].idxmax()]
        rows.append(
            {
                "model": model,
                "checkpoint_step": step,
                "acc_affected_best": float(best["acc_affected"]),
                "best_layer": int(best["layer"]),
                "majority_affected_acc": float(sub["majority_affected_acc"].iloc[0]),
            }
        )
    return pd.DataFrame(rows, columns=list(TRANSFER_COLUMNS))


def _maybe_read(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        print(f"[evt] {path.name}: not found, skipped")
        return None
    return pd.read_csv(path)


def _load_family(results_dir: Path, suffix: str) -> pd.DataFrame | None:
    """Every ``probe_traj_{arm}{suffix}.csv`` (arm in ``ARMS``) plus
    ``probe_traj_theta0{suffix}.csv`` (all arms' theta0 rows, one file),
    concatenated. ``None`` if nothing was found at all."""
    parts = []
    for arm, _color in ARMS:
        d = _maybe_read(results_dir / f"probe_traj_{arm}{suffix}.csv")
        if d is not None:
            parts.append(d)
    theta0 = _maybe_read(results_dir / f"probe_traj_theta0{suffix}.csv")
    if theta0 is not None:
        parts.append(theta0)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def _plot_x(step: pd.Series) -> pd.Series:
    """theta0 (``checkpoint_step == -1``) displays as step 0; log-x plots
    ``step + 1`` so theta0 lands at x=1 rather than x=0."""
    return step.clip(lower=0) + 1


def _plot_traj(ax, table: pd.DataFrame) -> float | None:
    # majority_affected_acc is the same for every arm (one shared
    # RoutingReference/split per set) -- take it from whichever arm has data
    # FIRST in ARMS order, deterministically, rather than the last one drawn.
    chance = None
    for arm, color in ARMS:
        sub = table[table["model"] == arm].sort_values("checkpoint_step")
        if sub.empty:
            continue
        ax.plot(
            _plot_x(sub["checkpoint_step"]),
            sub["acc_affected_best"],
            color=color,
            marker="o",
            ms=5,
            lw=1.8,
            label=arm,
        )
        if chance is None:
            chance = float(sub["majority_affected_acc"].iloc[0])
    return chance


def _plot_transfer(ax, op_task: pd.DataFrame, task_task: pd.DataFrame) -> None:
    for arm, color in ARMS:
        sub = op_task[op_task["model"] == arm].sort_values("checkpoint_step")
        if not sub.empty:
            ax.plot(
                _plot_x(sub["checkpoint_step"]),
                sub["acc_affected_best"],
                color=color,
                marker="o",
                ms=5,
                lw=1.8,
                label=arm,
            )
        ceil_sub = task_task[task_task["model"] == arm].sort_values("checkpoint_step")
        if not ceil_sub.empty:
            ax.plot(
                _plot_x(ceil_sub["checkpoint_step"]),
                ceil_sub["acc_affected_best"],
                color=color,
                ls=":",
                lw=1.4,
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="probe_traj_* tables (default $GEODE_STORE/results/ts38mt_probe_traj)",
    )
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="dataset size, for the title/filename")
    ap.add_argument(
        "--out", type=Path, default=None, help="figure path (default figures/probe_traj_n<N>.png)"
    )
    args = ap.parse_args()

    results_dir = args.results_dir
    if results_dir is None:
        store = os.environ.get("GEODE_STORE")
        if store is None:
            raise SystemExit("--results-dir or $GEODE_STORE required")
        results_dir = Path(store) / "results" / "ts38mt_probe_traj"

    routing_df = _load_family(results_dir, "")
    transfer_df = _load_family(results_dir, "_transfer")
    if routing_df is None and transfer_df is None:
        raise SystemExit(f"no probe_traj_*.csv files found under {results_dir}")

    empty_routing = pd.DataFrame(
        columns=["model", "checkpoint_step", "layer", "acc_affected", "majority_affected_acc"]
    )
    empty_transfer = pd.DataFrame(
        columns=[
            "model",
            "checkpoint_step",
            "direction",
            "layer",
            "acc_affected",
            "majority_affected_acc",
        ]
    )
    traj = trajectory_table(routing_df if routing_df is not None else empty_routing)
    op_task = transfer_table(
        transfer_df if transfer_df is not None else empty_transfer, "op_to_task"
    )
    task_task = transfer_table(
        transfer_df if transfer_df is not None else empty_transfer, "task_to_task"
    )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5))
    chance = _plot_traj(ax_l, traj)
    if chance is not None:
        ax_l.axhline(chance, color="black", lw=0.9, ls="--", alpha=0.6, label="chance")
    ax_l.set_xscale("log")
    ax_l.set_xlabel("training step + 1 (theta0 = 0)")
    ax_l.set_ylabel("carry-subset accuracy (best layer)")
    ax_l.set_title("task-probe carry-subset accuracy vs. step")
    ax_l.legend(fontsize=8)
    ax_l.grid(True, alpha=0.2)

    _plot_transfer(ax_r, op_task, task_task)
    ax_r.set_xscale("log")
    ax_r.set_xlabel("training step + 1 (theta0 = 0)")
    ax_r.set_ylabel("transfer acc_affected (best layer)")
    ax_r.set_title("op_to_task transfer vs. step (dotted = task_to_task ceiling)")
    ax_r.legend(fontsize=8)
    ax_r.grid(True, alpha=0.2)

    fig.suptitle(f"probe trajectory, n={args.n}")
    fig.tight_layout()

    out = args.out or (FIGURES / f"probe_traj_n{args.n}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[evt] wrote {out}")

    summary = pd.merge(
        traj.rename(
            columns={
                "acc_affected_best": "probe_acc_affected_best",
                "best_layer": "probe_best_layer",
                "acc_affected_l8": "probe_acc_affected_l8",
                "majority_affected_acc": "probe_majority_affected_acc",
            }
        ),
        op_task.rename(
            columns={
                "acc_affected_best": "transfer_acc_affected_best",
                "best_layer": "transfer_best_layer",
                "majority_affected_acc": "transfer_majority_affected_acc",
            }
        ),
        on=["model", "checkpoint_step"],
        how="outer",
    )
    summary_path = out.with_name(f"{out.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[evt] wrote {summary_path} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
