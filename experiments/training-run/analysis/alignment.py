"""Gradient-alignment analysis driver (spec 02 §7 metric 1, §9 deliverable).

For each run's probe dumps (written by ``scripts/extract.py``), computes the
cross-example activation-gradient alignment per (snapshot, layer) via
``geode.probe.gradient_alignment`` — pairwise-cosine mean + top-PC explained
variance — conditioning on per-example probe loss > 0 (spec §7: late
gradients are numerically degenerate). Results land as one long-format
parquet through the ZOO-4 writer (``results/gradient_alignment.parquet``,
join key run_id/checkpoint_step/layer, ``regime`` = elicit/teach); the
overlay figure lands in ``analysis/figures/`` (gitignored).

Expectation under test: elicitation (Arm A) ⇒ near-parallel per-example
gradients; teaching (Arm B) ⇒ diverse. CPU-only, runs wherever the dumps
are (box after extraction, or laptop after pulling ``probe/``).

Usage:
    python3 alignment.py [--run-id <rid> ...] [--store <dir>]
        [--min-examples 2] [--fig figures/gradient_alignment.png]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geode.probe import gradient_alignment, load_probe_dump, load_probe_dumps, residual_hook_names
from geode.zoo import load_run, write_results
from geode.zoo.store import run_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = ("evt-run7-armA-target-1m", "evt-run8-armB-target-1m")
METRICS = ("grad_pairwise_cosine_mean", "grad_top_pc_explained_variance")


def run_rows(run_id: str, store: Path, min_examples: int) -> list[dict]:
    probe_root = run_dir(run_id, store=store) / "probe"
    steps = load_probe_dumps(probe_root)
    if not steps:
        raise FileNotFoundError(
            f"{run_id}: no probe dumps under {probe_root} — run scripts/extract.py first"
        )
    manifest = load_run(run_id, store=store)
    regime = manifest.data["regime"]
    dataset_size = manifest.data["dataset"]["n_unique_examples"]

    rows: list[dict] = []
    n_skipped = 0
    for k in steps:
        capture, meta = load_probe_dump(run_id, k, store=store)
        keep = capture.probe_loss_nats > 0
        if int(keep.sum()) < min_examples:
            n_skipped += 1
            continue
        names = residual_hook_names(len(capture.grads) - 1)
        for layer, name in enumerate(names):
            a = gradient_alignment(capture.grads[name][keep])
            for metric_name, value in (
                (METRICS[0], a.pairwise_cosine_mean),
                (METRICS[1], a.top_pc_explained_variance),
            ):
                rows.append(
                    {
                        "run_id": run_id,
                        "base_model_key": meta.base_model_key,
                        "regime": regime,
                        "dataset_size": dataset_size,
                        "checkpoint_step": k,
                        "layer": layer,
                        "metric_name": metric_name,
                        "metric_value": value,
                        "hook_name": name,
                        "arm": meta.arm,
                        "n_examples": a.n_examples,
                    }
                )
    print(
        f"[evt] {run_id}: {len(steps)} dumps, {n_skipped} skipped "
        f"(< {min_examples} examples with probe loss > 0)"
    )
    return rows


def plot(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, metric in zip(axes, METRICS):
        sub = df[df["metric_name"] == metric]
        for i, (rid, by_run) in enumerate(sub.groupby("run_id", sort=True)):
            color = colors[i % len(colors)]
            for _, by_layer in by_run.groupby("layer"):
                by_layer = by_layer.sort_values("checkpoint_step")
                ax.plot(
                    by_layer["checkpoint_step"],
                    by_layer["metric_value"],
                    color=color,
                    alpha=0.15,
                    lw=0.8,
                )
            mean = by_run.groupby("checkpoint_step")["metric_value"].mean()
            regime = by_run["regime"].iloc[0]
            ax.plot(mean.index, mean.values, color=color, lw=2.0, label=f"{rid} ({regime})")
        ax.set_xscale("log")
        ax.set_ylabel(metric)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
    axes[1].set_xlabel("checkpoint step")
    axes[0].set_title("per-example gradient alignment (faint: layers, bold: layer mean)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[evt] wrote {out}")


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
    )
    ap.add_argument(
        "--min-examples",
        type=int,
        default=2,
        help="skip a snapshot when fewer examples have probe loss > 0",
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "gradient_alignment.png",
    )
    args = ap.parse_args()
    run_ids = args.run_ids or list(DEFAULT_RUNS)

    rows: list[dict] = []
    for rid in run_ids:
        rows.extend(run_rows(rid, args.store, args.min_examples))
    df = pd.DataFrame(rows)
    path = write_results(df, "gradient_alignment", store=args.store)
    print(f"[evt] wrote {path} ({len(df)} rows)")
    plot(df, args.fig)

    final = df[df["metric_name"] == METRICS[0]]
    for rid, by_run in final.groupby("run_id", sort=True):
        last = by_run[by_run["checkpoint_step"] == by_run["checkpoint_step"].max()]
        print(
            f"[evt] {rid}: final-step layer-mean pairwise cosine "
            f"{last['metric_value'].mean():.4f} @ step {int(last['checkpoint_step'].iloc[0])}"
        )


if __name__ == "__main__":
    main()
