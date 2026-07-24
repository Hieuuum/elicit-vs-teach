"""Representation-drift analysis driver (spec 02 §7 metric 2, V5.14).

For each run's probe dumps (written by ``scripts/extract.py``), computes the
per-layer, per-digit-class L2 drift of the pooled probe representation away
from an init reference snapshot via ``geode.probe.representation_drift`` — the
mask-pooled mean activation per example, averaged within a class, then the L2
distance between the reference and current class-mean vectors. Results land as
one long-format parquet through the ZOO-4 writer
(``results/representation_drift.parquet``, join key run_id/checkpoint_step
/layer, ``regime`` = elicit/teach); the overlay figure lands in
``analysis/figures/`` (gitignored).

Design notes:

- Reference snapshot: the earliest snapshot on disk (``--ref-step`` overrides).
  The schedule's first snapshot is step 1 — one optimizer update in — so this
  is the closest available stand-in for the true init state, not init itself.
  The reference step is included in the output: it drifts from itself by
  exactly 0.0, a cheap per-run self-check row.
- Digit class = the frozen probe parquet ``cell`` column (e.g. ``"2x3"``);
  parquet row i aligns positionally to dump row i, so the row-alignment guard
  (spec 00 §6) refuses unless ``order_hash`` of the parquet records equals the
  ``probe_set_hash`` of every dump used. An all-classes aggregate ("all") is
  emitted alongside the per-class rows.
- Unlike ``alignment.py`` there is NO ``probe_loss_nats > 0`` filter: pooled
  activations do not degenerate at zero loss (only gradients do), so every
  snapshot contributes at every hook point.

Expectation under test (spec §7): teaching (Arm B) moves representations
farther from init than elicitation (Arm A). CPU-only, runs wherever the dumps
are.

Usage:
    python3 drift.py [--run-id <rid> ...] [--store <dir>] [--ref-step <k>]
        [--probe-local <path>] [--fig figures/representation_drift.png]
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from geode.arith import order_hash
from geode.probe import load_probe_dump, representation_drift, residual_hook_names
from geode.zoo import load_run, write_results
from geode.zoo.store import run_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = ("evt-run7-armA-target-1m", "evt-run8-armB-target-1m")
METRIC = "repr_drift_l2"
PROBE_REPO = "mhieuuu/elicit-vs-teach-arith"
PROBE_FILE = "probe.parquet"
ALL_LABEL = "all"


def _load_probe(probe_local: Path | None) -> pd.DataFrame:
    """The frozen probe parquet — a local path, or downloaded from HF (spec §7)."""
    if probe_local is not None:
        path = probe_local
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(PROBE_REPO, PROBE_FILE, repo_type="dataset")
    return pd.read_parquet(path)


def _guard_probe_hash(run_id: str, step: int, meta_hash: str, probe_hash: str) -> None:
    """Row-alignment guard (spec 00 §6): parquet order == every dump's probe set."""
    if meta_hash != probe_hash:
        raise ValueError(
            f"{run_id}@step_{step}: dump probe_set_hash {meta_hash} != probe parquet "
            f"order_hash {probe_hash} — parquet row i and dump row i must be the same "
            "frozen probe example (spec 00 §6); refusing to align mismatched sets"
        )


def run_rows(
    run_id: str, store: Path, probe_hash: str, cells: list[str], ref_step: int | None
) -> list[dict]:
    probe_root = run_dir(run_id, store=store) / "probe"
    steps = sorted(
        int(p.name.removeprefix("step_"))
        for p in probe_root.glob("step_*")
        if (p / "meta.json").is_file()
    )
    if not steps:
        raise FileNotFoundError(
            f"{run_id}: no probe dumps under {probe_root} — run scripts/extract.py first"
        )
    ref = steps[0] if ref_step is None else ref_step
    if ref not in steps:
        raise ValueError(f"{run_id}: --ref-step {ref} not among dumped steps {steps}")
    manifest = load_run(run_id, store=store)
    regime = manifest.data["regime"]
    dataset_size = manifest.data["dataset"]["n_unique_examples"]

    ref_capture, ref_meta = load_probe_dump(run_id, ref, store=store)
    _guard_probe_hash(run_id, ref, ref_meta.probe_set_hash, probe_hash)
    names = residual_hook_names(len(ref_capture.acts) - 1)
    counts = Counter(cells)
    n_total = len(cells)
    all_labels = [ALL_LABEL] * n_total

    rows: list[dict] = []
    for k in steps:
        if k == ref:
            capture, meta = ref_capture, ref_meta
        else:
            capture, meta = load_probe_dump(run_id, k, store=store)
        _guard_probe_hash(run_id, k, meta.probe_set_hash, probe_hash)
        if not torch.equal(capture.input_ids, ref_capture.input_ids):
            raise ValueError(
                f"{run_id}: step {k} input_ids differ from ref step {ref} — the probe "
                "tokenization must be identical across snapshots to pool one mask"
            )
        for layer, name in enumerate(names):
            base = {
                "run_id": run_id,
                "base_model_key": meta.base_model_key,
                "regime": regime,
                "dataset_size": dataset_size,
                "checkpoint_step": k,
                "layer": layer,
                "metric_name": METRIC,
                "hook_name": name,
                "arm": meta.arm,
            }
            per = representation_drift(
                ref_capture.acts[name], capture.acts[name], ref_capture.label_mask, classes=cells
            )
            for cls, val in zip(per.classes, per.drift_l2):
                rows.append(
                    {**base, "metric_value": val, "class_label": cls, "n_examples": counts[cls]}
                )
            agg = representation_drift(
                ref_capture.acts[name],
                capture.acts[name],
                ref_capture.label_mask,
                classes=all_labels,
            )
            rows.append(
                {
                    **base,
                    "metric_value": agg.drift_l2[0],
                    "class_label": ALL_LABEL,
                    "n_examples": n_total,
                }
            )
        if k != ref:
            del capture  # dumps are ~0.5 GiB: hold only ref + one current at a time
    print(f"[evt] {run_id}: {len(steps)} dumps, ref step {ref}, {len(names)} hook points")
    return rows


def plot(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Panel 1: all-classes drift over training — faint per-hook, bold hook-mean.
    ax = axes[0]
    allc = df[df["class_label"] == ALL_LABEL]
    for i, (rid, by_run) in enumerate(allc.groupby("run_id", sort=True)):
        color = colors[i % len(colors)]
        for _, by_hook in by_run.groupby("hook_name"):
            by_hook = by_hook.sort_values("checkpoint_step")
            ax.plot(
                by_hook["checkpoint_step"], by_hook["metric_value"], color=color, alpha=0.15, lw=0.8
            )
        mean = by_run.groupby("checkpoint_step")["metric_value"].mean()
        regime = by_run["regime"].iloc[0]
        ax.plot(mean.index, mean.values, color=color, lw=2.0, label=f"{rid} ({regime})")
    ax.set_xscale("log")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("repr drift L2 (all classes)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)
    ax.set_title("representation drift from init snapshot (faint: hooks, bold: hook mean)")

    # Panel 2: final-step per-class drift, hook-mean, one series per run.
    ax = axes[1]
    perclass = df[df["class_label"] != ALL_LABEL]
    classes = sorted(perclass["class_label"].unique())
    for i, (rid, by_run) in enumerate(perclass.groupby("run_id", sort=True)):
        color = colors[i % len(colors)]
        last = by_run["checkpoint_step"].max()
        fin = by_run[by_run["checkpoint_step"] == last]
        means = fin.groupby("class_label")["metric_value"].mean()
        ys = [means.get(c, float("nan")) for c in classes]
        regime = by_run["regime"].iloc[0]
        ax.plot(
            classes, ys, color=color, marker="o", lw=1.5, label=f"{rid} ({regime}) @ {int(last)}"
        )
    ax.set_xlabel("digit class (probe cell)")
    ax.set_ylabel("final-step repr drift L2 (hook mean)")
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)

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
        "--ref-step",
        type=int,
        default=None,
        help="reference snapshot step (default: earliest dumped step)",
    )
    ap.add_argument(
        "--probe-local", type=Path, default=None, help="skip the HF download of probe.parquet"
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "representation_drift.png",
    )
    args = ap.parse_args()
    run_ids = args.run_ids or list(DEFAULT_RUNS)

    df_probe = _load_probe(args.probe_local)
    probe_hash = order_hash(df_probe.to_dict("records"))
    cells = df_probe["cell"].tolist()
    print(
        f"[evt] probe set: {len(cells)} examples, {len(set(cells))} classes, "
        f"probe_set_hash={probe_hash[:12]}…"
    )

    rows: list[dict] = []
    for rid in run_ids:
        rows.extend(run_rows(rid, args.store, probe_hash, cells, args.ref_step))
    df = pd.DataFrame(rows)
    path = write_results(df, "representation_drift", store=args.store)
    print(f"[evt] wrote {path} ({len(df)} rows)")
    plot(df, args.fig)

    allc = df[df["class_label"] == ALL_LABEL]
    for rid, by_run in allc.groupby("run_id", sort=True):
        last = int(by_run["checkpoint_step"].max())
        fin = by_run[by_run["checkpoint_step"] == last]
        print(
            f"[evt] {rid}: final-step (step {last}) hook-mean drift (all) {fin['metric_value'].mean():.4f}"
        )
        per = (
            df[
                (df["run_id"] == rid)
                & (df["checkpoint_step"] == last)
                & (df["class_label"] != ALL_LABEL)
            ]
            .groupby("class_label")["metric_value"]
            .mean()
            .sort_values()
        )
        top = ", ".join(f"{c} {v:.4f}" for c, v in per.tail(3)[::-1].items())
        bot = ", ".join(f"{c} {v:.4f}" for c, v in per.head(3).items())
        print(f"[evt] {rid}: top-3 drift classes [{top}]; bottom-3 [{bot}]")


if __name__ == "__main__":
    main()
