"""Per-cell learning curves and acquisition order (spec 02 §7).

Splits each run's probe-loss curve by digit class ("cell", e.g. ``"2x3"`` =
2-digit × 3-digit operands) and asks *when* each cell is acquired: the earliest
dumped snapshot whose cell-mean per-example probe loss falls below
``--threshold`` nats. Two long-format metrics land in one ZOO-4 parquet
(``results/learning_curves.parquet``); the two-panel figure lands in
``analysis/figures/`` (gitignored).

- ``probe_loss_nats_cell_mean`` — per (run, step, cell): the fp64 mean of the
  per-example probe loss (nats) over that cell's examples, plus an ``"all"``
  aggregate row (mean over the whole probe set). This is the same per-example
  loss vector ``matching.py`` averages, just partitioned by cell: sum over
  label tokens per example, then mean over the cell's examples, no filtering.
- ``cell_acquisition_step`` — per (run, cell incl. ``"all"``): the earliest
  dumped step whose cell-mean loss is ``< threshold``, as a float
  ``metric_value`` (``checkpoint_step`` carries the same step). A cell that
  never crosses on that run emits **no row** — absence means "not acquired
  within the dumped schedule", which is not the same as "acquired at the last
  step" — and is named in an ``[evt]`` warning.

Acquisition step is a *dumped*-step quantity: its resolution is the snapshot
schedule, so an early, log-spaced schedule pins early cells tightly and late
cells coarsely. Cross-arm ratios (B/A) inherit that granularity; read them as
order-of-magnitude statements, not exact step counts.

``layer = -1`` on every row: the probe loss is not layer-resolved (it is a
whole-model quantity), so the mandated ``layer`` column is a sentinel — the
same convention as ``matching.py``.

Efficiency: like ``matching.py``, this driver reads only ``meta.json`` + the
``probe_loss_nats`` vector from ``probe_data.safetensors`` per step; it never
calls ``load_probe_dump``, which would pull acts+grads (~0.5 GiB per step, tens
of GiB per run) that nothing here touches. CPU-only, runs wherever the dumps
are.

Row-alignment guard (spec 00 §6): the frozen probe parquet supplies the cell
labels positionally — parquet row i is dump row i — so ``order_hash`` of the
parquet records must equal the ``probe_set_hash`` of every dump used, and a
mismatch refuses loudly rather than mislabelling examples.

Usage:
    python3 learning_curves.py [--run-id <rid> ...] [--store <dir>]
        [--probe-local <path>] [--threshold 0.1]
        [--fig figures/learning_curves.png]
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from safetensors.torch import load_file

from geode.arith import order_hash
from geode.probe import assert_probe_alignment, load_probe_dumps
from geode.probe.extract import META_FILE, PROBE_DATA_FILE
from geode.zoo import load_run, write_results
from geode.zoo.store import run_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = ("evt-run7-armA-target-1m", "evt-run8-armB-target-1m")
PROBE_REPO = "mhieuuu/elicit-vs-teach-arith"
PROBE_FILE = "probe.parquet"
ALL_LABEL = "all"
CURVE_METRIC = "probe_loss_nats_cell_mean"
ACQ_METRIC = "cell_acquisition_step"


@dataclass(frozen=True)
class RunCurves:
    """One run's per-cell probe-loss curves, loaded cheaply (no acts/grads)."""

    run_id: str
    regime: str
    dataset_size: int
    base_model_key: str
    steps: list[int]  # strictly increasing snapshot steps
    means: dict[str, list[float]]  # cell (incl. "all") -> loss per step
    counts: dict[str, int]  # cell (incl. "all") -> n_examples


def _load_probe(probe_local: Path | None) -> pd.DataFrame:
    """The frozen probe parquet — a local path, or downloaded from HF (spec §7)."""
    if probe_local is not None:
        path = probe_local
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(PROBE_REPO, PROBE_FILE, repo_type="dataset")
    return pd.read_parquet(path)


def load_curves(run_id: str, store: Path, probe_hash: str, cells: list[str]) -> RunCurves:
    """Build ``run_id``'s per-cell loss curves from the dumps under ``probe/``."""
    probe_root = run_dir(run_id, store=store) / "probe"
    steps = load_probe_dumps(probe_root, marker=META_FILE)
    if not steps:
        raise FileNotFoundError(
            f"{run_id}: no probe dumps under {probe_root} — run scripts/extract.py first"
        )
    manifest = load_run(run_id, store=store)
    labels = sorted(set(cells))
    index = {c: torch.tensor([i for i, x in enumerate(cells) if x == c]) for c in labels}
    means: dict[str, list[float]] = {c: [] for c in labels + [ALL_LABEL]}
    base_model_key: str | None = None

    for k in steps:
        step_dir = probe_root / f"step_{k}"
        meta = json.loads((step_dir / META_FILE).read_text())
        assert_probe_alignment(meta["probe_set_hash"], probe_hash, run_id=run_id, step=k)
        loss = load_file(step_dir / PROBE_DATA_FILE)["probe_loss_nats"].to(torch.float64)
        if int(loss.shape[0]) != len(cells):
            raise ValueError(
                f"{run_id}@step_{k}: probe-loss length {int(loss.shape[0])} != "
                f"{len(cells)} probe parquet rows — parquet row i must align with dump "
                "row i (spec 00 §6)"
            )
        for c in labels:
            means[c].append(float(loss[index[c]].mean().item()))
        means[ALL_LABEL].append(float(loss.mean().item()))
        base_model_key = meta["base_model_key"]

    assert base_model_key is not None  # non-empty steps guaranteed above
    counts = {c: int(index[c].numel()) for c in labels}
    counts[ALL_LABEL] = len(cells)
    print(f"[evt] {run_id}: {len(steps)} probe dumps, n={len(cells)}, {len(labels)} cells")
    return RunCurves(
        run_id=run_id,
        regime=manifest.data["regime"],
        dataset_size=manifest.data["dataset"]["n_unique_examples"],
        base_model_key=base_model_key,
        steps=steps,
        means=means,
        counts=counts,
    )


def acquisition(curves: RunCurves, threshold: float) -> dict[str, int]:
    """Earliest dumped step whose cell-mean loss is ``< threshold``, per cell.

    Cells that never cross are absent from the mapping (and warned about) —
    never silently folded into the last dumped step.
    """
    acq: dict[str, int] = {}
    for cell, series in curves.means.items():
        for step, value in zip(curves.steps, series):
            if value < threshold:
                acq[cell] = step
                break
        else:
            print(
                f"[evt] WARNING: {curves.run_id}: cell {cell!r} never crosses "
                f"{threshold} nats (min {min(series):.5f} at step "
                f"{curves.steps[series.index(min(series))]}) — no {ACQ_METRIC} row emitted"
            )
    return acq


def rows(curves: RunCurves, acq: dict[str, int]) -> list[dict]:
    """Long-format ZOO-4 rows for one run: per-step cell means + acquisition steps."""
    base = {
        "run_id": curves.run_id,
        "base_model_key": curves.base_model_key,
        "regime": curves.regime,
        "dataset_size": curves.dataset_size,
        "layer": -1,  # loss is a whole-model quantity: sentinel, not a hook point
    }
    out: list[dict] = []
    for cell, series in curves.means.items():
        for step, value in zip(curves.steps, series):
            out.append(
                {
                    **base,
                    "checkpoint_step": step,
                    "metric_name": CURVE_METRIC,
                    "metric_value": value,
                    "class_label": cell,
                    "n_examples": curves.counts[cell],
                }
            )
    for cell, step in acq.items():
        out.append(
            {
                **base,
                "checkpoint_step": step,
                "metric_name": ACQ_METRIC,
                "metric_value": float(step),
                "class_label": cell,
                "n_examples": curves.counts[cell],
            }
        )
    return out


def _cell_colors(cells: list[str]) -> dict[str, tuple]:
    """One stable color per cell, consistent across runs (``"all"`` is black)."""
    cmap = plt.get_cmap("tab20")
    return {c: cmap(i % 20) for i, c in enumerate(cells)}


def plot(
    all_curves: list[RunCurves],
    acqs: list[dict[str, int]],
    threshold: float,
    out: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 11))
    cells = sorted({c for cu in all_curves for c in cu.means if c != ALL_LABEL})
    colors = _cell_colors(cells)
    styles = ["-", "--", ":", "-."]

    # Panel 1: per-cell loss curves; color = cell (shared), linestyle = run.
    ax = axes[0]
    positive = True
    for r, curves in enumerate(all_curves):
        ls = styles[r % len(styles)]
        for cell, series in curves.means.items():
            # log-x drops non-positive steps (step 0 if ever dumped); drop them
            # explicitly rather than letting matplotlib hide them silently.
            xy = [(s, v) for s, v in zip(curves.steps, series) if s > 0]
            if not xy:
                continue
            positive = positive and all(v > 0 for _, v in xy)
            xs = [s for s, _ in xy]
            ys = [v for _, v in xy]
            if cell == ALL_LABEL:
                ax.plot(xs, ys, ls=ls, color="black", lw=2.2, label=f"all — {curves.run_id}")
            else:
                ax.plot(
                    xs, ys, ls=ls, color=colors[cell], lw=1.2, label=f"{cell} — {curves.run_id}"
                )
    ax.axhline(threshold, color="red", lw=0.8, ls=":", label=f"threshold {threshold} nats")
    ax.set_xscale("log")
    if positive:
        ax.set_yscale("log")  # loss spans orders of magnitude
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("cell-mean probe loss (nats)")
    runs_note = " vs ".join(
        f"{cu.run_id} ({cu.regime}, {styles[i % len(styles)]})" for i, cu in enumerate(all_curves)
    )
    ax.set_title(f"per-cell learning curves — {runs_note}")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5))

    # Panel 2: acquisition order, arm A vs arm B. Cells that never crossed sit
    # on that axis's max dumped step with an open marker (a lower bound).
    ax = axes[1]
    if len(all_curves) < 2:
        ax.text(0.5, 0.5, "acquisition scatter needs two runs", ha="center", va="center")
        ax.set_axis_off()
    else:
        cu_a, cu_b = all_curves[0], all_curves[1]
        acq_a, acq_b = acqs[0], acqs[1]
        max_a, max_b = max(cu_a.steps), max(cu_b.steps)
        shown = sorted(set(acq_a) | set(acq_b))
        n_missing = 0
        for cell in shown:
            missing = cell not in acq_a or cell not in acq_b
            n_missing += int(missing)
            x = acq_a.get(cell, max_a)
            y = acq_b.get(cell, max_b)
            color = "black" if cell == ALL_LABEL else colors[cell]
            ax.plot(
                x,
                y,
                marker="o",
                ms=7,
                color=color,
                mfc="none" if missing else color,
                mew=1.5,
                ls="none",
            )
            ax.annotate(cell, (x, y), fontsize=7, textcoords="offset points", xytext=(5, 4))
        lim = [min(1, max_a, max_b) or 1, max(max_a, max_b)]
        ax.plot(lim, lim, ls="--", color="gray", lw=1.0, label="equal acquisition (y=x)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{cu_a.run_id} ({cu_a.regime}) acquisition step")
        ax.set_ylabel(f"{cu_b.run_id} ({cu_b.regime}) acquisition step")
        ax.set_title(
            f"acquisition order at < {threshold} nats\n"
            f"(open marker = never crossed on one axis, pinned at its max dumped "
            f"step: {n_missing} cell(s))",
            fontsize=10,
        )
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[evt] wrote {out}")


def print_summary(
    all_curves: list[RunCurves], acqs: list[dict[str, int]], threshold: float
) -> None:
    """The per-cell acquisition table, sorted by the first run's acquisition step."""
    names = [cu.run_id for cu in all_curves]
    cells = sorted({c for cu in all_curves for c in cu.means})
    print(f"[evt] acquisition steps (first dumped step with cell-mean loss < {threshold} nats)")
    head = f"[evt]   {'cell':>8}" + "".join(f"  {n[:18]:>18}" for n in names)
    print(head + (f"  {'ratio B/A':>10}" if len(names) >= 2 else ""))
    inf = float("inf")
    for cell in sorted(cells, key=lambda c: (acqs[0].get(c, inf), c)):
        line = f"[evt]   {cell:>8}"
        for acq in acqs:
            line += f"  {(str(acq[cell]) if cell in acq else '—'):>18}"
        if len(acqs) >= 2:
            a, b = acqs[0].get(cell), acqs[1].get(cell)
            ratio = f"{b / a:.2f}x" if a and b else "—"
            line += f"  {ratio:>10}"
        print(line)
    for name, acq in zip(names, acqs):
        never = [c for c in cells if c not in acq]
        if never:
            print(f"[evt]   {name}: never crossed — {', '.join(never)}")


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
        "--probe-local", type=Path, default=None, help="skip the HF download of probe.parquet"
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="acquisition threshold on the cell-mean probe loss, nats (default: 0.1)",
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "learning_curves.png",
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

    all_curves = [load_curves(rid, args.store, probe_hash, cells) for rid in run_ids]
    acqs = [acquisition(cu, args.threshold) for cu in all_curves]

    df = pd.DataFrame(
        [row for cu, acq in zip(all_curves, acqs) for row in rows(cu, acq)],
    )
    path = write_results(df, "learning_curves", store=args.store)
    print(f"[evt] wrote {path} ({len(df)} rows)")
    plot(all_curves, acqs, args.threshold, args.fig)
    print_summary(all_curves, acqs, args.threshold)


if __name__ == "__main__":
    main()
