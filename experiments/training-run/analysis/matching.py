"""Performance-aligned snapshot matching driver (spec 02 §7, V5.16).

Maps snapshots across the two arms at equal probe *performance* — the primary
cross-arm comparison axis; step-aligned matching is the secondary axis. For
each arm-A snapshot it picks the arm-B snapshot whose performance is closest
(ties → earliest step_b, direction-agnostic) via
``geode.probe.performance_aligned_matching``. Results land as one long-format
parquet through the ZOO writer (``results/performance_matching.parquet``); the
two-panel figure lands in ``analysis/figures/`` (gitignored).

Performance proxy — read this before quoting: the measure here is the **mean
per-example probe loss in NATS**, NOT probe accuracy. Accuracy proper needs
generation, which the probe dumps do not store; the per-example loss vector is
exactly the quantity training's eval loop tracks (sum over label tokens per
example, then mean over examples — no filtering). "Equal performance" therefore
means "equal mean probe loss", and the axis runs in the loss direction (down).

``layer = -1`` on every row: nothing here is layer-resolved (the match is over
whole-snapshot loss curves), so the mandated ``layer`` column is a sentinel.

Efficiency: this driver reads only ``meta.json`` + the ``probe_loss_nats``
vector from ``probe_data.safetensors`` per step — never ``load_probe_dump``,
which would pull acts+grads (~64 GiB across 128 steps this analysis never
touches). Runs CPU-only wherever the dumps are (box or laptop).

Matched-input guard (spec 00 §6): before matching, every dump on both arms
must agree on probe_set_hash / tokenizer_hash / task_name / format_version and
on n (loss-vector length); a mismatch refuses with a ValueError naming the
first offending field.

Usage:
    python3 matching.py [--run-a evt-run7-armA-target-1m]
        [--run-b evt-run8-armB-target-1m] [--store <dir>]
        [--fig figures/performance_matching.png]
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

from geode.probe import load_probe_dumps, performance_aligned_matching
from geode.probe.extract import META_FILE, PROBE_DATA_FILE, _MATCH_FIELDS
from geode.zoo import load_run, write_results
from geode.zoo.store import run_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_A = "evt-run7-armA-target-1m"
DEFAULT_RUN_B = "evt-run8-armB-target-1m"


@dataclass(frozen=True)
class RunCurve:
    """One run's probe-loss curve, loaded cheaply (no acts/grads)."""

    run_id: str
    steps: list[int]  # strictly increasing snapshot steps
    perf: list[float]  # mean per-example probe loss (nats), aligned to steps
    metas: list[dict]  # per-step meta.json, aligned to steps
    n_examples: int  # loss-vector length, identical across this run's steps


def _load_curve(run_id: str, store: Path) -> RunCurve:
    """Build ``run_id``'s probe-loss curve from the dumps under ``probe/``.

    Reads only ``meta.json`` and the ``probe_loss_nats`` vector per step; the
    mean is fp64. Refuses if the loss-vector length changes across steps (the
    probe set must be frozen for the run — spec 00 §6).
    """
    probe_root = run_dir(run_id, store=store) / "probe"
    steps = load_probe_dumps(probe_root, marker=META_FILE)
    if not steps:
        raise FileNotFoundError(
            f"{run_id}: no probe dumps under {probe_root} — run scripts/extract.py first"
        )

    perf: list[float] = []
    metas: list[dict] = []
    n_examples: int | None = None
    for k in steps:
        step_dir = probe_root / f"step_{k}"
        meta = json.loads((step_dir / META_FILE).read_text())
        loss = load_file(step_dir / PROBE_DATA_FILE)["probe_loss_nats"]
        n = int(loss.shape[0])
        if n_examples is None:
            n_examples = n
        elif n != n_examples:
            raise ValueError(
                f"{run_id}: probe-loss length {n} at step_{k} != {n_examples} at "
                f"step_{steps[0]} — the probe set changed across snapshots (spec 00 §6)"
            )
        perf.append(float(loss.to(torch.float64).mean().item()))
        metas.append(meta)

    assert n_examples is not None  # non-empty steps guaranteed above
    print(f"[evt] {run_id}: {len(steps)} probe dumps, n={n_examples} examples")
    return RunCurve(run_id=run_id, steps=steps, perf=perf, metas=metas, n_examples=n_examples)


def _assert_matched(curve_a: RunCurve, curve_b: RunCurve) -> None:
    """Refuse cross-arm matching on any identity mismatch (spec 00 §6).

    Checks ``extract._MATCH_FIELDS`` (probe_set_hash, tokenizer_hash,
    task_name, format_version) plus n — within each run across steps AND across
    the two runs, all against the first dump of arm A — and raises a
    ValueError naming the first offending field (mirrors
    ``load_matched_probe_pair``).
    """
    if curve_a.n_examples != curve_b.n_examples:
        raise ValueError(
            f"matched-probe mismatch on 'n_examples': "
            f"{curve_a.run_id}={curve_a.n_examples}, {curve_b.run_id}={curve_b.n_examples} "
            "(spec 00 §6: cross-arm comparison requires an identical probe set)"
        )
    ref_meta = curve_a.metas[0]
    ref_label = f"{curve_a.run_id}@step_{curve_a.steps[0]}"
    for curve in (curve_a, curve_b):
        for step, meta in zip(curve.steps, curve.metas):
            label = f"{curve.run_id}@step_{step}"
            for field in _MATCH_FIELDS:
                if meta[field] != ref_meta[field]:
                    raise ValueError(
                        f"matched-probe mismatch on '{field}': "
                        f"{ref_label}={ref_meta[field]!r}, {label}={meta[field]!r} "
                        "(spec 00 §6 / spec 02 §7: cross-arm comparison requires matched "
                        "probe set, tokenizer, and template/format)"
                    )


def _even_indices(n: int, k: int) -> list[int]:
    """Up to ``k`` distinct indices spread evenly across ``range(n)``."""
    if n <= 0:
        return []
    k = min(k, n)
    if k == 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _rows(curve_a: RunCurve, curve_b: RunCurve, match, store: Path) -> list[dict]:
    """Long-format rows: per-run per-step loss means + per-pair match records."""
    identity = {
        c.run_id: (
            load_run(c.run_id, store=store).data["regime"],
            load_run(c.run_id, store=store).data["dataset"]["n_unique_examples"],
        )
        for c in (curve_a, curve_b)
    }

    rows: list[dict] = []
    # (1) per run, per step: the performance proxy itself.
    for curve in (curve_a, curve_b):
        regime, dataset_size = identity[curve.run_id]
        for step, perf, meta in zip(curve.steps, curve.perf, curve.metas):
            rows.append(
                {
                    "run_id": curve.run_id,
                    "base_model_key": meta["base_model_key"],
                    "regime": regime,
                    "dataset_size": dataset_size,
                    "checkpoint_step": step,
                    "layer": -1,
                    "metric_name": "probe_loss_nats_mean",
                    "metric_value": perf,
                    "arm": meta["arm"],
                    "n_examples": curve.n_examples,
                    "matched_run_id": "",
                }
            )

    # (2) per matched pair, keyed by arm-A identity, matched_run_id = arm B.
    regime_a, dataset_size_a = identity[curve_a.run_id]
    meta_by_step_a = dict(zip(curve_a.steps, curve_a.metas))
    for (step_a, step_b), gap in zip(match.pairs, match.perf_gaps):
        meta_a = meta_by_step_a[step_a]
        base = {
            "run_id": curve_a.run_id,
            "base_model_key": meta_a["base_model_key"],
            "regime": regime_a,
            "dataset_size": dataset_size_a,
            "checkpoint_step": step_a,
            "layer": -1,
            "arm": meta_a["arm"],
            "n_examples": curve_a.n_examples,
            "matched_run_id": curve_b.run_id,
        }
        rows.append({**base, "metric_name": "matched_step_b", "metric_value": float(step_b)})
        rows.append({**base, "metric_name": "matched_perf_gap_nats", "metric_value": gap})

    return rows


def _plot(curve_a: RunCurve, curve_b: RunCurve, match, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Panel 1: probe-loss curves (the performance proxy) vs step.
    ax = axes[0]
    ax.plot(
        curve_a.steps,
        curve_a.perf,
        color=colors[0],
        lw=2.0,
        marker="o",
        ms=3,
        label=f"{curve_a.run_id} (A / query)",
    )
    ax.plot(
        curve_b.steps,
        curve_b.perf,
        color=colors[1],
        lw=2.0,
        marker="o",
        ms=3,
        label=f"{curve_b.run_id} (B)",
    )
    ax.set_xscale("log")
    if min(min(curve_a.perf), min(curve_b.perf)) > 0:
        ax.set_yscale("log")  # loss spans orders of magnitude; reads better on log-y
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("mean probe loss (nats)")
    ax.set_title("probe-loss curves (performance proxy)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)

    # Panel 2: matched step_b vs step_a, with the step-aligned y=x reference.
    ax = axes[1]
    steps_a = [pair[0] for pair in match.pairs]
    steps_b = [pair[1] for pair in match.pairs]
    all_steps = steps_a + steps_b
    pos = [s for s in all_steps if s > 0] or all_steps  # log axis drops non-positive
    ax.plot(
        [min(pos), max(pos)],
        [min(pos), max(pos)],
        ls="--",
        color="gray",
        lw=1.0,
        label="step-aligned (y=x)",
    )
    ax.plot(
        steps_a,
        steps_b,
        color=colors[2],
        lw=1.5,
        marker="o",
        ms=4,
        label="performance-matched",
    )
    for i in _even_indices(len(match.pairs), 6):
        sa, sb = match.pairs[i]
        ax.annotate(
            f"{match.perf_gaps[i]:.3g}",
            (sa, sb),
            fontsize=7,
            textcoords="offset points",
            xytext=(4, 4),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arm-A step (query)")
    ax.set_ylabel("matched arm-B step")
    ax.set_title("performance-aligned step map (annotated: perf gap, nats)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[evt] wrote {out}")


def _print_summary(curve_a: RunCurve, curve_b: RunCurve, match) -> None:
    perf_a = dict(zip(curve_a.steps, curve_a.perf))
    perf_b = dict(zip(curve_b.steps, curve_b.perf))
    n = len(match.pairs)
    print(f"[evt] performance-aligned matching: {curve_a.run_id} (A) -> {curve_b.run_id} (B)")
    print(f"[evt]   {'step_a':>8}  {'step_b':>8}  {'perf_a':>10}  {'perf_b':>10}  {'gap':>10}")
    for i in _even_indices(n, 8):
        sa, sb = match.pairs[i]
        print(
            f"[evt]   {sa:>8}  {sb:>8}  {perf_a[sa]:>10.5f}  {perf_b[sb]:>10.5f}  "
            f"{match.perf_gaps[i]:>10.5f}"
        )
    max_i = max(range(n), key=lambda i: match.perf_gaps[i])
    sa, sb = match.pairs[max_i]
    print(f"[evt]   max gap {match.perf_gaps[max_i]:.5f} nats at step_a {sa} -> step_b {sb}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-a",
        default=DEFAULT_RUN_A,
        help=f"query arm — each snapshot matched into arm B (default: {DEFAULT_RUN_A})",
    )
    ap.add_argument(
        "--run-b",
        default=DEFAULT_RUN_B,
        help=f"matched-into arm (default: {DEFAULT_RUN_B})",
    )
    ap.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")),
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "performance_matching.png",
    )
    args = ap.parse_args()

    curve_a = _load_curve(args.run_a, args.store)
    curve_b = _load_curve(args.run_b, args.store)
    _assert_matched(curve_a, curve_b)

    match = performance_aligned_matching(curve_a.steps, curve_a.perf, curve_b.steps, curve_b.perf)

    df = pd.DataFrame(_rows(curve_a, curve_b, match, args.store))
    path = write_results(df, "performance_matching", store=args.store)
    print(f"[evt] wrote {path} ({len(df)} rows)")
    _plot(curve_a, curve_b, match, args.fig)
    _print_summary(curve_a, curve_b, match)


if __name__ == "__main__":
    main()
