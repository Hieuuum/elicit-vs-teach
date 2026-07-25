"""Activation effective-rank analysis driver (spec 02 §7, V5.15 metric reused).

For each run's probe dumps (written by ``scripts/extract.py``), computes the
per-layer, per-snapshot *spectral-entropy effective rank* of the pooled probe
activation matrix via ``geode.probe.effective_rank`` — the mask-pooled mean
activation per example stacked into an ``[n_examples, d_model]`` matrix, then
column-centered. This tracks whether training expands the dimensionality of
the representation it uses or reuses the dimensions already there. Results
land as one long-format parquet through the ZOO-4 writer
(``results/activation_rank.parquet``, join key run_id/checkpoint_step/layer,
``regime`` = elicit/teach); the overlay figure lands in ``analysis/figures/``
(gitignored).

Design notes:

- Pooling is the same masked-mean as ``geode.probe.representation_drift``:
  per example, the mean activation vector over the ``label_mask``-True
  positions, in fp64 (dumps are bf16). Padding activations are NOT zero, so
  pooling must go through the mask, never flatten.
- Column-centering: the effective rank asked here is the rank of the *spread*
  of representations across probe examples, not of their shared offset — an
  uncentered matrix is dominated by the mean vector (a rank-1 direction every
  example shares), which would drag every effective rank toward 1. A
  column-centered ``[n, d]`` matrix has rank ≤ ``n − 1`` (the centering
  removes one degree of freedom), so the ceiling on the metric is
  ``min(n − 1, d)``; the ``act_effective_rank_frac`` row normalizes by exactly
  that ceiling, making the two runs' layers comparable when ``n < d``.
- Unlike ``alignment.py`` there is NO ``probe_loss_nats > 0`` filter: pooled
  activations do not degenerate at zero loss (only gradients do — a
  zero-loss example still has a perfectly well-defined residual stream), so
  every snapshot contributes at every hook point. Same rationale as
  ``drift.py``.
- Parquet row i aligns positionally to dump row i, so the row-alignment guard
  (spec 00 §6) refuses unless ``order_hash`` of the frozen probe parquet
  records equals the ``probe_set_hash`` of every dump used, and every
  snapshot's ``input_ids`` must equal the first snapshot's (the probe
  tokenization is fixed across training).

CPU-only, runs wherever the dumps are.

Usage:
    python3 act_rank.py [--run-id <rid> ...] [--store <dir>]
        [--probe-local <path>] [--fig figures/activation_rank.png]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from geode.arith import order_hash
from geode.probe import effective_rank, load_probe_dump, residual_hook_names
from geode.zoo import load_run, write_results
from geode.zoo.store import run_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = ("evt-run7-armA-target-1m", "evt-run8-armB-target-1m")
METRIC = "act_effective_rank"
METRIC_FRAC = "act_effective_rank_frac"
PROBE_REPO = "mhieuuu/elicit-vs-teach-arith"
PROBE_FILE = "probe.parquet"


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


def pooled_matrix(acts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask-pooled, column-centered ``[n_examples, d_model]`` fp64 matrix.

    ``acts`` is ``[n, seq, d]``, ``mask`` the ``[n, seq]`` bool label mask; per
    example the representation is the mean activation over the mask-True
    positions (identical pooling to ``geode.probe.representation_drift``),
    then every column is centered across examples.
    """
    m = mask.to(torch.bool)
    counts = m.sum(dim=1)
    if bool((counts == 0).any()):
        zero_rows = torch.nonzero(counts == 0).flatten().tolist()
        raise ValueError(
            f"pooled_matrix: examples {zero_rows} have zero mask-True positions "
            "— cannot pool a representation"
        )
    mf = m.to(torch.float64)[:, :, None]
    pooled = (acts.to(torch.float64) * mf).sum(dim=1) / counts.to(torch.float64)[:, None]
    return pooled - pooled.mean(dim=0, keepdim=True)


def run_rows(run_id: str, store: Path, probe_hash: str, n_probe: int) -> list[dict]:
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
    manifest = load_run(run_id, store=store)
    regime = manifest.data["regime"]
    dataset_size = manifest.data["dataset"]["n_unique_examples"]

    rows: list[dict] = []
    ref_input_ids: torch.Tensor | None = None
    names: list[str] = []
    for k in steps:
        capture, meta = load_probe_dump(run_id, k, store=store)
        _guard_probe_hash(run_id, k, meta.probe_set_hash, probe_hash)
        if ref_input_ids is None:
            ref_input_ids = capture.input_ids
            names = residual_hook_names(len(capture.acts) - 1)
            if capture.input_ids.shape[0] != n_probe:
                raise ValueError(
                    f"{run_id}@step_{k}: dump has {capture.input_ids.shape[0]} examples but "
                    f"the probe parquet has {n_probe} rows — row i must be the same example"
                )
        elif not torch.equal(capture.input_ids, ref_input_ids):
            raise ValueError(
                f"{run_id}: step {k} input_ids differ from step {steps[0]} — the probe "
                "tokenization must be identical across snapshots to pool one mask"
            )
        for layer, name in enumerate(names):
            centered = pooled_matrix(capture.acts[name], capture.label_mask)
            n, d_model = centered.shape
            ceiling = min(n - 1, d_model)
            er = effective_rank(centered)
            base = {
                "run_id": run_id,
                "base_model_key": meta.base_model_key,
                "regime": regime,
                "dataset_size": dataset_size,
                "checkpoint_step": k,
                "layer": layer,
                "hook_name": name,
                "d_model": d_model,
                "n_examples": n,
            }
            rows.append({**base, "metric_name": METRIC, "metric_value": er})
            rows.append({**base, "metric_name": METRIC_FRAC, "metric_value": er / ceiling})
        del capture  # dumps are ~0.5 GiB: hold only one at a time
    print(f"[evt] {run_id}: {len(steps)} dumps, {len(names)} hook points, {n_probe} examples")
    return rows


def plot(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rank = df[df["metric_name"] == METRIC]

    # Panel 1: effective rank over training — faint per-hook, bold hook-mean.
    ax = axes[0]
    for i, (rid, by_run) in enumerate(rank.groupby("run_id", sort=True)):
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
    ax.set_ylabel("activation effective rank")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)
    ax.set_title("pooled activation effective rank (faint: hooks, bold: hook mean)")

    # Panel 2: final-step effective rank by layer, one series per run.
    ax = axes[1]
    for i, (rid, by_run) in enumerate(rank.groupby("run_id", sort=True)):
        color = colors[i % len(colors)]
        last = by_run["checkpoint_step"].max()
        fin = by_run[by_run["checkpoint_step"] == last].sort_values("layer")
        regime = by_run["regime"].iloc[0]
        ax.plot(
            fin["layer"],
            fin["metric_value"],
            color=color,
            marker="o",
            lw=1.5,
            label=f"{rid} ({regime}) @ {int(last)}",
        )
    ax.set_xlabel("layer (residual hook index)")
    ax.set_xticks(sorted(rank["layer"].unique()))
    ax.set_ylabel("final-step activation effective rank")
    ax.grid(True, alpha=0.2)
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
        "--probe-local", type=Path, default=None, help="skip the HF download of probe.parquet"
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "activation_rank.png",
    )
    args = ap.parse_args()
    run_ids = args.run_ids or list(DEFAULT_RUNS)

    df_probe = _load_probe(args.probe_local)
    probe_hash = order_hash(df_probe.to_dict("records"))
    print(f"[evt] probe set: {len(df_probe)} examples, probe_set_hash={probe_hash[:12]}…")

    rows: list[dict] = []
    for rid in run_ids:
        rows.extend(run_rows(rid, args.store, probe_hash, len(df_probe)))
    df = pd.DataFrame(rows)
    path = write_results(df, "activation_rank", store=args.store)
    print(f"[evt] wrote {path} ({len(df)} rows)")
    plot(df, args.fig)

    rank = df[df["metric_name"] == METRIC]
    for rid, by_run in rank.groupby("run_id", sort=True):
        first, last = int(by_run["checkpoint_step"].min()), int(by_run["checkpoint_step"].max())
        first_mean = by_run[by_run["checkpoint_step"] == first]["metric_value"].mean()
        last_mean = by_run[by_run["checkpoint_step"] == last]["metric_value"].mean()
        ceiling = min(int(by_run["n_examples"].iloc[0]) - 1, int(by_run["d_model"].iloc[0]))
        print(
            f"[evt] {rid}: hook-mean effective rank {first_mean:.2f} @ step {first} → "
            f"{last_mean:.2f} @ step {last} (ceiling {ceiling})"
        )


if __name__ == "__main__":
    main()
