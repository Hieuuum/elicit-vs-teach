"""Cross-arm representation similarity at matched performance (spec 02 §7, V5.63).

The question: *when the two arms perform equally well, how similar are their
layer-wise representations?* Elicitation and teaching reach the same probe loss
at very different steps, so comparing them at equal step compares different
levels of competence; the performance-aligned step map from ``matching.py``
(V5.16) removes that confound, and linear CKA (``geode.probe.linear_cka``)
compares the two arms' pooled probe representations at each matched pair —
rotation- and scale-invariant, so it is meaningful across two independently
trained models.

Per matched pair ``(step_a, step_b)`` both dumps are loaded through
``extract.load_matched_probe_pair`` (the V5.12 matched-load guard: identical
probe set, tokenizer, and template/format, or refuse), each example's
representation is the fp64 mask-pooled mean activation over its label positions
(the pooling semantics of ``representation_drift``), and CKA is computed per
residual hook point. Results land as one long-format parquet through the ZOO-4
writer (``results/cka_matched.parquet``); the figure lands in
``analysis/figures/`` (gitignored).

Row identity is arm A's (``run_id`` = ``--run-a``, ``checkpoint_step`` =
``step_a``); arm B travels in the ``matched_run_id`` / ``step_b`` extras — the
same convention ``matching.py`` uses for its pair rows.

Prerequisite: ``results/performance_matching.parquet`` (run ``matching.py``
first). CPU-only, runs wherever the dumps are; one pair of dumps (~1 GiB) is
held at a time.

Usage:
    python3 cka.py [--run-a evt-run7-armA-target-1m]
        [--run-b evt-run8-armB-target-1m] [--store <dir>]
        [--fig figures/cka_matched.png]
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

from geode.probe import linear_cka, residual_hook_names
from geode.probe.extract import load_matched_probe_pair
from geode.zoo import load_run, read_results, write_results

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_A = "evt-run7-armA-target-1m"
DEFAULT_RUN_B = "evt-run8-armB-target-1m"
MATCH_TABLE = "performance_matching"
MATCH_METRIC = "matched_step_b"
METRIC = "cka_linear_matched"


def matched_pairs(store: Path, run_a: str, run_b: str) -> list[tuple[int, int]]:
    """The ``(step_a, step_b)`` pairs written by ``matching.py``, in step_a order.

    The pair rows are ``metric_name == "matched_step_b"``: ``checkpoint_step``
    is step_a and ``metric_value`` is the matched step_b (V5.16).
    """
    path = store / "results" / f"{MATCH_TABLE}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found — run analysis/matching.py first: it writes the "
            "performance-aligned step map (V5.16) this driver consumes"
        )
    df = read_results(MATCH_TABLE, store=store)
    sel = df[
        (df["metric_name"] == MATCH_METRIC)
        & (df["run_id"] == run_a)
        & (df["matched_run_id"] == run_b)
    ].sort_values("checkpoint_step")
    if sel.empty:
        raise ValueError(
            f"{path} has no '{MATCH_METRIC}' rows for run_id={run_a!r} matched into "
            f"{run_b!r} — re-run matching.py with --run-a {run_a} --run-b {run_b}"
        )
    return [(int(a), int(b)) for a, b in zip(sel["checkpoint_step"], sel["metric_value"])]


def pooled(acts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-example representation: fp64 mean activation over mask-True positions.

    ``acts`` is ``[n, seq, d]``, ``mask`` the ``[n, seq]`` label mask ⇒ ``[n, d]``.
    Padding activations are NOT zero, so pooling must go through the mask
    (same semantics as ``representation_drift``).
    """
    m = mask.to(torch.bool)
    counts = m.sum(dim=1)
    if bool((counts == 0).any()):
        zero_rows = torch.nonzero(counts == 0).flatten().tolist()
        raise ValueError(
            f"cka: examples {zero_rows} have zero mask-True positions — cannot pool "
            "a representation"
        )
    weighted = acts.to(torch.float64) * m.to(torch.float64)[:, :, None]
    return weighted.sum(dim=1) / counts.to(torch.float64)[:, None]


def pair_rows(
    run_a: str, run_b: str, step_a: int, step_b: int, store: Path, identity: dict
) -> list[dict]:
    """One matched pair ⇒ one row per residual hook point."""
    cap_a, cap_b, meta_a, _ = load_matched_probe_pair(run_a, step_a, run_b, step_b, store=store)
    if not torch.equal(cap_a.label_mask, cap_b.label_mask):
        raise ValueError(
            f"{run_a}@step_{step_a} and {run_b}@step_{step_b} have different label masks "
            "despite matching probe/tokenizer hashes — refusing to pool mismatched positions"
        )
    names = residual_hook_names(len(cap_a.acts) - 1)
    n_examples = int(cap_a.label_mask.shape[0])

    rows: list[dict] = []
    for layer, name in enumerate(names):
        value = linear_cka(
            pooled(cap_a.acts[name], cap_a.label_mask), pooled(cap_b.acts[name], cap_b.label_mask)
        )
        rows.append(
            {
                "run_id": run_a,
                "base_model_key": meta_a.base_model_key,
                "regime": identity["regime"],
                "dataset_size": identity["dataset_size"],
                "checkpoint_step": step_a,
                "layer": layer,
                "metric_name": METRIC,
                "metric_value": value,
                "hook_name": name,
                "matched_run_id": run_b,
                "step_b": step_b,
                "n_examples": n_examples,
            }
        )
    mean = sum(r["metric_value"] for r in rows) / len(rows)
    print(f"[evt] step_a {step_a} -> step_b {step_b}: hook-mean CKA {mean:.4f}")
    del cap_a, cap_b  # dumps are ~0.5 GiB each: hold one pair at a time
    return rows


def plot(df: pd.DataFrame, run_a: str, run_b: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    layers = sorted(df["layer"].unique())
    cmap = plt.get_cmap("viridis")
    for i, layer in enumerate(layers):
        sub = df[df["layer"] == layer].sort_values("checkpoint_step")
        color = cmap(i / max(len(layers) - 1, 1))
        ax.plot(
            sub["checkpoint_step"],
            sub["metric_value"],
            color=color,
            lw=1.2,
            marker="o",
            ms=2.5,
            label=f"layer {int(layer)} ({sub['hook_name'].iloc[0]})",
        )
    mean = df.groupby("checkpoint_step")["metric_value"].mean().sort_index()
    ax.plot(mean.index, mean.values, color="black", lw=2.5, label="hook mean")
    ax.set_xscale("log")
    ax.set_xlabel(f"arm-A checkpoint step ({run_a})")
    ax.set_ylabel("linear CKA vs performance-matched arm-B snapshot")
    ax.set_title(f"cross-arm representation similarity at matched performance\n{run_a} vs {run_b}")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[evt] wrote {out}")


def print_summary(df: pd.DataFrame, pairs: list[tuple[int, int]]) -> None:
    mean = df.groupby("checkpoint_step")["metric_value"].mean()
    step_b = dict(pairs)
    for label, i in (("first", 0), ("middle", len(pairs) // 2), ("final", len(pairs) - 1)):
        step_a = pairs[i][0]
        print(
            f"[evt] {label:>6} pair: step_a {step_a} -> step_b {step_b[step_a]}, "
            f"hook-mean CKA {mean[step_a]:.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-a", default=DEFAULT_RUN_A, help=f"query arm (default: {DEFAULT_RUN_A})")
    ap.add_argument(
        "--run-b", default=DEFAULT_RUN_B, help=f"matched-into arm (default: {DEFAULT_RUN_B})"
    )
    ap.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")),
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "cka_matched.png",
    )
    args = ap.parse_args()

    pairs = matched_pairs(args.store, args.run_a, args.run_b)
    print(f"[evt] {len(pairs)} performance-matched pairs: {args.run_a} (A) -> {args.run_b} (B)")

    manifest = load_run(args.run_a, store=args.store)
    identity = {
        "regime": manifest.data["regime"],
        "dataset_size": manifest.data["dataset"]["n_unique_examples"],
    }

    rows: list[dict] = []
    for step_a, step_b in pairs:
        rows.extend(pair_rows(args.run_a, args.run_b, step_a, step_b, args.store, identity))
    df = pd.DataFrame(rows)
    path = write_results(df, "cka_matched", store=args.store)
    print(f"[evt] wrote {path} ({len(df)} rows)")
    plot(df, args.run_a, args.run_b, args.fig)
    print_summary(df, pairs)


if __name__ == "__main__":
    main()
