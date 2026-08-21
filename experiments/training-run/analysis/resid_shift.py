"""Task-specific vs. generic residual shift (Phase 0 test 10): did training
move the residual stream mostly where the task lives, or everywhere?

Given θ0 (``--model-a``), θ_T (``--model-b``) and two prompt sets
(``--task-parquet``, ``--generic-local``), per layer (``mech_lib``'s residual
convention: 0 = embed, i+1 = ``blocks.{i}``):

- ``task``: at the position right before the first answer token (``p-1``).
- ``generic``: mean-pooled over every non-pad position per example.

``rel_shift`` = mean over examples of ‖h_T − h_0‖₂ / ‖h_0‖₂. Direction
consistency of the per-example shift vectors ``d_i = h_T,i − h_0,i``:
``mean_cos_to_mean`` = mean_i cos(d_i, mean_j d_j) and ``top_pc_evr`` =
explained-variance ratio of the first PC of the UNCENTERED {d_i} — same
deliberate choice as ``geode.probe.metrics.gradient_alignment``'s
``top_pc_explained_variance``, and for the same reason: a perfectly
consistent shift (every ``d_i`` identical, e.g. a constant additive bias
injected at one layer) is EXACTLY the shared-direction structure this metric
exists to detect, and mean-centering would subtract it out to a zero matrix
before the SVD ever ran, forcing ``top_pc_evr`` to 0 in the one case it
should read 1. Both metrics are defined as 0 rather than NaN when every
``d_i`` is exactly zero (identical models): norms are clamped away from 0
before dividing, which also makes near-zero-shift rows quietly stable
instead of numerically explosive.

Usage:
    python3 resid_shift.py --model-a dir:$GEODE_STORE/runs/evt-run1-base-v3-ext/model \\
        --model-b dir:$GEODE_STORE/runs/evt-ts38pp-parent/model \\
        --task-parquet ../data/full/D_algo_eval_bare.parquet \\
        --generic-local /path/to/tinystories.txt --out resid_shift_ts38pp.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import nn

from mech_lib import (
    capture_residuals,
    load_any_model,
    load_generic_texts,
    load_task_examples,
    pad_examples,
    write_table,
)

from geode.arith.spans import SftExample

DEFAULT_TOKENIZER = Path(__file__).resolve().parent.parent / "tokenizer"
_EPS = 1e-30


def rel_shift_and_deltas(h0: torch.Tensor, ht: torch.Tensor) -> tuple[float, torch.Tensor]:
    """``(rel_shift, d)``: ``d = h_T - h_0`` (fp64, ``[n, d_model]``) and the
    mean over examples of ``||d_i|| / ||h0_i||`` (denominator norm-clamped)."""
    d = (ht - h0).to(torch.float64)
    norm_h0 = h0.to(torch.float64).norm(dim=1).clamp_min(_EPS)
    ratio = d.norm(dim=1) / norm_h0
    return float(ratio.mean().item()), d


def shift_consistency(d: torch.Tensor) -> tuple[float, float]:
    """``(mean_cos_to_mean, top_pc_evr)`` for a ``[n, d_model]`` shift matrix.

    ``top_pc_evr`` is the UNCENTERED spectrum (module docstring). Both norm
    and the spectrum total are clamp_min'd rather than branched on zero, so
    an all-zero (or all-cancelling) ``d`` yields exactly 0.0 for both rather
    than NaN.
    """
    d = d.to(torch.float64)
    n = d.shape[0]
    mean_vec = d.mean(dim=0)
    norms = d.norm(dim=1).clamp_min(_EPS)
    mean_norm = mean_vec.norm().clamp_min(_EPS)
    cos = (d @ mean_vec) / (norms * mean_norm)
    mean_cos = float(cos.mean().item())

    if n < 2:
        return mean_cos, 0.0
    s = torch.linalg.svdvals(d)
    total = (s**2).sum().clamp_min(_EPS)
    evr = float((s[0] ** 2 / total).item())
    return mean_cos, evr


def _task_vectors(
    acts: dict[str, torch.Tensor], name: str, ex_idx: torch.Tensor, pos: torch.Tensor
) -> torch.Tensor:
    return acts[name][ex_idx, pos, :]


def _generic_vectors(
    acts: dict[str, torch.Tensor], name: str, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pooled per example over every non-pad position."""
    m = attention_mask.to(torch.float64)[:, :, None]
    h = acts[name].to(torch.float64)
    counts = m.sum(dim=1).clamp_min(_EPS)
    return (h * m).sum(dim=1) / counts


def resid_shift_rows(
    model_a: nn.Module,
    model_b: nn.Module,
    task_examples: Sequence[SftExample],
    generic_texts: Sequence[str],
    tokenizer,
) -> list[dict]:
    """One row per (layer, set) — see the module docstring for the metrics."""
    if not task_examples:
        raise ValueError("resid_shift_rows: empty task example list")
    if not generic_texts:
        raise ValueError("resid_shift_rows: empty generic text list")
    device = next(model_a.parameters()).device
    rows: list[dict] = []

    input_ids, attention_mask = pad_examples(task_examples, device=str(device))
    acts_a = capture_residuals(model_a, input_ids, attention_mask)
    acts_b = capture_residuals(model_b, input_ids, attention_mask)
    names = list(acts_a.keys())
    ex_idx = torch.arange(len(task_examples), device=device)
    pos = torch.tensor([ex.label_span[0] - 1 for ex in task_examples], device=device)
    for layer, name in enumerate(names):
        h0 = _task_vectors(acts_a, name, ex_idx, pos)
        ht = _task_vectors(acts_b, name, ex_idx, pos)
        rs, d = rel_shift_and_deltas(h0, ht)
        mean_cos, evr = shift_consistency(d)
        rows.append(
            {
                "layer": layer,
                "set": "task",
                "rel_shift": rs,
                "mean_cos_to_mean": mean_cos,
                "top_pc_evr": evr,
                "n": len(task_examples),
            }
        )

    enc = tokenizer(
        list(generic_texts), add_special_tokens=False, padding=True, return_tensors="pt"
    )
    g_ids = enc["input_ids"].to(device)
    g_mask = enc["attention_mask"].to(device).bool()
    acts_a_g = capture_residuals(model_a, g_ids, g_mask)
    acts_b_g = capture_residuals(model_b, g_ids, g_mask)
    for layer, name in enumerate(names):
        h0 = _generic_vectors(acts_a_g, name, g_mask)
        ht = _generic_vectors(acts_b_g, name, g_mask)
        rs, d = rel_shift_and_deltas(h0, ht)
        mean_cos, evr = shift_consistency(d)
        rows.append(
            {
                "layer": layer,
                "set": "generic",
                "rel_shift": rs,
                "mean_cos_to_mean": mean_cos,
                "top_pc_evr": evr,
                "n": len(generic_texts),
            }
        )
    return rows


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    task = df[df["set"] == "task"].sort_values("layer")
    generic = df[df["set"] == "generic"].sort_values("layer").set_index("layer")["rel_shift"]
    print("[evt] layer  task_rel_shift  generic_rel_shift  ratio")
    for r in task.itertuples():
        g = float(generic.get(r.layer, float("nan")))
        ratio = r.rel_shift / g if g > 0 else float("nan")
        print(f"[evt] {r.layer:5d}  {r.rel_shift:14.6f}  {g:18.6f}  {ratio:6.3f}")
    peak = task.loc[task["rel_shift"].idxmax()]
    print(f"[evt] max task rel_shift at layer {int(peak['layer'])} ({peak['rel_shift']:.6f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-a", required=True, help="theta0: run:<run_id> or dir:<path>")
    ap.add_argument("--model-b", required=True, help="theta_T: run:<run_id> or dir:<path>")
    ap.add_argument("--task-parquet", type=Path, required=True)
    ap.add_argument("--generic-local", type=Path, required=True, help=".txt or .parquet")
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--store", type=Path, default=None, help="override $GEODE_STORE for run: specs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N examples per set")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "resid_shift.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model_a = load_any_model(args.model_a, device=args.device, store=args.store)
    model_b = load_any_model(args.model_b, device=args.device, store=args.store)

    df = pd.read_parquet(args.task_parquet)
    task_examples = load_task_examples(df, tokenizer, limit=args.limit)
    generic_texts = load_generic_texts(args.generic_local, limit=args.limit)

    rows = resid_shift_rows(model_a, model_b, task_examples, generic_texts, tokenizer)
    write_table(pd.DataFrame(rows), args.out)
    print(f"[evt] wrote {args.out} ({len(rows)} rows)")
    print_summary(rows)


if __name__ == "__main__":
    main()
