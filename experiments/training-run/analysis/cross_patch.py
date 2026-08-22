"""Cross-model residual patching θ_T → θ0 (Phase-0/Tier-2 test 4): does
plugging θ_T's own residual stream into θ0's forward pass recover θ_T's
answer, and how much of the sequence needs to carry it?

For every ``mech_lib`` residual layer ℓ (0 = embed, i+1 = block i), every
patch scope (``answer`` = only the position right before the first answer
token, ``all`` = every position — see ``cross_patch_rows``'s docstring for
why ``patch_residual``'s own ``positions=None`` already means "every
non-pad position" here), and both directions (``T_into_0``: θ_T's residual
patched into θ0's forward pass; ``0_into_T``: θ0's residual patched into
θ_T's forward pass, the DAMAGE direction — does replacing θ_T's own
representation with θ0's weaker one destroy its answer?): run θ0 (``--model-
a``) and θ_T (``--model-b``) clean, capture θ_T's/θ0's residual at ℓ, patch
the OTHER model with it, and measure the answer log-prob (nats) and top-1
accuracy at the first answer token (``mech_nodes``'s metric convention,
identical to ``logit_lens.py``'s ``first_answer``: position
``label_span[0]-1``, target ``input_ids[label_span[0]]``).

``recovery_frac = (metric_patched - metric_clean_a) / (metric_clean_b -
metric_clean_a)`` — ONE formula for both directions, always anchored to
θ0's/θ_T's own clean baselines regardless of which model got patched: 0.0
means "post-patch behavior matches θ0's own clean run", 1.0 means "matches
θ_T's own clean run". For ``T_into_0`` this reads as literal recovery (0 =
no gain, 1 = fully recovers θ_T's answer); for ``0_into_T`` it reads as the
INVERSE of damage (1 = undamaged, 0 = fully collapsed to θ0's own
performance) — the module docstring's "damage direction" label refers to
what the patch DOES (weakens θ_T), not to the sign of this column. NaN
when ``|metric_clean_b - metric_clean_a| < _DENOM_EPS`` (division by
~zero — the identical-models case).

Not pre-registered (decisions.md 2026-08-21 night, Tier 2: "no owner
signature pre-registered for either arm"). Candidate readout, stated as a
hypothesis to check, not a result: **elicit** (θ0 = ``evt-ts38pp-parent``)
— a single mid/late layer's residual from θ_T recovers most of the gain in
θ0 (the downstream machinery θ0 already has computes the answer once given
the right representation), recovery concentrated at one or two layers, and
``answer``-scope alone gets most of the way there. **teach** (θ0 = ``evt-
ts38mt-fmt-parent``) — recovery only appears at the LAST layers, or only
under ``all``-scope patching (the representation itself, not just its
value at the answer position, needs replacing).

Usage:
    python3 cross_patch.py --model-a dir:$GEODE_STORE/runs/evt-ts38pp-parent/model \\
        --model-b run:evt-ts38mt-pp-n21544 \\
        --prompt-parquet ../data/full/D_algo_eval_bare.parquet \\
        --out cross_patch_ts38mt_pp_n21544.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import Tensor, nn

from mech_lib import (
    capture_residuals,
    load_any_model,
    load_task_examples,
    pad_examples,
    write_table,
)
from mech_nodes import answer_targets, patch_residual

from geode.arith.spans import SftExample

DEFAULT_TOKENIZER = Path(__file__).resolve().parent.parent / "tokenizer"
_DENOM_EPS = 1e-6


def _logprob_and_top1(logits_at_pos: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """``(logprob[b] fp64 nats, top1_correct[b] bool)`` at one set of
    per-example logit rows — pure-tensor metric, the same fp64
    ``log_softmax`` convention as ``logit_lens.lens_metrics``."""
    logp = torch.log_softmax(logits_at_pos.to(torch.float64), dim=-1)
    target_logp = logp.gather(1, targets[:, None]).squeeze(1)
    top1 = logits_at_pos.argmax(dim=-1) == targets
    return target_logp, top1


def _run_metrics(
    model: nn.Module, input_ids: Tensor, attention_mask: Tensor, positions: Tensor, targets: Tensor
) -> tuple[Tensor, Tensor]:
    """One forward pass (no grad) + ``_logprob_and_top1`` at ``positions``."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask.long()).logits
    b = input_ids.shape[0]
    idx = torch.arange(b, device=logits.device)
    sel = logits[idx, positions.to(logits.device), :]
    return _logprob_and_top1(sel, targets.to(logits.device))


def recovery_frac(metric_patched: float, metric_clean_a: float, metric_clean_b: float) -> float:
    """``(patched - clean_a) / (clean_b - clean_a)``; ``nan`` when the
    denominator is smaller than ``_DENOM_EPS`` in magnitude (θ0/θ_T give
    indistinguishable clean metrics — most commonly, identical models)."""
    denom = metric_clean_b - metric_clean_a
    if abs(denom) < _DENOM_EPS:
        return math.nan
    return (metric_patched - metric_clean_a) / denom


def cross_patch_rows(
    model_a: nn.Module, model_b: nn.Module, examples: Sequence[SftExample]
) -> list[dict]:
    """One row per (layer, scope, direction) — 2 * 2 * (n_layers+1) rows.

    ``scope="all"`` patches with ``patch_residual``'s own ``positions=None``
    (literally every position, including any padding) rather than an
    explicit non-pad position set: ``mech_lib.pad_examples`` right-pads and
    every model here uses causal + padding-masked attention, so a padded
    position's residual value can never reach any real position's
    computation (it is masked out of every attention sum a real query
    position could read) — "patch everywhere" and "patch every non-pad
    position" are functionally identical for the answer-position metric
    this function measures, without needing a variable-length position set
    ``patch_residual``'s per-example-single-index contract doesn't support.
    """
    if not examples:
        raise ValueError("cross_patch_rows: empty examples")
    device = next(model_a.parameters()).device
    input_ids, attention_mask = pad_examples(examples, device=str(device))
    answer_pos, targets = answer_targets(examples)
    answer_pos, targets = answer_pos.to(device), targets.to(device)

    clean_logp_a, clean_top1_a = _run_metrics(
        model_a, input_ids, attention_mask, answer_pos, targets
    )
    clean_logp_b, clean_top1_b = _run_metrics(
        model_b, input_ids, attention_mask, answer_pos, targets
    )
    clean_metric_a = float(clean_logp_a.mean())
    clean_metric_b = float(clean_logp_b.mean())
    clean_top1_a = float(clean_top1_a.to(torch.float64).mean())
    clean_top1_b = float(clean_top1_b.to(torch.float64).mean())

    resid_a = capture_residuals(model_a, input_ids, attention_mask)
    resid_b = capture_residuals(model_b, input_ids, attention_mask)
    names = list(resid_a.keys())
    n_layers = len(names) - 1

    directions = (
        ("T_into_0", model_a, resid_b),
        ("0_into_T", model_b, resid_a),
    )

    rows: list[dict] = []
    for layer in range(n_layers + 1):
        name = names[layer]
        for scope in ("answer", "all"):
            patch_pos = answer_pos if scope == "answer" else None
            for direction, model_patch, source_resid in directions:
                with patch_residual(model_patch, layer, source_resid[name], positions=patch_pos):
                    patched_logp, patched_top1 = _run_metrics(
                        model_patch, input_ids, attention_mask, answer_pos, targets
                    )
                metric_patched = float(patched_logp.mean())
                top1_patched = float(patched_top1.to(torch.float64).mean())
                rows.append(
                    {
                        "layer": layer,
                        "scope": scope,
                        "direction": direction,
                        "metric_clean_a": clean_metric_a,
                        "metric_clean_b": clean_metric_b,
                        "metric_patched": metric_patched,
                        "recovery_frac": recovery_frac(
                            metric_patched, clean_metric_a, clean_metric_b
                        ),
                        "top1_clean_a": clean_top1_a,
                        "top1_clean_b": clean_top1_b,
                        "top1_patched": top1_patched,
                        "n": len(examples),
                    }
                )
    return rows


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    for direction, sub in df.groupby("direction"):
        print(f"[evt] direction={direction}")
        for scope, sub2 in sub.groupby("scope"):
            sub2 = sub2.sort_values("layer")
            best = (
                sub2.loc[sub2["recovery_frac"].idxmax()]
                if sub2["recovery_frac"].notna().any()
                else None
            )
            print(f"[evt]   scope={scope}")
            print(
                sub2[["layer", "recovery_frac", "metric_patched", "top1_patched"]].to_string(
                    index=False
                )
            )
            if best is not None:
                print(
                    f"[evt]   best recovery at layer {int(best['layer'])}: "
                    f"{best['recovery_frac']:.4f}"
                )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-a", required=True, help="theta0: run:<run_id> or dir:<path>")
    ap.add_argument("--model-b", required=True, help="theta_T: run:<run_id> or dir:<path>")
    ap.add_argument("--prompt-parquet", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--store", type=Path, default=None, help="override $GEODE_STORE for run: specs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N prompt rows")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "cross_patch.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model_a = load_any_model(args.model_a, device=args.device, store=args.store)
    model_b = load_any_model(args.model_b, device=args.device, store=args.store)
    df = pd.read_parquet(args.prompt_parquet)
    examples = load_task_examples(df, tokenizer, limit=args.limit)

    rows = cross_patch_rows(model_a, model_b, examples)
    write_table(pd.DataFrame(rows), args.out)
    print(f"[evt] wrote {args.out} ({len(rows)} rows)")
    print_summary(rows)


if __name__ == "__main__":
    main()
