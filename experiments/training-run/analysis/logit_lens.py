"""Logit-lens depth (Phase 0 test 6): how early is the answer readable?

For one model and one prompt parquet, projects every residual-stream point
(embed + each block's ``hook_resid_post``, ``mech_lib``'s layer convention:
0 = embed, i+1 = ``blocks.{i}``) through the model's own final norm and
``lm_head``, at the position right before the first answer token (``p-1``).
Two ``position_kind`` row-sets per (model, set, layer):

- ``first_answer``: only the first label token, one (example, position) pair
  per example.
- ``all_label``: every label position, teacher-forced (one pair per label
  token in every example's answer span).

Usage:
    python3 logit_lens.py --model run:evt-ts38pp-parent \\
        --prompt-parquet ../data/full/D_algo_eval_bare.parquet --set-name task \\
        --out logit_lens_ts38pp_task.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import nn

from mech_lib import (
    capture_residuals,
    final_norm_and_head,
    load_any_model,
    load_task_examples,
    pad_examples,
    write_table,
)

from geode.arith.spans import SftExample

DEFAULT_TOKENIZER = Path(__file__).resolve().parent.parent / "tokenizer"
FIRST_ANSWER = "first_answer"
ALL_LABEL = "all_label"


def position_target_pairs(
    examples: Sequence[SftExample], kind: str
) -> tuple[list[int], list[int], list[int]]:
    """(example_idx, position, target_token) triples for one ``position_kind``.

    ``position`` is always ``p - 1`` for the label token at index ``p`` (a
    causal LM predicts token ``p`` from position ``p - 1``); ``first_answer``
    keeps only ``p = label_span[0]``, ``all_label`` ranges over every ``p``
    in ``[label_span[0], label_span[1])``.
    """
    ex_idx: list[int] = []
    pos: list[int] = []
    target: list[int] = []
    for i, ex in enumerate(examples):
        start, end = ex.label_span
        token_range = range(start, start + 1) if kind == FIRST_ANSWER else range(start, end)
        for t in token_range:
            ex_idx.append(i)
            pos.append(t - 1)
            target.append(ex.input_ids[t])
    return ex_idx, pos, target


def lens_metrics(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    """``(top1_acc, mean_logprob_nats, mean_rank)`` for one ``[n, vocab]`` batch.

    ``rank`` is the count of tokens with strictly greater logit than the
    target's own (0 = the target is the top-1 prediction).
    """
    logits64 = logits.to(torch.float64)
    logp = torch.log_softmax(logits64, dim=-1)
    target_logp = logp.gather(1, target[:, None]).squeeze(1)
    target_logit = logits64.gather(1, target[:, None]).squeeze(1)
    top1 = logits64.argmax(dim=-1) == target
    rank = (logits64 > target_logit[:, None]).sum(dim=-1)
    return (
        float(top1.double().mean().item()),
        float(target_logp.mean().item()),
        float(rank.double().mean().item()),
    )


def logit_lens_rows(
    model: nn.Module, examples: Sequence[SftExample], model_label: str, set_label: str
) -> list[dict]:
    """One row per (layer, position_kind): the lens metrics at that depth."""
    if not examples:
        raise ValueError("logit_lens_rows: empty example list")
    device = next(model.parameters()).device
    input_ids, attention_mask = pad_examples(examples, device=str(device))
    acts = capture_residuals(model, input_ids, attention_mask)
    names = list(acts.keys())
    norm, lm_head = final_norm_and_head(model)

    rows: list[dict] = []
    with torch.no_grad():
        for kind in (FIRST_ANSWER, ALL_LABEL):
            ex_idx, pos, target = position_target_pairs(examples, kind)
            ex_idx_t = torch.tensor(ex_idx, device=device)
            pos_t = torch.tensor(pos, device=device)
            target_t = torch.tensor(target, device=device)
            for layer, name in enumerate(names):
                h = acts[name][ex_idx_t, pos_t, :]
                logits = lm_head(norm(h))
                top1_acc, mean_logprob, mean_rank = lens_metrics(logits, target_t)
                rows.append(
                    {
                        "model": model_label,
                        "set": set_label,
                        "layer": layer,
                        "position_kind": kind,
                        "top1_acc": top1_acc,
                        "mean_logprob_nats": mean_logprob,
                        "mean_rank": mean_rank,
                        "n": len(ex_idx),
                    }
                )
    return rows


def emergence_layer(rows: list[dict]) -> float:
    """Smallest ``first_answer`` layer whose ``top1_acc`` is at least half the
    final layer's; ``nan`` if the final layer's ``top1_acc`` is 0."""
    fa = sorted((r for r in rows if r["position_kind"] == FIRST_ANSWER), key=lambda r: r["layer"])
    if not fa:
        return math.nan
    final = fa[-1]["top1_acc"]
    if final <= 0.0:
        return math.nan
    for r in fa:
        if r["top1_acc"] >= 0.5 * final:
            return float(r["layer"])
    return math.nan


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    for (model_label, set_label), sub in df.groupby(["model", "set"]):
        layer = emergence_layer(sub.to_dict("records"))
        fa = sub[sub["position_kind"] == FIRST_ANSWER].sort_values("layer")
        print(f"[evt] {model_label} / {set_label}: emergence layer = {layer}")
        print(fa[["layer", "top1_acc", "mean_logprob_nats", "mean_rank"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="run:<run_id> or dir:<path>")
    ap.add_argument("--model-name", default=None, help="'model' column label (default: --model)")
    ap.add_argument("--prompt-parquet", type=Path, required=True)
    ap.add_argument("--set-name", default=None, help="'set' column label (default: parquet stem)")
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--store", type=Path, default=None, help="override $GEODE_STORE for run: specs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N prompt rows")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "logit_lens.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_any_model(args.model, device=args.device, store=args.store)
    df = pd.read_parquet(args.prompt_parquet)
    examples = load_task_examples(df, tokenizer, limit=args.limit)

    model_label = args.model_name or args.model
    set_label = args.set_name or args.prompt_parquet.stem
    rows = logit_lens_rows(model, examples, model_label, set_label)
    write_table(pd.DataFrame(rows), args.out)
    print(f"[evt] wrote {args.out} ({len(rows)} rows)")
    print_summary(rows)


if __name__ == "__main__":
    main()
