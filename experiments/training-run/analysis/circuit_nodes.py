"""Node attribution scores for the arithmetic task — one model, one regime.

Mechanistic phase, metric 2/3 groundwork (owner 2026-08-24). Scores every
attention head (query-head granularity: 32/layer under GQA) and every MLP
block of a Llama-architecture model for its causal contribution to bare-NL
addition/subtraction, via ATTRIBUTION PATCHING (grad x activation-delta, the
first-order approximation of activation patching — Nanda 2023 / Syed et al.
2023): score(node) = sum_pos (a_corrupt - a_clean) . dM/da_clean, where M is
the logit difference metric below. One forward+backward per pair scores all
528 nodes at once.

Protocol:
- Pairs: rows of the frozen eval file's REPORTING block, tokenized and
  bucketed by exact prompt token length; within a bucket, consecutive rows
  with different first answer tokens form (clean, corrupt) pairs — same
  length, same format, different operands/answer.
- Metric M = logit(clean's first answer token) - logit(corrupt's first
  answer token) at the final prompt position (next-token prediction).
- --shots K prepends K fully-rendered exemplars (rows before the query
  range) to BOTH prompts of a pair, joined by blank lines (the G5 16-shot
  convention): base models only perform the task in-context, so their
  circuit exists only in that regime (Prakash et al. 2024's protocol).
  Fine-tuned models run 0-shot.
- SANITY line: mean M over clean runs. A model that cannot do the task in
  the chosen regime has mean M ~ 0 and its scores are NOISE — the compare
  step must not interpret Jaccard against such a map as circuit reuse.

Output: parquet of (layer, node_type, head, score, abs_score) +
a JSON sidecar with the sanity metric and config. GPU, box-only.

Usage:
    python3 circuit_nodes.py --model <dir-or-hub-id> --out <stem> \
        [--shots 0] [--n-pairs 256] [--batch-size 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "training-run" / "scripts"))

from train import load_config  # noqa: E402
from train_sft import load_frozen_parquet  # noqa: E402

from geode.arith import few_shot_prompt  # noqa: E402
from geode.edl import EVAL_STOP_ROWS  # noqa: E402

EVAL_CONFIG = REPO_ROOT / "experiments/training-run/configs/eval_bare_target_data_llama.yaml"


def build_pairs(df, tokenizer, n_pairs: int, shots: int):
    """(clean_ids, corrupt_ids, clean_ans_tok, corrupt_ans_tok) tuples,
    exact-length-matched within each pair."""
    shot_rows = df.iloc[EVAL_STOP_ROWS : EVAL_STOP_ROWS + shots]
    exemplars = shot_rows["full_text"].tolist() if shots else []
    query_rows = df.iloc[EVAL_STOP_ROWS + shots : EVAL_STOP_ROWS + shots + n_pairs * 8]

    buckets: dict[int, list] = defaultdict(list)
    for _, r in query_rows.iterrows():
        prompt = few_shot_prompt(exemplars, r["prompt_text"]) if shots else r["prompt_text"]
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ans_tok = tokenizer(r["answer_text"], add_special_tokens=False)["input_ids"][0]
        buckets[len(ids)].append((ids, ans_tok))

    pairs = []
    for bucket in buckets.values():
        for a, b in zip(bucket[0::2], bucket[1::2]):
            if a[1] != b[1]:  # first answer tokens must differ or M is degenerate
                pairs.append((a[0], b[0], a[1], b[1]))
            if len(pairs) >= n_pairs:
                return pairs
    return pairs


class NodeTaps:
    """Forward hooks exposing per-node residual-stream contributions.

    attn node activation = the o_proj INPUT reshaped (B, T, n_heads, d_head)
    (the concatenated per-query-head outputs, pre-mix — a per-head causal
    handle); mlp node activation = the down_proj OUTPUT (B, T, d_model), the
    block's full write into the residual stream.
    """

    def __init__(self, model):
        self.acts: dict[tuple, torch.Tensor] = {}
        self.handles = []
        cfg = model.config
        self.n_heads = cfg.num_attention_heads
        self.d_head = cfg.hidden_size // cfg.num_attention_heads
        for i, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._attn_hook(i))
            )
            self.handles.append(layer.mlp.down_proj.register_forward_hook(self._mlp_hook(i)))

    def _attn_hook(self, i):
        def hook(_mod, inputs):
            x = inputs[0]  # (B, T, d_model) = concat heads
            x = x.view(*x.shape[:-1], self.n_heads, self.d_head)
            x.retain_grad() if x.requires_grad else None
            self.acts[("attn", i)] = x
            return (x.view(*x.shape[:-2], self.n_heads * self.d_head),)

        return hook

    def _mlp_hook(self, i):
        def hook(_mod, _inputs, output):
            output.retain_grad() if output.requires_grad else None
            self.acts[("mlp", i)] = output
            return output

        return hook

    def clear(self):
        self.acts = {}

    def remove(self):
        for h in self.handles:
            h.remove()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="checkpoint dir or hub id")
    ap.add_argument("--out", required=True, help="output stem: writes <stem>.parquet + <stem>.json")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(EVAL_CONFIG, None)
    df = load_frozen_parquet(cfg)  # hash-verified

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to(args.device).eval()
    # Params stay grad-ENABLED: with everything frozen no activation would
    # require grad and the metric backward would have nothing to reach the
    # taps through. Param grads are zeroed (freed) after every batch; the
    # transient cost is one extra model-size of grad memory.

    pairs = build_pairs(df, tokenizer, args.n_pairs, args.shots)
    if len(pairs) < args.n_pairs // 2:
        print(f"[circuit] WARNING: only {len(pairs)} length-matched pairs found")

    taps = NodeTaps(model)
    n_layers = model.config.num_hidden_layers
    n_heads = taps.n_heads
    scores = {("attn", i, h): 0.0 for i in range(n_layers) for h in range(n_heads)}
    scores.update({("mlp", i, -1): 0.0 for i in range(n_layers)})
    sanity_m = []

    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        clean_ids = torch.tensor([p[0] for p in batch], device=args.device)
        corr_ids = torch.tensor([p[1] for p in batch], device=args.device)
        c_tok = torch.tensor([p[2] for p in batch], device=args.device)
        x_tok = torch.tensor([p[3] for p in batch], device=args.device)

        # corrupt pass (no grad): reference activations
        taps.clear()
        with torch.no_grad():
            model(corr_ids)
        corr_acts = {k: v.detach() for k, v in taps.acts.items()}

        # clean pass (grad): activations + metric backward
        taps.clear()
        with torch.enable_grad():
            logits = model(clean_ids).logits[:, -1].float()
            metric = (
                logits.gather(1, c_tok[:, None]) - logits.gather(1, x_tok[:, None])
            ).sum()
            metric.backward()
        sanity_m.append((metric / len(batch)).item())
        model.zero_grad(set_to_none=True)

        for (kind, i), a_clean in taps.acts.items():
            if a_clean.grad is None:
                continue
            delta = (corr_acts[(kind, i)] - a_clean.detach()).float()
            contrib = (delta * a_clean.grad.float()).sum(dim=(0, 1))  # (heads, d_head) or (d,)
            if kind == "attn":
                per_head = contrib.sum(dim=-1)
                for h in range(n_heads):
                    scores[("attn", i, h)] += per_head[h].item()
            else:
                scores[("mlp", i, -1)] += contrib.sum().item()

    taps.remove()
    rows = [
        {"node_type": k, "layer": i, "head": h, "score": s, "abs_score": abs(s)}
        for (k, i, h), s in scores.items()
    ]
    out = Path(args.out)
    pd.DataFrame(rows).to_parquet(out.with_suffix(".parquet"), index=False)
    sanity = sum(sanity_m) / len(sanity_m)
    meta = {
        "model": args.model,
        "shots": args.shots,
        "n_pairs": len(pairs),
        "mean_logit_diff": sanity,
        "performing_regime": bool(sanity > 1.0),
        "note": "scores from a non-performing model (mean_logit_diff ~ 0) are NOISE; "
        "do not interpret circuit overlap against them as reuse",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[circuit] {args.model} shots={args.shots}: mean logit_diff {sanity:.3f} "
          f"({'PERFORMING' if meta['performing_regime'] else 'NOT PERFORMING — scores are noise'})")
    print(f"[circuit] wrote {out.with_suffix('.parquet')} ({len(rows)} nodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
