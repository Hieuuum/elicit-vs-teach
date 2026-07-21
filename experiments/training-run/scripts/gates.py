"""Verification gates (spec 02 §8) — record verdicts into run manifests.

Currently implements **G1** (after run 2): Arm A near ceiling on NL add/sub.
Protocol (owner 2026-07-20, revised 2026-07-21, spec 02 §8): 1,024 examples
seeded-sampled from D_algo's held-out val split — re-derived with the same
``geode.train.split_indices`` call the launcher used, so the gate provably
never scores trained questions. Prompts are **token-level prefixes of the
training tokenization** (``input_ids[:label_span.start]`` via
``tokenize_with_spans``): re-tokenizing the char-sliced prompt string ends in
a standalone space token the model never saw in training (the byte-level BPE
merges that space into the first answer token) and makes the merged `` -``
sign token unreachable — measured sign-drop on negative answers, 2026-07-21.
Greedy decoding stops at the trained EOS (V5.43); ``geode.arith.exact_match``
on ``"Answer:" + completion``; pass at ≥ 0.95. Later gates (G2-G5) slot in as
further subcommands.

CPU-only friendly and no ``--confirm-cost``: evaluation, not training.

Usage:
    python gates.py g1 --run evt-run2-armA-algo --config configs/run2_algo.yaml \
        [--checkpoint <dir>] [--device cuda] [--n 1024] [--sample-seed 316]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import torch

from geode.arith import exact_match, greedy_completions, tokenize_with_spans
from geode.train import split_indices
from geode.zoo import load_run
from train import REPO_ROOT, load_config
from train_sft import load_frozen_parquet

G1_THRESHOLD = 0.95


def run_g1(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, None)
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])
    manifest = load_run(args.run, store=store)

    from transformers import AutoTokenizer, LlamaForCausalLM

    local = (args.config.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])
    checkpoint = args.checkpoint or store / "runs" / args.run / "sft" / "model"
    print(f"[evt] G1: loading checkpoint {checkpoint} ...", flush=True)
    model = LlamaForCausalLM.from_pretrained(checkpoint).to(args.device)

    df = load_frozen_parquet(cfg)
    _, val_idx = split_indices(len(df), cfg["data"]["val_fraction"], cfg["data"]["seed"])
    sampled = random.Random(args.sample_seed).sample(val_idx, args.n)
    rows = df.iloc[sampled]
    # Prompt = token-level prefix of the training tokenization (everything
    # before the label span), so the decode context is exactly a training
    # context — see module docstring for why char-sliced prompts are not.
    texts = rows["full_text"].tolist()
    char_spans = list(
        zip(rows["answer_char_start"].astype(int), rows["answer_char_end"].astype(int))
    )
    examples = tokenize_with_spans(texts, char_spans, tokenizer)
    prompt_ids = [ex.input_ids[: ex.label_span[0]] for ex in examples]
    answers = rows["true_answer"].astype(int).tolist()
    ops = rows["op"].tolist()

    completions = greedy_completions(
        model, tokenizer, prompt_ids, device=args.device, batch_size=args.batch_size
    )
    # The completion starts inside the answer slot (its leading space rides on
    # the first answer token), so "Answer:" + completion re-enters the tested
    # parser with the marker it expects.
    hits = [exact_match("Answer:" + c, a) for c, a in zip(completions, answers)]
    accuracy = sum(hits) / len(hits)
    by_op = {
        op: sum(h for h, o in zip(hits, ops) if o == op) / max(1, ops.count(op))
        for op in sorted(set(ops))
    }
    passed = accuracy >= args.threshold
    print(
        f"[evt] G1 accuracy {accuracy:.4f} on n={args.n} (by op: {by_op}) -> "
        f"{'PASS' if passed else 'FAIL'} (threshold {args.threshold})"
    )

    manifest.data.setdefault("experiment", {}).setdefault("gates", {})["G1"] = {
        "pass": passed,
        "accuracy": accuracy,
        "accuracy_by_op": by_op,
        "n": args.n,
        "threshold": args.threshold,
        "sample_seed": args.sample_seed,
        "checkpoint": str(checkpoint),
        "protocol": (
            "token-prefix prompts, greedy EOS-stopped first-line, "
            "seeded sample of D_algo val split, exact_match"
        ),
    }
    manifest.save(store / "runs" / args.run / "manifest.json")
    print(f"[evt] G1 verdict recorded in {store / 'runs' / args.run / 'manifest.json'}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="gate", required=True)
    g1 = sub.add_parser("g1", help="Arm A near ceiling on NL add/sub (after run 2)")
    g1.add_argument("--run", required=True, help="run_id whose manifest records the verdict")
    g1.add_argument("--config", type=Path, required=True, help="the run's YAML (data + split)")
    g1.add_argument(
        "--checkpoint", type=Path, default=None, help="default: store/runs/<run>/sft/model"
    )
    g1.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g1.add_argument("--n", type=int, default=1024)
    g1.add_argument("--sample-seed", type=int, default=316)
    g1.add_argument("--batch-size", type=int, default=128)
    g1.add_argument("--threshold", type=float, default=G1_THRESHOLD)
    g1.set_defaults(func=run_g1)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
