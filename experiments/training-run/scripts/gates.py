"""Verification gates (spec 02 §8) — record verdicts into run manifests.

Currently implements **G1** (after run 2): Arm A near ceiling on NL add/sub.
Protocol (owner decision 2026-07-20, spec 02 §8): 1,024 examples seeded-sampled
from D_algo's held-out val split — re-derived with the same
``geode.train.split_indices`` call the launcher used, so the gate provably
never scores trained questions — greedy decoding, ``geode.arith.exact_match``,
pass at ≥ 0.95. Later gates (G2-G5) slot in as further subcommands.

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

from geode.arith import exact_match
from geode.train import split_indices
from geode.zoo import load_run
from train import REPO_ROOT, load_config
from train_sft import load_frozen_parquet

G1_THRESHOLD = 0.95


@torch.no_grad()
def greedy_completions(
    model, tokenizer, prompts: list[str], *, device: str, batch_size: int, max_new_tokens: int = 8
) -> list[str]:
    """First generated line for each prompt (greedy, left-padded batches)."""
    model.eval()
    tokenizer.padding_side = "left"
    out: list[str] = []
    for start in range(0, len(prompts), batch_size):
        enc = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        generated = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = generated[:, enc["input_ids"].shape[1] :]
        for row in tokenizer.batch_decode(new_tokens, skip_special_tokens=True):
            out.append(row.split("\n")[0])
    return out


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
    # Prompt = everything before the answer chars (ends "Answer: "), the G5
    # few_shot_prompt empty-slot convention. exact_match parses the final
    # Answer: slot of prompt + first generated line via the tested parser.
    prompts = [t[:cs] for t, cs in zip(rows["full_text"], rows["answer_char_start"].astype(int))]
    answers = rows["true_answer"].astype(int).tolist()
    ops = rows["op"].tolist()

    completions = greedy_completions(
        model, tokenizer, prompts, device=args.device, batch_size=args.batch_size
    )
    hits = [exact_match(p + c, a) for p, c, a in zip(prompts, completions, answers)]
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
        "protocol": "greedy first-line, seeded sample of D_algo val split, exact_match",
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
