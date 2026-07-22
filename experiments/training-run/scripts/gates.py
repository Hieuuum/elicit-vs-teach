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
on ``"Answer:" + completion``; pass at ≥ 0.95.

**G2** (after run 3) is the identical eval and bar re-run against an installer
checkpoint — spec 02 §8: same protocol as G1, no separate δ, and the shared
``--sample-seed`` means it scores the same 1,024 questions G1 did. **G4**
(runs 3-4) re-scores the installers' in-loop stopping metric (spec 02 §6):
format validity on the same seeded 512-prompt sample of the installer
config's val split, ``geode.arith.format_valid`` on ``"Answer:" +
completion``, pass at the config's threshold. Later gates (G3, G5) slot in
as further subcommands.

CPU-only friendly and no ``--confirm-cost``: evaluation, not training.

Usage:
    python gates.py g1 --run evt-run2-armA-algo --config configs/run2_algo.yaml \
        [--checkpoint <dir>] [--device cuda] [--n 1024] [--sample-seed 316]
    python gates.py g2 --run evt-run3-sweep-lr3e-4 --config configs/run2_algo.yaml
    python gates.py g4 --run evt-run3-sweep-lr3e-4 --config configs/run3_inst.yaml
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from functools import partial
from pathlib import Path

import torch

from geode.arith import (
    exact_match,
    format_valid,
    greedy_completions,
    parse_answer,
    tokenize_with_spans,
)
from geode.train import split_indices
from geode.zoo import load_run
from train import REPO_ROOT, load_config
from train_sft import load_frozen_parquet

# G1 and G2 share the bar: 0.95 is the committed definition of "capability
# present", no separate installer δ (owner 2026-07-21, spec 02 §8).
EXACT_MATCH_THRESHOLD = 0.95


def run_exact_match_gate(args: argparse.Namespace, gate: str) -> int:
    cfg = load_config(args.config, None)
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])
    manifest = load_run(args.run, store=store)

    from transformers import AutoTokenizer, LlamaForCausalLM

    local = (args.config.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])
    checkpoint = args.checkpoint or store / "runs" / args.run / "sft" / "model"
    print(f"[evt] {gate}: loading checkpoint {checkpoint} ...", flush=True)
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
        f"[evt] {gate} accuracy {accuracy:.4f} on n={args.n} (by op: {by_op}) -> "
        f"{'PASS' if passed else 'FAIL'} (threshold {args.threshold})"
    )
    if args.dump:
        # Failure-mode discriminator: a miss whose slot still parses is a
        # clean-but-wrong integer (arithmetic loss); an unparseable slot means
        # the model left the answer format (e.g. slipped into another trained
        # format on these prompts).
        misses = [i for i, h in enumerate(hits) if not h]
        parsed = {i: parse_answer("Answer:" + completions[i]) for i in misses}
        n_malformed = sum(p is None for p in parsed.values())
        print(
            f"[evt] {gate} dump: {len(misses)} misses, {n_malformed} with an "
            f"unparseable answer slot ({len(misses) - n_malformed} clean-but-wrong)"
        )
        for i in misses[: args.dump]:
            prompt_text = tokenizer.decode(prompt_ids[i])
            print(
                f"  {prompt_text!r} -> {completions[i]!r} (want {answers[i]}, parsed {parsed[i]!r})"
            )

    manifest.data.setdefault("experiment", {}).setdefault("gates", {})[gate] = {
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
    print(f"[evt] {gate} verdict recorded in {store / 'runs' / args.run / 'manifest.json'}")
    return 0 if passed else 1


def run_g4(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, None)
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])
    manifest = load_run(args.run, store=store)

    from transformers import AutoTokenizer, LlamaForCausalLM

    local = (args.config.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])
    checkpoint = args.checkpoint or store / "runs" / args.run / "sft" / "model"
    print(f"[evt] G4: loading checkpoint {checkpoint} ...", flush=True)
    model = LlamaForCausalLM.from_pretrained(checkpoint).to(args.device)

    s = cfg["train"]["stopping"]
    if s.get("metric") != "format_validity":
        raise SystemExit(
            "[evt] G4 needs an installer config with a behavioral stopping block "
            "(e.g. run3_inst.yaml) — it re-scores the in-loop metric"
        )
    df = load_frozen_parquet(cfg)
    _, val_idx = split_indices(len(df), cfg["data"]["val_fraction"], cfg["data"]["seed"])
    # Identical prompt set to the in-loop stopping eval (train_sft.py): the
    # seeded sample is over val-split *positions*, so re-deriving it here and
    # indexing into val_idx selects the same questions the trainer scored.
    picks = random.Random(s["prompt_seed"]).sample(range(len(val_idx)), s["n_prompts"])
    rows = df.iloc[[val_idx[i] for i in picks]]
    texts = rows["full_text"].tolist()
    char_spans = list(
        zip(rows["answer_char_start"].astype(int), rows["answer_char_end"].astype(int))
    )
    examples = tokenize_with_spans(texts, char_spans, tokenizer, append_eos=True)
    prompt_ids = [ex.input_ids[: ex.label_span[0]] for ex in examples]

    completions = greedy_completions(
        model, tokenizer, prompt_ids, device=args.device, batch_size=args.batch_size
    )
    rate = sum(format_valid("Answer:" + c) for c in completions) / len(completions)
    passed = rate >= s["threshold"]
    print(
        f"[evt] G4 format_validity {rate:.4f} on n={s['n_prompts']} -> "
        f"{'PASS' if passed else 'FAIL'} (threshold {s['threshold']})"
    )

    manifest.data.setdefault("experiment", {}).setdefault("gates", {})["G4"] = {
        "pass": passed,
        "format_validity": rate,
        "n": s["n_prompts"],
        "threshold": s["threshold"],
        "prompt_seed": s["prompt_seed"],
        "checkpoint": str(checkpoint),
        "protocol": (
            "token-prefix prompts, greedy EOS-stopped, in-loop stopping sample "
            "(seeded over val-split positions), format_valid"
        ),
    }
    manifest.save(store / "runs" / args.run / "manifest.json")
    print(f"[evt] G4 verdict recorded in {store / 'runs' / args.run / 'manifest.json'}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="gate", required=True)

    def common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run", required=True, help="run_id whose manifest records the verdict")
        p.add_argument(
            "--config",
            type=Path,
            required=True,
            help="eval-data YAML: the D_algo run YAML for g1/g2, the installer YAML for g4",
        )
        p.add_argument(
            "--checkpoint", type=Path, default=None, help="default: store/runs/<run>/sft/model"
        )
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--batch-size", type=int, default=128)

    g1 = sub.add_parser("g1", help="Arm A near ceiling on NL add/sub (after run 2)")
    g2 = sub.add_parser("g2", help="arithmetic intact after the installer (run 3) — G1 protocol")
    for gate, p in (("G1", g1), ("G2", g2)):
        common_args(p)
        p.add_argument("--n", type=int, default=1024)
        p.add_argument("--sample-seed", type=int, default=316)
        p.add_argument("--threshold", type=float, default=EXACT_MATCH_THRESHOLD)
        p.add_argument(
            "--dump",
            type=int,
            default=0,
            help="print up to N misses (prompt -> completion) for failure diagnosis",
        )
        p.set_defaults(func=partial(run_exact_match_gate, gate=gate))

    g4 = sub.add_parser("g4", help="format validity, in-loop metric re-scored (runs 3-4)")
    common_args(g4)
    g4.set_defaults(func=run_g4)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
