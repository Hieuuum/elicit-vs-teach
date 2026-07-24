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
``--sample-seed`` means it scores the same 1,024 questions G1 did. **G3**
(after run 4) is the same eval against the Arm B installer with the bar
inverted — pass iff accuracy <= threshold (default 0.02, chance + margin):
the random-label installer must not have taught real add/sub. **G4**
(runs 3-4) re-scores the installers' in-loop stopping metric (spec 02 §6):
format validity on the same seeded 512-prompt sample of the installer
config's val split, ``geode.arith.format_valid`` on ``"Answer:" +
completion``, pass at the config's threshold. **G5** (runs 3-6) measures
zero- and 16-shot exact match on operator add/sub — recorded evidence, no
pass bar: spec 02 §8 states expectations (A ~2%/12%, B 0%/0%), not
thresholds. Its data is the frozen shared eval file (``D_target_eval``,
owner 2026-07-22): question-disjoint from D_target ∪ D_algo ∪ probe by
construction, so nothing G5 scores was ever trained on, and the questions
are FIXED slices of the file — the identical set for every run, no
sampling. It also records the shared-set test loss (masked NLL over the
eval file's reporting block), putting every run's loss on identical data.
Few-shot prompts are composed as exemplars + the *complete* query (true
answer filled in), then token-prefix sliced from a training-style
tokenization of the composed text — never a re-tokenized char slice (same
trailing-space incident as above).

CPU-only friendly and no ``--confirm-cost``: evaluation, not training.

Usage:
    python gates.py g1 --run evt-run2-armA-algo --config configs/run2_algo.yaml \
        [--checkpoint <dir>] [--device cuda] [--n 1024] [--sample-seed 316]
    python gates.py g2 --run evt-run3-sweep-lr3e-4 --config configs/run2_algo.yaml
    python gates.py g3 --run evt-run4-armB-inst --config configs/run2_algo.yaml
    python gates.py g4 --run evt-run3-sweep-lr3e-4 --config configs/run3_inst.yaml
    python gates.py g5 --run evt-run3-armA-inst --config configs/eval_target_data.yaml
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
    few_shot_prompt,
    format_valid,
    greedy_completions,
    parse_answer,
    tokenize_with_spans,
)
from geode.edl.masking import TaskFormat
from geode.train import evaluate_sft_nll_nats, split_indices
from geode.zoo import checkpoint_dir, load_model, load_run
from train import REPO_ROOT, load_config
from train_sft import load_frozen_parquet
from train_target import EVAL_STOP_ROWS

# G1 and G2 share the bar: 0.95 is the committed definition of "capability
# present", no separate installer δ (owner 2026-07-21, spec 02 §8).
EXACT_MATCH_THRESHOLD = 0.95
# G3 inverts the bar (spec 02 §8): Arm B's random-label installer must NOT
# have taught real add/sub — pass iff accuracy <= chance + margin. Exact match
# on signed multi-digit integers has ~0 chance rate; 0.02 is the margin.
G3_LEAK_THRESHOLD = 0.02
# G5 is zero/16-shot (spec 02 §8): the shot count is protocol, not a knob.
G5_N_SHOTS = 16


def run_exact_match_gate(args: argparse.Namespace, gate: str, invert: bool = False) -> int:
    cfg = load_config(args.config, None)
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])
    manifest = load_run(args.run, store=store)

    from transformers import AutoTokenizer

    local = (args.config.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    checkpoint = args.checkpoint or checkpoint_dir(args.run, store=store)
    print(f"[evt] {gate}: loading checkpoint {checkpoint} ...", flush=True)
    model = load_model(args.run, store=store, device=args.device, checkpoint=checkpoint)

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
    passed = accuracy <= args.threshold if invert else accuracy >= args.threshold
    print(
        f"[evt] {gate} accuracy {accuracy:.4f} on n={args.n} (by op: {by_op}) -> "
        f"{'PASS' if passed else 'FAIL'} (threshold {'<=' if invert else '>='} {args.threshold})"
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

    protocol = (
        "token-prefix prompts, greedy EOS-stopped first-line, "
        "seeded sample of D_algo val split, exact_match"
    )
    if invert:
        protocol += "; pass = accuracy <= threshold: random labels didn't leak"
    manifest.data.setdefault("experiment", {}).setdefault("gates", {})[gate] = {
        "pass": passed,
        "accuracy": accuracy,
        "accuracy_by_op": by_op,
        "n": args.n,
        "threshold": args.threshold,
        "sample_seed": args.sample_seed,
        "checkpoint": str(checkpoint),
        "protocol": protocol,
    }
    manifest.save(store / "runs" / args.run / "manifest.json")
    print(f"[evt] {gate} verdict recorded in {store / 'runs' / args.run / 'manifest.json'}")
    return 0 if passed else 1


def run_g4(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, None)
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])
    manifest = load_run(args.run, store=store)

    from transformers import AutoTokenizer

    local = (args.config.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    checkpoint = args.checkpoint or checkpoint_dir(args.run, store=store)
    print(f"[evt] G4: loading checkpoint {checkpoint} ...", flush=True)
    model = load_model(args.run, store=store, device=args.device, checkpoint=checkpoint)

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


def run_g5(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, None)
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])
    manifest = load_run(args.run, store=store)

    from transformers import AutoTokenizer

    local = (args.config.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    checkpoint = args.checkpoint or checkpoint_dir(args.run, store=store)
    print(f"[evt] G5: loading checkpoint {checkpoint} ...", flush=True)
    model = load_model(args.run, store=store, device=args.device, checkpoint=checkpoint)

    df = load_frozen_parquet(cfg)  # D_target_eval, order_hash verified
    # The eval file is question-disjoint from D_target ∪ D_algo ∪ probe by
    # construction (make_data --eval-set, verified at generation), so
    # nothing here was ever trained on. Belt and suspenders: the one way a
    # run could still have seen these questions is training on the eval
    # file itself.
    exp = manifest.data.get("experiment", {})
    if exp.get("data_order_hash") == cfg["data"]["order_hash"]:
        raise ValueError(
            f"{args.run}: trained on the G5 eval file itself — "
            "no un-trained eval questions exist for this run"
        )
    # Fixed slices of the reporting block (spec 02 §8) — the identical shots
    # and questions for every run, no sampling. Rows before EVAL_STOP_ROWS
    # are the in-loop stopping block and stay out of reported numbers.
    q_start = EVAL_STOP_ROWS + G5_N_SHOTS
    if len(df) < q_start + args.n:
        raise ValueError(f"--n {args.n}: eval file has {len(df)} rows, needs {q_start + args.n}")
    shots = df.iloc[EVAL_STOP_ROWS:q_start]["full_text"].tolist()
    rows = df.iloc[q_start : q_start + args.n]
    q_texts = rows["full_text"].tolist()
    q_spans = list(zip(rows["answer_char_start"].astype(int), rows["answer_char_end"].astype(int)))
    answers = rows["true_answer"].astype(int).tolist()

    def accuracy_with(exemplars: list[str]) -> float:
        # Compose exemplars + the COMPLETE query (true answer filled in) so
        # the composed text is a training-style rendering, then slice the
        # token prefix before the query's answer span — the same rule as
        # G1/G2/G3 (module docstring): a re-tokenized char-sliced prompt ends
        # in a standalone trailing-space token the model never saw. Zero-shot
        # is the identical procedure with exemplars == [].
        texts, char_spans = [], []
        for text, (cs, ce) in zip(q_texts, q_spans):
            composed = few_shot_prompt(exemplars, text)
            offset = len(composed) - len(text)  # the query is the join's last element
            texts.append(composed)
            char_spans.append((offset + cs, offset + ce))
        examples = tokenize_with_spans(texts, char_spans, tokenizer)
        prompt_ids = [ex.input_ids[: ex.label_span[0]] for ex in examples]
        completions = greedy_completions(
            model, tokenizer, prompt_ids, device=args.device, batch_size=args.batch_size
        )
        hits = [exact_match("Answer:" + c, a) for c, a in zip(completions, answers)]
        return sum(hits) / len(hits)

    zero = accuracy_with([])
    print(f"[evt] G5 zero-shot exact_match {zero:.4f} on n={args.n}")
    sixteen = accuracy_with(shots)
    print(f"[evt] G5 {G5_N_SHOTS}-shot exact_match {sixteen:.4f} on n={args.n}")

    # Shared-set test loss: masked NLL over the eval file's full reporting
    # block, append_eos=True to match the training tokenization — the same
    # number the runs-5/6 harness writes as θ_T test loss, so every run's
    # loss lands on identical data. --skip-test-loss for CPU smoke runs.
    test_loss, test_n = None, None
    if not args.skip_test_loss:
        rep = df.iloc[EVAL_STOP_ROWS:]
        rep_spans = list(
            zip(rep["answer_char_start"].astype(int), rep["answer_char_end"].astype(int))
        )
        rep_examples = tokenize_with_spans(
            rep["full_text"].tolist(), rep_spans, tokenizer, append_eos=True
        )
        task_format = TaskFormat(
            name=cfg["task"]["name"], format_version=cfg["task"]["format_version"]
        )
        test_loss = evaluate_sft_nll_nats(
            model, rep_examples, task_format, batch_size=args.batch_size, device=args.device
        )
        test_n = len(rep_examples)
        print(f"[evt] G5 shared-set test loss {test_loss:.4f} nats over n={test_n}")

    manifest.data.setdefault("experiment", {}).setdefault("gates", {})["G5"] = {
        # Always true: G5 is recorded evidence with no pass bar (spec 02 §8
        # gives expectations, not thresholds), and require_parent_ready
        # (spec 00 V0.6) refuses ANY child launch if a recorded gate lacks
        # pass: true — a false here would wrongly block the DAG.
        "pass": True,
        "zero_shot_accuracy": zero,
        "sixteen_shot_accuracy": sixteen,
        "test_loss_nats": test_loss,
        "test_loss_n": test_n,
        "n": args.n,
        "n_shots": G5_N_SHOTS,
        "eval_file": cfg["data"]["file"],
        "eval_order_hash": cfg["data"]["order_hash"],
        "eval_stop_rows": EVAL_STOP_ROWS,
        "checkpoint": str(checkpoint),
        "protocol": (
            "token-prefix prompts of few-shot composed texts (exemplars + "
            "complete query, blank-line separated), greedy EOS-stopped "
            "first-line, fixed slices of the frozen D_target_eval reporting "
            "block (question-disjoint from all training data by "
            "construction), exact_match + shared-set masked NLL; evidence "
            "only, no pass bar"
        ),
    }
    manifest.save(store / "runs" / args.run / "manifest.json")
    print(f"[evt] G5 evidence recorded in {store / 'runs' / args.run / 'manifest.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="gate", required=True)

    def common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run", required=True, help="run_id whose manifest records the verdict")
        p.add_argument(
            "--config",
            type=Path,
            required=True,
            help=(
                "eval-data YAML: the D_algo run YAML for g1/g2/g3, the installer "
                "YAML for g4, eval_target_data.yaml for g5"
            ),
        )
        p.add_argument(
            "--checkpoint",
            type=Path,
            default=None,
            help="default: the run's checkpoint via geode.zoo.checkpoint_dir "
            "(flat model/, or legacy <phase>/model)",
        )
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--batch-size", type=int, default=128)

    g1 = sub.add_parser("g1", help="Arm A near ceiling on NL add/sub (after run 2)")
    g2 = sub.add_parser("g2", help="arithmetic intact after the installer (run 3) — G1 protocol")
    g3 = sub.add_parser("g3", help="Arm B ~0%% on real NL add/sub — no label leak (after run 4)")
    for gate, p, threshold in (
        ("G1", g1, EXACT_MATCH_THRESHOLD),
        ("G2", g2, EXACT_MATCH_THRESHOLD),
        ("G3", g3, G3_LEAK_THRESHOLD),
    ):
        common_args(p)
        p.add_argument("--n", type=int, default=1024)
        p.add_argument("--sample-seed", type=int, default=316)
        p.add_argument("--threshold", type=float, default=threshold)
        p.add_argument(
            "--dump",
            type=int,
            default=0,
            help="print up to N misses (prompt -> completion) for failure diagnosis",
        )
        p.set_defaults(func=partial(run_exact_match_gate, gate=gate, invert=gate == "G3"))

    g4 = sub.add_parser("g4", help="format validity, in-loop metric re-scored (runs 3-4)")
    common_args(g4)
    g4.set_defaults(func=run_g4)

    g5 = sub.add_parser(
        "g5",
        help="zero/16-shot + shared-set test loss on D_target_eval — recorded evidence (runs 3-6)",
    )
    common_args(g5)
    g5.add_argument("--n", type=int, default=1024)
    g5.add_argument(
        "--skip-test-loss",
        action="store_true",
        help="skip the reporting-block NLL (CPU smoke); accuracies still record",
    )
    g5.set_defaults(func=run_g5)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
