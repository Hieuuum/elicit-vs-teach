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
completion``, pass at the config's threshold. ``--prompt-config`` (2026-07-26)
switches the prompt source to a FIXED slice of a frozen external eval file
for runs that carve no val split (the new-phase dose installers, spec 02 §6):
the run's own parquet is never read, ``--threshold`` carries the phase's
shared bar, and the recorded protocol names the file, hash and row range. **G5** (runs 3-6) measures
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
trailing-space incident as above). **G6** (phase-3 translation bridge) scores
text exact match in the answer slot over the entire frozen held-out bridge
eval. It requires both directions to clear the same threshold and records the
aggregate plus per-direction rates. G4/G5 reject ``arith_translate`` configs:
their integer parser and arithmetic reporting protocol do not apply to text
answers.

**G8** (ts38 mini, 2026-08-14) is the pre-taught parent's TinyStories
RETENTION gate: mean validation loss in nats/token on *run 1's own frozen
validation stream*, pass iff ``<= --bar`` (an absolute number the caller must
name — see ``run_g8`` for why it is not defaulted). A pre-training
intervention that teaches arithmetic while destroying the language model
underneath produces an "elicitation" arm that is really a damaged base; G8 is
the check that the parent is still the same model. It shares G4's
``--no-record`` / ``--record-only-pass`` record modes, for the same
shared-parent reason.

CPU-only friendly and no ``--confirm-cost``: evaluation, not training. The one
exception is G8's stream reconstruction, whose first (uncached) call downloads
~2 GB and spends 30-60 min of CPU packing 2.7M documents into ~4.3 GB of
resident int64 rows — run it once with ``--no-record`` before any GPU spend
and let ``--val-cache`` serve every later call.

Usage:
    python3 gates.py g1 --run evt-run2-armA-algo --config configs/archive/runs/run2_algo.yaml \
        [--checkpoint <dir>] [--device cuda] [--n 1024] [--sample-seed 316]
    python3 gates.py g2 --run evt-run3-sweep-lr3e-4 --config configs/archive/runs/run2_algo.yaml
    python3 gates.py g3 --run evt-run4-armB-inst --config configs/archive/runs/run2_algo.yaml
    python3 gates.py g4 --run evt-run3-sweep-lr3e-4 --config configs/archive/runs/run3_inst.yaml
    python3 gates.py g4 --run evt-p2-armA-dose1 --config configs/archive/phase2/p2_armA_dose.yaml \
        --prompt-config configs/eval_target_data.yaml --threshold 0.90
    python3 gates.py g5 --run evt-run3-armA-inst --config configs/eval_target_data.yaml
    python3 gates.py g8 --run evt-ts38-pretaught-parent \
        --config configs/archive/runs/run1_pretrain.yaml --bar 1.1718
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch

from geode.arith import (
    exact_match,
    exact_match_accuracy,
    few_shot_prompt,
    format_valid,
    greedy_completions,
    parse_answer,
    text_exact_match,
    tokenize_with_spans,
)
from geode.edl import EVAL_STOP_ROWS, G5_N_SHOTS
from geode.edl.masking import TaskFormat
from geode.train import evaluate_nll_nats, evaluate_sft_nll_nats, pack_corpus, split_indices
from geode.zoo import checkpoint_dir, load_model, load_run, tokenizer_hash
from train import REPO_ROOT, load_config, load_texts
from train_sft import load_frozen_parquet

# G1 and G2 share the bar: 0.95 is the committed definition of "capability
# present", no separate installer δ (owner 2026-07-21, spec 02 §8).
EXACT_MATCH_THRESHOLD = 0.95
# G6 uses the same "capability present" bar, applied independently to both
# translation directions so one cannot mask a weak reverse direction.
G6_THRESHOLD = 0.95
# G3 inverts the bar (spec 02 §8): Arm B's random-label installer must NOT
# have taught real add/sub — pass iff accuracy <= chance + margin. Exact match
# on signed multi-digit integers has ~0 chance rate; 0.02 is the margin.
G3_LEAK_THRESHOLD = 0.02


def gate_record_decision(passed: bool, no_record: bool, record_only_pass: bool) -> tuple[bool, int]:
    """Whether a gate should write its verdict to the manifest, and the exit
    code to return either way. Pure function (no manifest/checkpoint/model
    needed) so this decision has a direct unit test —
    tests/experiments/scripts/test_gates.py.

    Three modes, purely additive over the pre-2026-07-30 default:
    - default (both False): always records, pass or fail — existing callers
      rely on this (e.g. a deliberately-recorded G3 FAIL is the pass case).
    - ``no_record``: never writes, whatever the verdict — score a shared
      parent as a baseline before deciding whether to commit anything.
    - ``record_only_pass``: writes only on a pass; a fail writes nothing and
      exits nonzero. For the RECORDING call after a --no-record scoring pass
      already read PASS: without this, that second call recomputes its own
      verdict and would unconditionally write it, so a near-threshold
      recompute divergence could still land a FAIL on a shared parent's
      manifest — exactly what the --no-record pre-check exists to prevent
      (2026-07-30 fig2-installer review). On divergence the caller sees a
      nonzero exit with nothing recorded and can retry the whole score+record
      block rather than hand-editing the manifest.
    """
    if no_record:
        return False, (0 if passed else 1)
    if record_only_pass and not passed:
        return False, 1
    return True, (0 if passed else 1)


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

    accuracy, completions = exact_match_accuracy(
        model,
        tokenizer,
        prompt_ids,
        answers,
        device=args.device,
        batch_size=args.batch_size,
    )
    # The completion starts inside the answer slot (its leading space rides on
    # the first answer token), so "Answer:" + completion re-enters the tested
    # parser with the marker it expects.
    hits = [exact_match("Answer:" + c, a) for c, a in zip(completions, answers)]
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
    # Same reason G4/G5 have --no-record: a recorded sub-threshold verdict on
    # a shared parent makes require_parent_ready (V0.6) refuse every child of
    # it, and un-recording is manual surgery on a manifest. --record-only-pass
    # closes the remaining gap (gate_record_decision docstring): the RECORDING
    # call after a passing --no-record score must not itself write a FAIL if
    # its own recompute diverges near the threshold.
    should_record, exit_code = gate_record_decision(
        passed, getattr(args, "no_record", False), getattr(args, "record_only_pass", False)
    )
    if not should_record:
        if getattr(args, "no_record", False):
            print("[evt] --no-record: nothing written to any manifest")
        else:
            print(f"[evt] --record-only-pass: {gate} did not pass — nothing written to any manifest")
        return exit_code
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
    return exit_code


def _reject_translate_config(cfg: dict, gate: str) -> None:
    if cfg["task"]["name"] == "arith_translate":
        raise SystemExit(
            f"[evt] {gate}: arith_translate has text answers; this gate uses the integer "
            "answer protocol. Use g6 for the held-out translation check."
        )


def run_g4(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, None)
    _reject_translate_config(cfg, "G4")
    pcfg = None
    if args.prompt_config is not None:
        pcfg = load_config(args.prompt_config, None)
        _reject_translate_config(pcfg, "G4")
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
    threshold = args.threshold
    if threshold is None:
        if s.get("metric") != "format_validity":
            raise SystemExit(
                "[evt] G4: this run's config stops on the loss plateau, so it carries no "
                "format-validity threshold — pass --threshold explicitly (the phase's "
                "shared G4 bar; 0.90 for the 2026-07-26 new phase)"
            )
        threshold = s["threshold"]

    if args.prompt_config is None:
        if s.get("metric") != "format_validity":
            raise SystemExit(
                "[evt] G4 needs an installer config with a behavioral stopping block "
                "(e.g. run3_inst.yaml) — it re-scores the in-loop metric. A run with no "
                "val split (dose runs, 2026-07-26) scores an external frozen prompt file "
                "instead: --prompt-config configs/eval_target_data.yaml"
            )
        df = load_frozen_parquet(cfg)
        _, val_idx = split_indices(len(df), cfg["data"]["val_fraction"], cfg["data"]["seed"])
        # Identical prompt set to the in-loop stopping eval (train_sft.py): the
        # seeded sample is over val-split *positions*, so re-deriving it here and
        # indexing into val_idx selects the same questions the trainer scored.
        picks = random.Random(s["prompt_seed"]).sample(range(len(val_idx)), s["n_prompts"])
        rows = df.iloc[[val_idx[i] for i in picks]]
        n_prompts = s["n_prompts"]
        source = {"prompt_seed": s["prompt_seed"]}
        protocol = (
            "token-prefix prompts, greedy EOS-stopped, in-loop stopping sample "
            "(seeded over val-split positions), format_valid"
        )
    else:
        # No-val runs (dose installers): prompts come from a frozen external
        # eval file, hash-verified through its own config pin. The run's own
        # parquet is never loaded — a dose config points at an artifact that
        # may not be published, and the questions it trained on are exactly
        # what a format check must not use. FIXED slice after the target runs'
        # in-loop stopping block: no sampling, so every run scores the
        # identical prompts.
        assert pcfg is not None
        pdf = load_frozen_parquet(pcfg)
        n_prompts = args.n_prompts
        stop, end = EVAL_STOP_ROWS, EVAL_STOP_ROWS + n_prompts
        if len(pdf) < end:
            raise SystemExit(
                f"[evt] G4: --n-prompts {n_prompts} needs {end} rows, "
                f"{pcfg['data']['file']} has {len(pdf)}"
            )
        rows = pdf.iloc[stop:end]
        source = {
            "prompt_file": pcfg["data"]["file"],
            "prompt_order_hash": pcfg["data"]["order_hash"],
            "prompt_rows": [stop, end],
        }
        protocol = (
            f"token-prefix prompts, greedy EOS-stopped, FIXED rows [{stop}:{end}] of the "
            f"frozen external {pcfg['data']['file']} (no sampling, identical for every "
            "run; skips the target runs' in-loop stopping block), format_valid"
        )
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
    passed = rate >= threshold
    print(
        f"[evt] G4 format_validity {rate:.4f} on n={n_prompts} -> "
        f"{'PASS' if passed else 'FAIL'} (threshold {threshold})"
    )

    # Baseline mode (2026-07-26): --no-record scores a checkpoint WITHOUT
    # writing a verdict — a sub-threshold verdict on a shared parent would
    # make require_parent_ready refuse every existing child of it (V0.6).
    # --record-only-pass closes the remaining gap for the RECORDING call that
    # follows a passing --no-record score (gate_record_decision docstring).
    should_record, exit_code = gate_record_decision(
        passed, args.no_record, getattr(args, "record_only_pass", False)
    )
    if not should_record:
        if args.no_record:
            print("[evt] --no-record: nothing written to any manifest")
        else:
            print("[evt] --record-only-pass: G4 did not pass — nothing written to any manifest")
        return exit_code

    manifest.data.setdefault("experiment", {}).setdefault("gates", {})["G4"] = {
        "pass": passed,
        "format_validity": rate,
        "n": n_prompts,
        "threshold": threshold,
        **source,
        "checkpoint": str(checkpoint),
        "protocol": protocol,
    }
    manifest.save(store / "runs" / args.run / "manifest.json")
    print(f"[evt] G4 verdict recorded in {store / 'runs' / args.run / 'manifest.json'}")
    return exit_code


def run_g5(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, None)
    _reject_translate_config(cfg, "G5")
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
        accuracy, _ = exact_match_accuracy(
            model,
            tokenizer,
            prompt_ids,
            answers,
            device=args.device,
            batch_size=args.batch_size,
        )
        return accuracy

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

    if args.no_record:
        # Baseline mode, same rationale as G4's: score a SHARED parent without
        # writing to its manifest. For the new phase's dose curve this is the
        # n=0 intercept — the elicit arm's parent measured on the identical
        # eval file and protocol as every dose, which is what makes "the dose
        # changed X" a statement rather than an assumption.
        print("[evt] --no-record: nothing written to any manifest")
        return 0

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


def _translation_rates(
    completions: list[str], answers: list[str], directions: list[str]
) -> tuple[float, dict[str, float]]:
    if not (len(completions) == len(answers) == len(directions)):
        raise ValueError("G6 completions, answers, and directions must have equal lengths")
    if set(directions) != {"to_op", "to_nl"}:
        raise ValueError(
            f"G6 eval file must contain both directions, got {sorted(set(directions))}"
        )
    hits = [text_exact_match("Answer:" + c, a) for c, a in zip(completions, answers)]
    by_direction = {
        direction: sum(h for h, d in zip(hits, directions) if d == direction)
        / directions.count(direction)
        for direction in sorted(set(directions))
    }
    return sum(hits) / len(hits), by_direction


def run_g6(args: argparse.Namespace) -> int:
    """Held-out bidirectional exact match for the phase-3 translation bridge."""
    cfg = load_config(args.config, None)
    if cfg["task"]["name"] != "arith_translate":
        raise SystemExit(f"[evt] G6 needs task.name=arith_translate, got {cfg['task']['name']!r}")
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])
    manifest = load_run(args.run, store=store)

    from transformers import AutoTokenizer

    local = (args.config.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    checkpoint = args.checkpoint or checkpoint_dir(args.run, store=store)
    print(f"[evt] G6: loading checkpoint {checkpoint} ...", flush=True)
    model = load_model(args.run, store=store, device=args.device, checkpoint=checkpoint)

    df = load_frozen_parquet(cfg)
    required = {"full_text", "answer_char_start", "answer_char_end", "answer_text", "direction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"G6 eval file lacks required columns: {sorted(missing)}")
    directions = df["direction"].astype(str).tolist()

    texts = df["full_text"].tolist()
    char_spans = list(zip(df["answer_char_start"].astype(int), df["answer_char_end"].astype(int)))
    examples = tokenize_with_spans(texts, char_spans, tokenizer)
    prompt_ids = [ex.input_ids[: ex.label_span[0]] for ex in examples]
    completions = greedy_completions(
        model,
        tokenizer,
        prompt_ids,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    answers = df["answer_text"].astype(str).tolist()
    accuracy, by_direction = _translation_rates(completions, answers, directions)
    passed = accuracy >= args.threshold and all(v >= args.threshold for v in by_direction.values())
    print(
        f"[evt] G6 translation exact_match {accuracy:.4f} on n={len(df)} "
        f"(by direction: {by_direction}) -> {'PASS' if passed else 'FAIL'} "
        f"(each >= {args.threshold})"
    )
    if args.dump:
        misses = [
            i
            for i, (completion, answer) in enumerate(zip(completions, answers))
            if not text_exact_match("Answer:" + completion, answer)
        ]
        for i in misses[: args.dump]:
            print(
                f"  {df.iloc[i]['direction']}: {tokenizer.decode(prompt_ids[i])!r} -> "
                f"{completions[i]!r} (want {answers[i]!r})"
            )

    if args.no_record:
        print("[evt] --no-record: nothing written to any manifest")
        return 0 if passed else 1
    manifest.data.setdefault("experiment", {}).setdefault("gates", {})["G6"] = {
        "pass": passed,
        "accuracy": accuracy,
        "accuracy_by_direction": by_direction,
        "n": len(df),
        "threshold": args.threshold,
        "eval_file": cfg["data"]["file"],
        "eval_order_hash": cfg["data"]["order_hash"],
        "checkpoint": str(checkpoint),
        "protocol": (
            "token-prefix prompts from the frozen held-out translation corpus, greedy "
            "EOS-stopped first-line decode, exact text match in the answer slot after "
            "surrounding-whitespace trim; aggregate and both directions must clear threshold"
        ),
    }
    manifest.save(store / "runs" / args.run / "manifest.json")
    print(f"[evt] G6 verdict recorded in {store / 'runs' / args.run / 'manifest.json'}")
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# G8 — TinyStories retention on run 1's own frozen validation stream
# ---------------------------------------------------------------------------
# The base of the ts38 family. Its manifest carries every fingerprint G8
# checks the reconstruction against, and its checkpoint is the anchor.
G8_BASE_RUN = "evt-run1-base-v3-ext"
# Anchor tolerance. The rebuilt stream is scored with the SAME evaluator
# (evaluate_nll_nats, fp32, batch-size invariant by V5.19) against the SAME
# fp32 checkpoint that produced the recorded number, so agreement should be
# at float noise; 1e-3 nats leaves room for GPU-vs-CPU reduction order only.
G8_ANCHOR_TOL_NATS = 1e-3


def _run1_val_stream(
    cfg: dict, tokenizer: Any, base_data: dict, cache: Path | None
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Rebuild run 1's frozen TinyStories validation stream, exactly.

    **Why this is reconstructible** (investigated 2026-08-14, before G8 was
    written — a retention number measured on any *other* validation stream
    would silently redefine the measurement):

    Run 1's val rows are a pure function of four pinned inputs, with no
    stored artifact needed —

    1. ``geode.train.split_documents`` over the frozen HF file
       ``roneneldan/TinyStories:TinyStoriesV2-GPT4-train.txt`` (deterministic,
       ``max_documents: null`` so the whole file is taken);
    2. ``geode.train.pack_corpus(texts, tokenizer, seq_len=512)`` — documented
       and property-tested as a pure function of exactly those three inputs
       (V5.17/V5.27), with the frozen 10K custom BPE whose sha256 the base
       manifest records;
    3. ``geode.train.split_indices(n, val_fraction=0.005, seed=316)`` — the
       seeded ``torch.randperm`` partition, byte-identical to the
       ``train_val_split`` the trainer called (V5.39);
    4. all four values pinned in ``configs/archive/runs/run1_pretrain.yaml``'s
       ``data`` block, and independently re-recorded in the base run's own
       manifest under ``experiment.data_config``.

    Three exact counts must agree with the base manifest before the stream is
    accepted — documents (2,717,495), total packed rows, and val rows (5,225).
    They are what actually pin the packing: a drifted dataset file, tokenizer,
    or ``seq_len`` moves at least one of them.

    The permutation is the one input the counts cannot pin (a different
    ``torch.randperm`` draw yields the same *number* of rows). ``run_g8``
    therefore also scores the base checkpoint itself as an anchor, and this
    function records ``val_index_sha256`` plus the ``torch`` version in the
    verdict so permutation drift stays auditable across runs even where the
    loss cannot resolve it.

    ``cache`` (a ``.pt`` path) skips the ~45-minute repack on later calls; the
    stored key refuses a cache built under any different pin.
    """
    d = cfg["data"]
    # An eval-data pin has a data block too, but not these keys — say so, rather
    # than dying on a bare KeyError three lines down.
    absent = [k for k in ("hf_id", "file", "seq_len", "val_fraction", "seed") if k not in d]
    if absent:
        raise SystemExit(
            f"[evt] G8: {absent} missing from {cfg.get('run_id', '<config>')}'s data block. "
            "--config must be run 1's PRETRAIN yaml (the corpus/packing/split pins), not "
            "an eval-data pin: configs/archive/runs/run1_pretrain.yaml"
        )
    expected = {
        "data_file": base_data["experiment"]["data_config"]["file"],
        "seq_len": base_data["experiment"]["data_config"]["seq_len"],
        "val_fraction": base_data["experiment"]["data_config"]["val_fraction"],
        "split_seed": base_data["dataset"]["seed"],
        "tokenizer_sha256": base_data["experiment"]["tokenizer"]["sha256"],
        "base_run": base_data["run_id"],
    }
    got = {
        "data_file": d["file"],
        "seq_len": d["seq_len"],
        "val_fraction": d["val_fraction"],
        "split_seed": d["seed"],
        "tokenizer_sha256": tokenizer_hash(tokenizer),
        "base_run": base_data["run_id"],
    }
    if got != expected:
        raise SystemExit(
            f"[evt] G8: the --config data/tokenizer pins do not match how "
            f"{base_data['run_id']} was trained, so the rebuilt stream would not be its "
            f"validation stream.\n  config+tokenizer: {got}\n  base manifest:    {expected}\n"
            "  Pass run 1's own pretrain config and the frozen ../tokenizer artifact."
        )
    if base_data["experiment"]["data_config"]["max_documents"] is not None:
        raise SystemExit(
            "[evt] G8: the base run packed a truncated corpus "
            f"(max_documents={base_data['experiment']['data_config']['max_documents']}); "
            "this reconstruction takes the whole file and would not reproduce it."
        )

    n_docs_want = base_data["dataset"]["n_unique_examples"]
    n_train_want = base_data["experiment"]["data_config"]["n_train_seqs"]
    n_val_want = base_data["experiment"]["data_config"]["n_val_seqs"]

    if cache is not None and cache.is_file():
        print(f"[evt] G8: loading cached val stream {cache} ...", flush=True)
        blob = torch.load(cache)
        if blob["key"] != got:
            raise SystemExit(
                f"[evt] G8: --val-cache {cache} was built for {blob['key']}, this call "
                f"needs {got} — delete it rather than scoring the wrong stream."
            )
        val_seqs = blob["val_seqs"].long()
        fingerprints = blob["fingerprints"]
    else:
        print(
            f"[evt] G8: rebuilding {base_data['run_id']}'s validation stream "
            f"({d['hf_id']}:{d['file']}, seq_len {d['seq_len']}) — download + pack is "
            "30-60 min of CPU and ~4.3 GB resident on a cache miss ...",
            flush=True,
        )
        texts = load_texts(cfg)
        if len(texts) != n_docs_want:
            raise SystemExit(
                f"[evt] G8: rebuilt {len(texts)} documents, base manifest recorded "
                f"{n_docs_want} — the dataset file this run trained on is not the file "
                "that just downloaded. Do NOT score a substitute stream; halt."
            )
        seqs = pack_corpus(texts, tokenizer, d["seq_len"])
        del texts
        if len(seqs) != n_train_want + n_val_want:
            raise SystemExit(
                f"[evt] G8: packed {len(seqs)} rows, base manifest recorded "
                f"{n_train_want + n_val_want} ({n_train_want} train + {n_val_want} val) — "
                "the packing differs (tokenizer or seq_len drift). Halt."
            )
        train_idx, val_idx = split_indices(len(seqs), d["val_fraction"], d["seed"])
        if (len(train_idx), len(val_idx)) != (n_train_want, n_val_want):
            raise SystemExit(
                f"[evt] G8: split gave {len(train_idx)}/{len(val_idx)} train/val rows, "
                f"base manifest recorded {n_train_want}/{n_val_want}. Halt."
            )
        val_seqs = seqs[val_idx]
        del seqs
        fingerprints = {
            "n_documents": n_docs_want,
            "n_train_seqs": n_train_want,
            "n_val_seqs": n_val_want,
            "val_index_sha256": hashlib.sha256(
                ",".join(map(str, val_idx)).encode("utf-8")
            ).hexdigest(),
            # str(): torch.__version__ is a TorchVersion object, which
            # torch.load's weights_only=True default (>=2.6) refuses to
            # unpickle out of the cache blob.
            "torch_version": str(torch.__version__),
        }
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            # uint16 storage: vocab 10K << 65536, same trick train.py's
            # --packed-cache uses. 5,225 x 512 rows -> ~5 MB.
            torch.save(
                {"key": got, "val_seqs": val_seqs.to(torch.uint16), "fingerprints": fingerprints},
                cache,
            )
            print(f"[evt] G8: wrote val-stream cache {cache}", flush=True)
    print(
        f"[evt] G8: val stream {tuple(val_seqs.shape)} "
        f"(docs {fingerprints['n_documents']}, val_index_sha256 "
        f"{fingerprints['val_index_sha256'][:16]}…) — counts match {base_data['run_id']}",
        flush=True,
    )
    return val_seqs, fingerprints


def run_g8(args: argparse.Namespace) -> int:
    """TinyStories retention on run 1's own frozen validation stream.

    Metric: mean next-token cross-entropy in nats/token over every predicted
    position of the rebuilt val rows — ``geode.train.evaluate_nll_nats``, the
    identical evaluator and reduction that wrote run 1's ``eval_log.jsonl``,
    so the number is directly comparable to its recorded ``min_val_nats``.
    PASS iff ``<= --bar``.

    ``--bar`` is REQUIRED and absolute. Gate thresholds are task-scoped
    ([[feedback-gate-thresholds-are-task-scoped]]): the ts38 mini's bar is
    1.1718 = the base's 1.0718 plus a delta pre-declared at 0.10 nats *before*
    any parent trained. Defaulting it here would let a later family inherit a
    number nobody re-declared. The verdict also records the base's loss
    measured on this same stream and the delta against it, so a re-baselining
    decision never needs a re-run.

    Anchor (always on): the base checkpoint is scored on the same rebuilt
    stream and must reproduce the manifest's ``min_val_nats`` to within
    ``--anchor-tol``. ``train_full`` saves the FINAL-step model with no
    restore-to-best, and run 1's final step *is* its minimum-val step, so an
    agreeing anchor proves the reconstruction end to end. A disagreement is a
    hard refusal, never a substituted val set (owner rule): the fix is to pin
    the environment or, if the owner chooses, re-baseline the bar against the
    measured base value this gate prints.
    """
    cfg = load_config(args.config, None)
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])
    manifest = load_run(args.run, store=store)
    base_manifest = load_run(args.base_run, store=store)
    # Resolved BEFORE the stream is built: on a cache miss that is a ~45-minute
    # pack, and discovering a missing checkpoint afterwards would throw it away.
    checkpoint = args.checkpoint or checkpoint_dir(args.run, store=store)
    checkpoint_dir(args.base_run, store=store)  # the anchor must exist too

    from transformers import AutoTokenizer

    # NOT the other gates' `local if local.is_dir() else <hf id>` line. Their
    # configs sit directly in configs/, where `../tokenizer` lands on the frozen
    # artifact; run 1's pretrain config has since been ARCHIVED two directories
    # deeper, so the same relative hop resolves to configs/archive/tokenizer,
    # which does not exist. Falling through to the HF-id branch would then ask
    # the hub for a repo literally named "../tokenizer" — or, run from
    # scripts/, silently succeed on a CWD-relative match. Neither is a resolver.
    # A directory is required (G8 is only ever about run 1's custom 10K BPE) and
    # _run1_val_stream still refuses anything whose sha256 misses the manifest.
    tok_path = args.tokenizer or (args.config.parent / cfg["tokenizer"]["path"])
    tok_dir = Path(tok_path).resolve()
    if not tok_dir.is_dir():
        raise SystemExit(
            f"[evt] G8: no tokenizer directory at {tok_dir} (from "
            f"{'--tokenizer' if args.tokenizer else f'{args.config} tokenizer.path'} "
            f"{tok_path}). Pass --tokenizer <experiments/training-run/tokenizer>."
        )
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    val_seqs, fingerprints = _run1_val_stream(cfg, tokenizer, base_manifest.data, args.val_cache)

    anchor_want = base_manifest.data["experiment"]["pretrain_result"]["min_val_nats"]
    print(f"[evt] G8: anchor — scoring {args.base_run}'s checkpoint on the rebuilt stream ...")
    base_model = load_model(args.base_run, store=store, device=args.device)
    anchor_got = evaluate_nll_nats(
        base_model, val_seqs, batch_size=args.batch_size, device=args.device
    )
    print(
        f"[evt] G8 anchor: {args.base_run} scores {anchor_got:.6f} nats, manifest recorded "
        f"{anchor_want:.6f} (|Δ| {abs(anchor_got - anchor_want):.2e}, tol {args.anchor_tol})"
    )
    if abs(anchor_got - anchor_want) > args.anchor_tol:
        raise SystemExit(
            f"[evt] G8: the rebuilt stream does NOT reproduce {args.base_run}'s recorded "
            f"validation loss ({anchor_got:.6f} vs {anchor_want:.6f}). Every exact count "
            "matched, so the likely cause is a torch-version drift in the seeded "
            "randperm — a different random 0.5% of the same corpus. Refusing to score "
            "retention on a stream that is not the one the bar was declared against. "
            "Fix the environment, or have the owner re-baseline the bar against the "
            f"measured {anchor_got:.6f} and re-declare the delta in decisions.md."
        )

    if args.run == args.base_run and args.checkpoint is None:
        # The pre-flight call (stage 1) scores the base against itself: same
        # checkpoint, same stream, so a second forward pass could only differ
        # by float noise. Reuse the anchor rather than pay for it twice.
        val_loss = anchor_got
    else:
        print(f"[evt] G8: loading checkpoint {checkpoint} ...", flush=True)
        model = load_model(args.run, store=store, device=args.device, checkpoint=checkpoint)
        val_loss = evaluate_nll_nats(
            model, val_seqs, batch_size=args.batch_size, device=args.device
        )
    passed = val_loss <= args.bar
    print(
        f"[evt] G8 val_loss_nats {val_loss:.4f} on n={len(val_seqs)} "
        f"(base {anchor_got:.4f}, delta {val_loss - anchor_got:+.4f}) -> "
        f"{'PASS' if passed else 'FAIL'} (threshold <= {args.bar})"
    )

    should_record, exit_code = gate_record_decision(passed, args.no_record, args.record_only_pass)
    if not should_record:
        if args.no_record:
            print("[evt] --no-record: nothing written to any manifest")
        else:
            print("[evt] --record-only-pass: G8 did not pass — nothing written to any manifest")
        return exit_code

    manifest.data.setdefault("experiment", {}).setdefault("gates", {})["G8"] = {
        "pass": passed,
        "val_loss_nats": val_loss,
        "bar_nats": args.bar,
        "base_run": args.base_run,
        "base_val_nats_measured": anchor_got,
        "base_val_nats_recorded": anchor_want,
        "delta_vs_base_nats": val_loss - anchor_got,
        "n": len(val_seqs),
        "checkpoint": str(checkpoint),
        **fingerprints,
        "protocol": (
            "mean next-token CE (nats/token, evaluate_nll_nats) over run 1's own "
            "validation stream, rebuilt from the pinned corpus + frozen tokenizer + "
            "seeded split and anchor-verified by re-scoring the base checkpoint against "
            "its recorded min_val_nats; pass = loss <= an absolute pre-declared bar"
        ),
    }
    manifest.save(store / "runs" / args.run / "manifest.json")
    print(f"[evt] G8 verdict recorded in {store / 'runs' / args.run / 'manifest.json'}")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    """Split out of ``main`` so the launcher shells' invocations can be checked
    at parse level without running a gate (tests/scripts/test_launcher_gate_args.py)."""
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
                "YAML for g4, eval_target_data.yaml for g5, bridge-eval YAML for g6"
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
        record_mode = p.add_mutually_exclusive_group()
        record_mode.add_argument(
            "--no-record",
            action="store_true",
            help="print the accuracy but write no verdict — score a shared parent "
            "before committing a pass/fail to its manifest (a recorded failure "
            "there would block every child)",
        )
        record_mode.add_argument(
            "--record-only-pass",
            action="store_true",
            help="record the verdict only if it passes; on a fail, write nothing "
            "and exit nonzero — for the RECORDING call after an already-passing "
            "--no-record score, so a recompute that diverges near the threshold "
            "can never land a FAIL on a shared parent's manifest",
        )
        p.add_argument(
            "--dump",
            type=int,
            default=0,
            help="print up to N misses (prompt -> completion) for failure diagnosis",
        )
        p.set_defaults(func=partial(run_exact_match_gate, gate=gate, invert=gate == "G3"))

    g4 = sub.add_parser("g4", help="format validity, in-loop metric re-scored (runs 3-4)")
    common_args(g4)
    g4.add_argument(
        "--prompt-config",
        type=Path,
        default=None,
        help="eval-data YAML supplying prompts from a frozen external file "
        "(e.g. eval_target_data.yaml) — required for a run with no val split",
    )
    g4.add_argument(
        "--n-prompts", type=int, default=512, help="--prompt-config only; in-loop default 512"
    )
    g4.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="default: the run config's stopping threshold; required when it has none",
    )
    g4_record_mode = g4.add_mutually_exclusive_group()
    g4_record_mode.add_argument(
        "--no-record",
        action="store_true",
        help="print the rate but write no verdict — for scoring a shared parent "
        "checkpoint as a baseline (a recorded failure there would block its children)",
    )
    g4_record_mode.add_argument(
        "--record-only-pass",
        action="store_true",
        help="record the verdict only if it passes; on a fail, write nothing "
        "and exit nonzero — for the RECORDING call after an already-passing "
        "--no-record score, so a recompute that diverges near the threshold "
        "can never land a FAIL on a shared parent's manifest",
    )
    g4.set_defaults(func=run_g4)

    g5 = sub.add_parser(
        "g5",
        help="zero/16-shot + shared-set test loss on D_target_eval — recorded evidence (runs 3-6)",
    )
    common_args(g5)
    g5.add_argument("--n", type=int, default=1024)
    g5.add_argument(
        "--no-record",
        action="store_true",
        help="print the numbers but write no evidence — for scoring a shared "
        "parent as the dose curve's n=0 baseline",
    )
    g5.add_argument(
        "--skip-test-loss",
        action="store_true",
        help="skip the reporting-block NLL (CPU smoke); accuracies still record",
    )
    g5.set_defaults(func=run_g5)

    g6 = sub.add_parser(
        "g6",
        help="bidirectional exact text match on the frozen phase-3 bridge eval",
    )
    common_args(g6)
    g6.add_argument(
        "--threshold",
        type=float,
        default=G6_THRESHOLD,
        help=f"minimum aggregate and per-direction exact-match rate (default: {G6_THRESHOLD})",
    )
    g6.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="decode ceiling; bridge answers use up to 26 answer tokens plus EOS",
    )
    g6.add_argument(
        "--no-record",
        action="store_true",
        help="print the rates but write no verdict",
    )
    g6.add_argument(
        "--dump",
        type=int,
        default=0,
        help="print up to N misses (prompt -> completion) for failure diagnosis",
    )
    g6.set_defaults(func=run_g6)

    g8 = sub.add_parser(
        "g8",
        help="TinyStories retention: val loss on run 1's own frozen validation stream",
    )
    # Declared inline rather than via common_args(): g8's --config is run 1's
    # PRETRAIN yaml (a corpus/tokenizer/split pin), not an eval-data yaml, and
    # it carries --base-run / --bar / --val-cache that no other gate has.
    g8.add_argument("--run", required=True, help="run_id whose manifest records the verdict")
    g8.add_argument(
        "--config",
        type=Path,
        required=True,
        help="run 1's pretrain YAML — the corpus, tokenizer, seq_len and split pins "
        "the validation stream is rebuilt from (configs/archive/runs/run1_pretrain.yaml)",
    )
    g8.add_argument(
        "--bar",
        type=float,
        required=True,
        help="absolute pass bar in nats/token; required, never defaulted (gate "
        "thresholds are task-scoped). ts38 mini: 1.1718 = base 1.0718 + a delta "
        "pre-declared at 0.10",
    )
    g8.add_argument(
        "--base-run",
        default=G8_BASE_RUN,
        help=f"run supplying the stream fingerprints and the anchor checkpoint "
        f"(default: {G8_BASE_RUN})",
    )
    g8.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="tokenizer DIRECTORY (default: --config's tokenizer.path resolved "
        "against the config dir). run 1's pretrain config was archived two levels "
        "below where it launched, so its relative '../tokenizer' no longer lands "
        "on the frozen artifact — pass experiments/training-run/tokenizer",
    )
    g8.add_argument(
        "--val-cache",
        type=Path,
        default=None,
        help="'.pt' path for the rebuilt val rows; written on a miss, reused after. "
        "Keep it OUT of runs/<id>/ so it never rides a relay push",
    )
    g8.add_argument(
        "--anchor-tol",
        type=float,
        default=G8_ANCHOR_TOL_NATS,
        help=f"max |measured - recorded| base loss before the reconstruction is "
        f"refused (default: {G8_ANCHOR_TOL_NATS})",
    )
    g8.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="default: the run's checkpoint via geode.zoo.checkpoint_dir",
    )
    g8.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g8.add_argument("--batch-size", type=int, default=32)
    g8_record_mode = g8.add_mutually_exclusive_group()
    g8_record_mode.add_argument(
        "--no-record",
        action="store_true",
        help="print the loss but write no verdict — for scoring a shared parent "
        "before committing a pass/fail that would block every child",
    )
    g8_record_mode.add_argument(
        "--record-only-pass",
        action="store_true",
        help="record the verdict only if it passes; on a fail, write nothing and "
        "exit nonzero — for the RECORDING call after an already-passing --no-record "
        "score",
    )
    g8.set_defaults(func=run_g8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
