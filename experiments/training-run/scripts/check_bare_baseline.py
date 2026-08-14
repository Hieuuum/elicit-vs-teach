"""Bare-format premise guard: the base model must be ~0% on BARE prompts.

Written for fig2nl3 (EXPERIMENTS §6.13), where the family exists on one
premise: without the Question:/Answer: scaffold, base Llama treats a math
question as text to continue rather than a question to answer — the paper's
stated regime ("these base models achieve 0% zero-shot accuracy"), and the
precondition for the Figure-2 pre-elicit transient to exist at all. The
scaffolded families measured base at ~0.31 zero-shot EM / ~0.83 format
validity, which is exactly why their arms coincided (§6.12 outcome). This
guard MEASURES the premise on the actual bare eval file BEFORE any training
happens; if the base still answers bare prompts, the family is pointless and
must halt for owner triage, not spend a GPU-day confirming nothing.

Parametrized 2026-08-14 for the ts38 mini (§6.14) without changing any
default, so the fig2nl3 invocation above keeps working verbatim. ts38 runs
the same guard against the 38.7M TinyStories base and the frozen 10K custom
BPE, and adds the two things that family needs:

- ``--max-format-validity``: ts38's premise is EM <= 0.05 **and** format
  validity <= 0.05. Both are premise conditions there — a base that emits
  well-formed answers already carries the output convention, which is the
  fig2nl2 failure mode. Unset (fig2nl3's default) leaves the EM-only bar.
- ``--step0-json``: guard 5 of the ts38 plan (the FIXED-COST control). A
  constant per-example digit cost decays like 1/n and can by itself fake a
  decreasing EDL/D limb, so the base's step-0 mean LABEL-token loss on this
  same slice is measured once and written out for the analysis to subtract
  against. It is computed with ``evaluate_sft_nll_nats`` and the eval config's
  ``TaskFormat`` — the identical call ``gates.py g5`` makes for its shared-set
  test loss, so the number is comparable to every run's test floor rather
  than to a hand-rolled mask.

Protocol: greedy EOS-stopped completions (the shared G4/G5 decode path) on a
fixed slice of the eval file's REPORTING block (rows
[EVAL_STOP_ROWS : EVAL_STOP_ROWS+n], the same rows G4 scores, hash-verified
through the eval config's pin), scored with the shared format_valid /
exact_match helpers. Tokenizing the slice through tokenize_with_spans also
proves the bare format's char->token span alignment on the real tokenizer
before any trainer touches the file.

PASS: zero-shot EM <= --max-em (and format validity <= --max-format-validity
when given) — the premise holds.
FAIL: exit 1 with both numbers printed — do NOT launch the sweep.

Box-only (GPU recommended; the fig2nl3 default base is a gated hub model).
Read-only with respect to every manifest and checkpoint: it loads the base
model, generates, prints, and writes nothing except the optional
``--step0-json``.

Usage:
    # fig2nl3 (unchanged)
    python3 check_bare_baseline.py [--n 256] [--max-em 0.05] [--device cuda]
    # ts38 mini
    python3 check_bare_baseline.py --eval-config ../configs/eval_bare_target_data.yaml \
        --base-model "$GEODE_STORE/runs/evt-run1-base-v3-ext/model" --dtype fp32 \
        --n 256 --max-em 0.05 --max-format-validity 0.05 \
        --step0-json "$GEODE_STORE/results/ts38_step0_baseline.json"
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import torch

from geode.arith import format_valid, parse_answer, tokenize_with_spans
from geode.arith.decode import greedy_completions
from geode.edl import EVAL_STOP_ROWS
from geode.edl.masking import TaskFormat
from geode.train import evaluate_sft_nll_nats
from geode.zoo import tokenizer_hash
from train import load_config  # scripts/ shared config loader (adds merge + pins)
from train_sft import load_frozen_parquet

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
EVAL_CONFIG = CONFIGS / "eval_bare_target_data_llama.yaml"
BASE_MODEL = "meta-llama/Llama-3.2-1B"
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=256, help="prompts scored (fixed slice, no sampling)")
    ap.add_argument("--max-em", type=float, default=0.05, help="premise bar: base EM must be <= this")
    ap.add_argument(
        "--max-format-validity",
        type=float,
        default=None,
        help="second premise bar: base format validity must be <= this (ts38: 0.05). "
        "Unset leaves the EM-only bar (fig2nl3 behavior)",
    )
    ap.add_argument(
        "--eval-config",
        type=Path,
        default=EVAL_CONFIG,
        help=f"eval-data YAML pinning the bare eval file and the tokenizer that must "
        f"match the model under test (default: {EVAL_CONFIG.name})",
    )
    ap.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help=f"hub id or local save_pretrained dir of the BASE model, no adapter "
        f"(default: {BASE_MODEL})",
    )
    ap.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="bf16",
        help="load dtype; fp32 for the small TinyStories base, so the step-0 label "
        "loss is comparable to the runs' fp32 test floors (default: bf16)",
    )
    ap.add_argument(
        "--step0-json",
        type=Path,
        default=None,
        help="write the step-0 mean label-token loss (nats) + both rates here — the "
        "ts38 fixed-cost control. Omitted: nothing is written anywhere",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.eval_config, None)
    df = load_frozen_parquet(cfg)  # hash-verified against the config pin
    stop, end = EVAL_STOP_ROWS, EVAL_STOP_ROWS + args.n
    if len(df) < end:
        raise SystemExit(f"[premise] --n {args.n} needs {end} rows, eval file has {len(df)}")
    rows = df.iloc[stop:end]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Relative tokenizer paths resolve against the config's directory (the
    # gates.py rule); anything that is not a local dir passes through as an HF
    # id, so the Llama default is unaffected.
    tok_path = cfg["tokenizer"]["path"]
    local = (args.eval_config.parent / tok_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else tok_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Span guard rides along for free: tokenize_with_spans raises loudly on
    # any inexact bare-format char->token alignment (V5.38 rules) — proven on
    # the real tokenizer here, before any trainer loads the 1M-row file.
    texts = rows["full_text"].tolist()
    char_spans = list(
        zip(rows["answer_char_start"].astype(int), rows["answer_char_end"].astype(int))
    )
    examples = tokenize_with_spans(texts, char_spans, tokenizer, append_eos=True)
    prompt_ids = [ex.input_ids[: ex.label_span[0]] for ex in examples]
    print(f"[premise] span alignment: PASS ({len(examples)} bare rows tokenized exactly)")

    print(f"[premise] loading {args.base_model} (base, no adapter) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=DTYPES[args.dtype])
    model.to(args.device)

    completions = greedy_completions(
        model, tokenizer, prompt_ids, device=args.device, batch_size=args.batch_size
    )
    answers = rows["true_answer"].astype(int).tolist()
    em = sum(
        parse_answer("Answer:" + c) == a for c, a in zip(completions, answers)
    ) / len(completions)
    fmt = sum(format_valid("Answer:" + c) for c in completions) / len(completions)

    checks = [("exact_match", em, args.max_em)]
    if args.max_format_validity is not None:
        checks.append(("format_validity", fmt, args.max_format_validity))
    verdict = "PASS" if all(value <= bar for _, value, bar in checks) else "FAIL"
    bars = ", ".join(f"{name} <= {bar}" for name, _, bar in checks)
    print(
        f"[premise] base zero-shot on BARE prompts: exact_match {em:.4f} "
        f"format_validity {fmt:.4f} on n={args.n} (bars: {bars}) -> {verdict}"
    )
    for c in completions[:3]:
        print(f"[premise]   sample completion: {c[:80]!r}")

    if args.step0_json is not None:
        # The FIXED-COST control (ts38 guard 5). Same evaluator, mask and slice
        # gates.py g5 uses for its shared-set test loss, so this is on the same
        # footing as every run's floor rather than a parallel definition.
        task_format = TaskFormat(
            name=cfg["task"]["name"], format_version=cfg["task"]["format_version"]
        )
        step0_nats = evaluate_sft_nll_nats(
            model, examples, task_format, batch_size=args.batch_size, device=args.device
        )
        print(
            f"[premise] step-0 mean LABEL-token loss {step0_nats:.4f} nats on the same "
            f"n={args.n} slice (fixed-cost control)"
        )
        args.step0_json.parent.mkdir(parents=True, exist_ok=True)
        args.step0_json.write_text(
            json.dumps(
                {
                    "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    "base_model": str(args.base_model),
                    "dtype": args.dtype,
                    "device": args.device,
                    "step0_label_loss_nats": step0_nats,
                    "exact_match": em,
                    "format_validity": fmt,
                    "n": args.n,
                    "rows": [stop, end],
                    "eval_config": str(args.eval_config),
                    "eval_file": cfg["data"]["file"],
                    "eval_order_hash": cfg["data"]["order_hash"],
                    "task": dict(cfg["task"]),
                    "tokenizer": {
                        "path": str(tok_path),
                        "sha256": tokenizer_hash(tokenizer),
                    },
                    "protocol": (
                        "mean masked LABEL-token cross-entropy (nats/label-token, "
                        "evaluate_sft_nll_nats) of the untrained base over a FIXED slice "
                        "of the eval file's reporting block — the n-independent cost floor "
                        "an EDL/D curve must be read against"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"[premise] step-0 control written to {args.step0_json}")

    if verdict == "FAIL":
        print(
            "[premise] the scaffold-free premise does NOT hold — the base answers "
            "bare prompts, so the format transient this family exists to measure is "
            "absent. Halt for owner triage; do not launch the sweep."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
