"""Fig2nl3 premise guard: base Llama must be ~0% zero-shot on BARE prompts.

The fig2nl3 family exists on one premise (EXPERIMENTS §6.13): without the
Question:/Answer: scaffold, base Llama treats a math question as text to
continue rather than a question to answer — the paper's stated regime ("these
base models achieve 0% zero-shot accuracy"), and the precondition for the
Figure-2 pre-elicit transient to exist at all. The scaffolded families
measured base at ~0.31 zero-shot EM / ~0.83 format validity, which is exactly
why their arms coincided (§6.12 outcome). This guard MEASURES the premise on
the actual bare eval file BEFORE any training happens; if base Llama still
answers bare prompts, the family is pointless and must halt for owner triage,
not spend a GPU-day confirming nothing.

Protocol: greedy EOS-stopped completions (the shared G4/G5 decode path) on a
fixed slice of D_algo_eval_bare's REPORTING block (rows
[EVAL_STOP_ROWS : EVAL_STOP_ROWS+n], the same rows G4 scores, hash-verified
through the eval config's pin), scored with the shared format_valid /
exact_match helpers. Tokenizing the slice through tokenize_with_spans also
proves the bare format's char->token span alignment on the real tokenizer
before any trainer touches the file.

PASS: zero-shot EM <= --max-em (default 0.05) — the premise holds.
FAIL: exit 1 with both numbers printed — do NOT launch the sweep.

Box-only (gated hub model, GPU recommended). Read-only: loads the base model,
generates, prints; writes nothing to any store or manifest.

Usage:
    python3 check_bare_baseline.py [--n 256] [--max-em 0.05] [--device cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from geode.arith import format_valid, parse_answer, tokenize_with_spans
from geode.arith.decode import greedy_completions
from geode.edl import EVAL_STOP_ROWS
from train import load_config  # scripts/ shared config loader (adds merge + pins)
from train_sft import load_frozen_parquet

EVAL_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "eval_bare_target_data_llama.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=256, help="prompts scored (fixed slice, no sampling)")
    ap.add_argument("--max-em", type=float, default=0.05, help="premise bar: base EM must be <= this")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B",
        help="model under test: a hub id or a local checkpoint dir (e.g. the "
        "ts1b pretrain's runs/evt-ts1b-base/model for the fig2ts premise)",
    )
    args = ap.parse_args()

    cfg = load_config(EVAL_CONFIG, None)
    df = load_frozen_parquet(cfg)  # hash-verified against the config pin
    stop, end = EVAL_STOP_ROWS, EVAL_STOP_ROWS + args.n
    if len(df) < end:
        raise SystemExit(f"[premise] --n {args.n} needs {end} rows, eval file has {len(df)}")
    rows = df.iloc[stop:end]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["path"])
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

    print(f"[premise] loading {args.model} (no adapter) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to(args.device)

    completions = greedy_completions(
        model, tokenizer, prompt_ids, device=args.device, batch_size=args.batch_size
    )
    answers = rows["true_answer"].astype(int).tolist()
    em = sum(
        parse_answer("Answer:" + c) == a for c, a in zip(completions, answers)
    ) / len(completions)
    fmt = sum(format_valid("Answer:" + c) for c in completions) / len(completions)

    verdict = "PASS" if em <= args.max_em else "FAIL"
    print(
        f"[premise] base zero-shot on BARE prompts: exact_match {em:.4f} "
        f"format_validity {fmt:.4f} on n={args.n} (bar: EM <= {args.max_em}) -> {verdict}"
    )
    for c in completions[:3]:
        print(f"[premise]   sample completion: {c[:80]!r}")
    if verdict == "FAIL":
        print(
            "[premise] the scaffold-free premise does NOT hold — base Llama answers "
            "bare prompts, so the format transient this family exists to measure is "
            "absent. Halt for owner triage; do not launch the sweep."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
