"""Format-transfer battery for the ts1b op-install premise (owner 2026-08-28).

The op-installed TS-1B knows add/sub in operator notation. The elicit-arm
premise is that this capability is LATENT for other surface forms. This
script measures how far it transfers, with zero training: every probe below
is rendered from the same frozen eval triples (D_algo_eval, question-disjoint
from all training prefixes) and evaluated by greedy decoding, at 0 and
k-shot, with an error taxonomy on the wrong answers.

Probe formats (installed -> target, decreasing surface overlap with install):
    bare_op      "23 + 45 = "                                   (install sanity)
    hybrid_op    "What is 23 + 45?\\n"                           (NL frame, op body)
    bridge       "What is the sum of 23 and 45?\\n23 + 45 = "    (NL + op restatement)
    scaffold_op  "Question: 23 + 45\\nAnswer: "                  (frozen scaffold, op body)
    bare_nl      "What is the sum of 23 and 45?\\n"              (the TARGET form)
    scaffold_nl  "Question: What is the sum of 23 and 45?\\nAnswer: "

Per (format, shots): exact-match, format validity, first-answer-token
logit-diff vs a same-format distractor answer (sensitive even when EM ~0),
error taxonomy over parsed-but-wrong outputs (wrong-operation / swapped
operands / operand copy / other — "correct math, wrong reading" vs noise),
and sample completions.

Reading the result: high bare_op + ~0 everywhere else = capability is
format-BOUND (not latent for NL; premise fails). Graded decay with taxonomy
dominated by wrong-operation/swap = the arithmetic runs and only the NL
binding is missing (latent behind a thin interface; premise plausible).

Pure inference (no grads). GPU recommended, box-only.

Usage:
    python3 premise_checks.py (--run-id <rid> | --model <id-or-path>) \
        [--n 256] [--shots 0 16] [--eval-parquet .../D_algo_eval.parquet]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from geode.arith.evals import exact_match_accuracy, format_valid  # noqa: E402
from geode.arith.formats import _NL_PHRASE, true_answer  # noqa: E402

EVAL_PARQUET = REPO_ROOT / "experiments/training-run/data/full/D_algo_eval.parquet"
# Skip the reporting block other tools consume (circuit pairs, G5 slices).
DEFAULT_ROW_OFFSET = 60_000


def nl_body(a: int, b: int, op: str) -> str:
    return _NL_PHRASE[op].format(a=a, b=b)


# name -> (prompt, full_exemplar); answer is always str(true_answer).
def render_probe(fmt: str, a: int, b: int, op: str, ans: int) -> tuple[str, str]:
    if fmt == "bare_op":
        p = f"{a} {op} {b} = "
    elif fmt == "hybrid_op":
        p = f"What is {a} {op} {b}?\n"
    elif fmt == "bridge":
        p = f"{nl_body(a, b, op)}\n{a} {op} {b} = "
    elif fmt == "scaffold_op":
        p = f"Question: {a} {op} {b}\nAnswer: "
    elif fmt == "bare_nl":
        p = f"{nl_body(a, b, op)}\n"
    elif fmt == "scaffold_nl":
        p = f"Question: {nl_body(a, b, op)}\nAnswer: "
    else:
        raise ValueError(f"unknown probe format {fmt!r}")
    return p, p + str(ans)


FORMATS = ("bare_op", "hybrid_op", "bridge", "scaffold_op", "bare_nl", "scaffold_nl")


def classify_error(pred: int, a: int, b: int, op: str) -> str:
    """Name the misreading a wrong-but-parsed answer corresponds to, if any."""
    if pred == true_answer(a, b, op):
        return "correct"
    candidates = {
        "wrong_op_add": a + b,
        "wrong_op_sub": a - b,
        "wrong_op_mult": a * b,
        "swapped_sub": b - a,
        "copy_a": a,
        "copy_b": b,
    }
    del candidates["wrong_op_add" if op == "+" else "wrong_op_sub" if op == "-" else "wrong_op_mult"]
    for name, val in candidates.items():
        if pred == val:
            return name
    return "other"


def parse_int(completion: str) -> int | None:
    from geode.arith.evals import parse_answer

    return parse_answer("Answer:" + completion)


@torch.no_grad()
def first_token_logit_diff(model, tokenizer, prompts, answers, distractors, device, batch_size):
    """Mean logit(correct first answer token) - logit(distractor's), at the
    final prompt position. Sensitive when EM is ~0; skips pairs whose first
    tokens coincide (the mapping tools' degeneracy rule)."""
    tokenizer.padding_side = "left"
    diffs = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s : s + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits[:, -1].float()
        for i, (ans, dis) in enumerate(
            zip(answers[s : s + batch_size], distractors[s : s + batch_size])
        ):
            ct = tokenizer(str(ans), add_special_tokens=False)["input_ids"][0]
            xt = tokenizer(str(dis), add_special_tokens=False)["input_ids"][0]
            if ct != xt:
                diffs.append((logits[i, ct] - logits[i, xt]).item())
    return sum(diffs) / max(1, len(diffs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None, help="zoo run (plain full-FT checkpoint)")
    ap.add_argument("--model", default=None, help="checkpoint dir or hub id")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--shots", type=int, nargs="+", default=[0, 16])
    ap.add_argument("--formats", nargs="+", default=list(FORMATS), choices=FORMATS)
    ap.add_argument("--eval-parquet", type=Path, default=EVAL_PARQUET)
    ap.add_argument("--row-offset", type=int, default=DEFAULT_ROW_OFFSET)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="premise_checks.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if (args.model is None) == (args.run_id is None):
        raise SystemExit("[premise] pass exactly one of --model / --run-id")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.run_id is not None:
        from geode.zoo import load_model as zoo_load_model

        store = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
        model = zoo_load_model(args.run_id, store=store, device=args.device)
        name = args.run_id
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        model.to(args.device)
        name = args.model
    model.eval()

    df = pd.read_parquet(args.eval_parquet)
    max_shots = max(args.shots)
    rows = df.iloc[args.row_offset : args.row_offset + max_shots + args.n]
    triples = [(int(r.a), int(r.b), str(r.op)) for r in rows.itertuples()]
    shot_triples, query_triples = triples[:max_shots], triples[max_shots:]
    answers = [true_answer(a, b, op) for a, b, op in query_triples]
    distractors = answers[1:] + answers[:1]  # same distribution, wrong row

    results: dict[str, dict] = {}
    for fmt in args.formats:
        for k in args.shots:
            exemplars = [
                render_probe(fmt, a, b, op, true_answer(a, b, op))[1]
                for a, b, op in shot_triples[:k]
            ]
            prompts = [
                "\n\n".join([*exemplars, render_probe(fmt, a, b, op, 0)[0]])
                for a, b, op in query_triples
            ]
            prompt_ids = [
                tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts
            ]
            em, comps = exact_match_accuracy(
                model, tokenizer, prompt_ids, answers,
                device=args.device, batch_size=args.batch_size,
            )
            fv = sum(format_valid("Answer:" + c) for c in comps) / len(comps)
            ld = first_token_logit_diff(
                model, tokenizer, prompts, answers, distractors, args.device, args.batch_size
            )
            taxonomy = Counter()
            for c, (a, b, op) in zip(comps, query_triples):
                pred = parse_int(c)
                if pred is None:
                    taxonomy["unparsed"] += 1
                else:
                    taxonomy[classify_error(pred, a, b, op)] += 1
            tax = dict(taxonomy.most_common())
            results[f"{fmt}@{k}shot"] = {
                "exact_match": em, "format_validity": fv, "logit_diff": ld,
                "taxonomy": tax, "samples": comps[:3],
            }
            print(f"[premise] {fmt:<12} {k:>2}-shot: EM {em:.4f}  fmt {fv:.4f}  "
                  f"logit_diff {ld:+7.2f}  {tax}")
            for c in comps[:2]:
                print(f"[premise]     sample: {c!r}")

    Path(args.out).write_text(json.dumps({"model": name, "n": args.n, "results": results}, indent=2))
    print(f"[premise] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
