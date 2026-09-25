"""Few-shot failure diagnosis (paper Table 11 discrepancy, owner 2026-08-29).

The paper's op-pre-taught TinyStories-1B scores 11.9% on NL add/sub at 16
shots (App. I.9); ours scores 0.000. This script finds out where the gap
lives by sweeping the few-shot construction and dissecting the errors:

- shot count k in {0, 2, 4, 8, 16, 32} — interference may grow with k
- exemplar separator: blank line ("\\n\\n", our G5 convention) vs single
  newline ("\\n", the denser sheet the paper plausibly used)
- per-cell error anatomy: EM, format validity, EXEMPLAR-COPY rate (is the
  output literally one of the in-context answers? — the smoking gun for
  interference), median relative error |pred-true|/|true|, and samples

on both the installed surface (bare_op) and the target (bare_nl).
If some (k, sep) cell reaches ~0.1 on bare_nl, our 0.000 was a prompt-
construction artifact and the paper's Table-11 row replicates; if every
cell is 0 with high exemplar-copy rates, the collapse is real interference
and the discrepancy is substantive (install recipe or eval difference).

Usage:
    python3 fewshot_diag.py --run-id evt-ts1b-op-install [--n 128]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from geode.arith.evals import exact_match_accuracy, format_valid  # noqa: E402
from geode.arith.formats import true_answer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from premise_checks import (  # noqa: E402
    DEFAULT_ROW_OFFSET,
    EVAL_PARQUET,
    parse_int,
    render_probe,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 2, 4, 8, 16, 32])
    ap.add_argument("--formats", nargs="+", default=["bare_op", "bare_nl"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if (args.model is None) == (args.run_id is None):
        raise SystemExit("[diag] pass exactly one of --model / --run-id")

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

    df = pd.read_parquet(EVAL_PARQUET)
    max_k = max(args.ks)
    rows = df.iloc[DEFAULT_ROW_OFFSET : DEFAULT_ROW_OFFSET + max_k + args.n]
    triples = [(int(r.a), int(r.b), str(r.op)) for r in rows.itertuples()]
    shot_triples, query_triples = triples[:max_k], triples[max_k:]
    answers = [true_answer(a, b, op) for a, b, op in query_triples]

    results = {}
    for fmt in args.formats:
        exemplars_all = [
            render_probe(fmt, a, b, op, true_answer(a, b, op))[1] for a, b, op in shot_triples
        ]
        shot_answers = {str(true_answer(a, b, op)) for a, b, op in shot_triples}
        for k in args.ks:
            for sep_name, sep in (("blank", "\n\n"), ("newline", "\n")):
                if k == 0 and sep_name == "newline":
                    continue  # identical to blank at k=0
                prompts = [
                    sep.join([*exemplars_all[:k], render_probe(fmt, a, b, op, 0)[0]])
                    for a, b, op in query_triples
                ]
                ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
                em, comps = exact_match_accuracy(
                    model, tokenizer, ids, answers,
                    device=args.device, batch_size=args.batch_size,
                )
                fv = n_copy = 0
                rel_errs = []
                for c, ans in zip(comps, answers):
                    fv += format_valid("Answer:" + c)
                    pred = parse_int(c)
                    if pred is not None:
                        if str(pred) in shot_answers and k > 0:
                            n_copy += 1
                        if pred != ans and ans != 0:
                            rel_errs.append(abs(pred - ans) / abs(ans))
                rel_errs.sort()
                med = rel_errs[len(rel_errs) // 2] if rel_errs else float("nan")
                key = f"{fmt}@k{k}/{sep_name}"
                results[key] = {
                    "em": em, "fmt": fv / len(comps),
                    "exemplar_copy": n_copy / len(comps), "median_rel_err": med,
                    "samples": comps[:2],
                }
                print(f"[diag] {fmt:<8} k={k:<3} sep={sep_name:<7}: "
                      f"EM {em:.4f}  fmt {fv / len(comps):.4f}  "
                      f"exemplar_copy {n_copy / len(comps):.4f}  "
                      f"med_rel_err {med:8.3f}  e.g. {comps[0]!r}")

    out = args.out or f"fewshot_diag_{name.replace('/', '_')}.json"
    Path(out).write_text(json.dumps({"model": name, "results": results}, indent=2))
    print(f"[diag] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
