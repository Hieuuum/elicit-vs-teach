"""Task-vector scaling: install the bridge binding WITHOUT breaking the install.

The full bridge dose (owner 2026-08-29) installed the NL->op binding but
damaged the op skill (bare_op EM 0.69 -> 0.24) — classic interference from
full-FT to ~zero loss on a narrow dose. Zero-retraining fix: both checkpoints
exist, so sweep the interpolation

    W(alpha) = W_op + alpha * (W_bridge - W_op)

and measure, per alpha, the three-way frontier:

    retention    bare_op 0-shot EM        (the installed capability)
    binding      rewrite_exact            (bare_nl -> the model's own 'a op b')
    composition  chain EM                 (its rewrite + ' = ' -> exact answer)

alpha=0 is the pure install, alpha=1 the damaged bridge; the useful patch is
wherever binding appears before retention falls. Pure inference per alpha.

Usage:
    python3 task_vector_scale.py --base-run evt-ts1b-op-install \
        --tuned-run evt-ts1b-op-bridge [--alphas 0 0.25 0.5 0.75 1.0] [--n 128]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from geode.arith.decode import greedy_completions  # noqa: E402
from geode.arith.evals import exact_match_accuracy  # noqa: E402
from geode.arith.formats import true_answer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from premise_checks import DEFAULT_ROW_OFFSET, EVAL_PARQUET, render_probe  # noqa: E402

EXPR_RE = re.compile(r"(-?\d+)\s*([+*-])\s*(-?\d+)")


def load_state(run_id: str, store: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(store / "runs" / run_id / "model" / "model.safetensors")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-run", default="evt-ts1b-op-install")
    ap.add_argument("--tuned-run", default="evt-ts1b-op-bridge")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    store = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))

    from geode.zoo import load_model as zoo_load_model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = zoo_load_model(args.base_run, store=store, device=args.device)
    model.eval()

    base_sd = load_state(args.base_run, store)
    tuned_sd = load_state(args.tuned_run, store)
    # per-parameter fp32 delta, kept on CPU; applied as base + alpha*delta
    deltas = {k: (tuned_sd[k].float() - v.float()) for k, v in base_sd.items()}
    del tuned_sd
    print(f"[tvs] delta over {len(deltas)} tensors "
          f"(mean |d| {sum(d.abs().mean().item() for d in deltas.values()) / len(deltas):.2e})")

    df = pd.read_parquet(EVAL_PARQUET)
    rows = df.iloc[DEFAULT_ROW_OFFSET : DEFAULT_ROW_OFFSET + args.n]
    triples = [(int(r.a), int(r.b), str(r.op)) for r in rows.itertuples()]
    answers = [true_answer(a, b, op) for a, b, op in triples]

    def ids_for(fmt: str) -> list[list[int]]:
        return [
            tokenizer(render_probe(fmt, a, b, op, 0)[0], add_special_tokens=False)["input_ids"]
            for a, b, op in triples
        ]

    op_ids, nl_ids = ids_for("bare_op"), ids_for("bare_nl")
    params = dict(model.named_parameters())
    missing = [k for k in deltas if k not in params]
    if missing:  # tied weights (e.g. lm_head <- embed_tokens) live only in the file
        print(f"[tvs] skipping {len(missing)} file-only tensors (tied): {missing[:3]}")
        for k in missing:
            deltas.pop(k)
    results = []
    for alpha in args.alphas:
        with torch.no_grad():
            for k, d in deltas.items():
                w = base_sd[k].float() + alpha * d
                params[k].copy_(w.to(params[k].dtype))

        retention, _ = exact_match_accuracy(
            model, tokenizer, op_ids, answers, device=args.device, batch_size=args.batch_size
        )
        rewrites = greedy_completions(
            model, tokenizer, nl_ids, device=args.device, batch_size=args.batch_size
        )
        n_exact = 0
        chain_ids = []
        for (a, b, op), rw in zip(triples, rewrites):
            line = rw.strip().splitlines()[0] if rw.strip() else ""
            m = EXPR_RE.search(line)
            if m and (int(m.group(1)), m.group(2), int(m.group(3))) == (a, op, b):
                n_exact += 1
            chain_ids.append(
                tokenizer(f"{m.group(0) if m else line[:24]} = ",
                          add_special_tokens=False)["input_ids"]
            )
        chain_em, comps = exact_match_accuracy(
            model, tokenizer, chain_ids, answers, device=args.device, batch_size=args.batch_size
        )
        tax = Counter()
        for c, (a, b, op) in zip(comps, triples):
            from premise_checks import classify_error, parse_int

            pred = parse_int(c)
            tax["unparsed" if pred is None else classify_error(pred, a, b, op)] += 1
        row = {"alpha": alpha, "retention_op_em": retention,
               "rewrite_exact": n_exact / len(triples), "chain_em": chain_em}
        results.append(row)
        print(f"[tvs] alpha {alpha:5.3f}: op_EM {retention:.4f}  "
              f"rewrite_exact {n_exact / len(triples):.4f}  chain_EM {chain_em:.4f}  "
              f"{dict(tax.most_common(4))}")

    out = args.out or f"tvs_{args.tuned_run}.json"
    Path(out).write_text(json.dumps(
        {"base": args.base_run, "tuned": args.tuned_run, "n": args.n,
         "results": results}, indent=2))
    print(f"[tvs] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
