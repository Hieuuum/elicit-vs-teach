"""Answer-free word<->symbol translation doses (ts1b op-install, 2026-08-28).

The lexical-binding hypothesis: TinyStories pretraining never taught what
"sum"/"difference" MEAN, so the op-installed arithmetic is unreachable from
NL phrasings — the missing piece is the word->operation binding, not the
capability. These doses teach exactly that binding and nothing else: bare
operator<->NL REWRITING in both directions, never showing a computed result,
so they can leak no arithmetic (the phase-3 bridge design, re-rendered bare
and extended to subtraction; the frozen scaffolded ``render_translate`` is
untouched).

    Rewrite in operator notation: What is the sum of 23 and 45?\\n23 + 45
    Rewrite in words: 23 - 45\\nWhat is the difference between 23 and 45?

Two artifacts:
- ``D_translate_dose``      — ops + and -, both directions   (the treatment)
- ``D_translate_dose_add``  — op + only, both directions     (the word-specificity
  control: if training it unlocks NL *addition* while NL *subtraction* stays
  dead, the binding is per-word — the hypothesis in its sharpest form)

Operand pairs are 1-4 digit, positive, distinct, and question-disjoint from
the frozen eval artifact (every ordered (a, b) appearing in D_algo_eval is
excluded, any op), so the doses can never pre-teach an eval question.
Deterministic: seeded RNG, no time or machine dependence.

Usage:
    python3 make_translate_dose.py --out ../data/full
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from geode.arith import order_hash
from geode.arith.formats import _NL_PHRASE, digits

EVAL_PIN = "5e422dafc7330a050a483002172e23b262c180ba4254b5d97e428506e6892fb3"
SEED = 20260717
PAIRS_PER_OP = 2048  # x2 directions per pair


def bare_translate(a: int, b: int, op: str, direction: str) -> tuple[str, tuple[int, int]]:
    """Bare (scaffold-free) rewrite example; answer span after the newline."""
    nl = _NL_PHRASE[op].format(a=a, b=b)
    sym = f"{a} {op} {b}"
    if direction == "to_op":
        prompt, answer = f"Rewrite in operator notation: {nl}\n", sym
    elif direction == "to_nl":
        prompt, answer = f"Rewrite in words: {sym}\n", nl
    else:
        raise ValueError(f"unknown direction {direction!r}")
    full = prompt + answer
    return full, (len(prompt), len(full))


def sample_pairs(rng: random.Random, k: int, blocked: set[tuple[int, int]]) -> list:
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < k:
        a = rng.randint(1, 9999)
        b = rng.randint(1, 9999)
        if (a, b) in blocked or (a, b) in seen:
            continue
        seen.add((a, b))
        pairs.append((a, b))
    return pairs


def build(name: str, ops: tuple[str, ...], blocked: set, out: Path) -> str:
    rng = random.Random(f"{SEED}:{name}")
    records = []
    for op in ops:
        for a, b in sample_pairs(rng, PAIRS_PER_OP, blocked):
            for direction in ("to_op", "to_nl"):
                full, (cs, ce) = bare_translate(a, b, op, direction)
                records.append(
                    {
                        "idx": len(records),
                        "dataset": name,
                        "a": a,
                        "b": b,
                        "op": op,
                        "x_digits": digits(a),
                        "y_digits": digits(b),
                        "format": "bare_translate",
                        "label_mode": "correct",
                        # translate convention (geode.arith.validate): the
                        # answer TEXT, never a computed number
                        "shown_answer": full[cs:ce],
                        "direction": direction,
                        "prompt_text": full[:cs],
                        "answer_text": full[cs:ce],
                        "full_text": full,
                        "answer_char_start": cs,
                        "answer_char_end": ce,
                    }
                )
    path = out / f"{name}.parquet"
    pd.DataFrame(records).to_parquet(path, index=False)
    h = order_hash(records)
    print(f"[evt] wrote {path}  n={len(records):,}  order_hash={h}")
    print(f"[evt]   row 0: {records[0]['full_text']!r}")
    print(f"[evt]   row 1: {records[1]['full_text']!r}")
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="dir holding the frozen parquets")
    args = ap.parse_args()

    eval_df = pd.read_parquet(args.out / "D_algo_eval.parquet")
    if order_hash(eval_df.to_dict("records")) != EVAL_PIN:
        raise SystemExit("D_algo_eval.parquet does not match its frozen pin; refusing")
    blocked = {(int(r.a), int(r.b)) for r in eval_df.itertuples()}
    print(f"[evt] excluding {len(blocked):,} eval (a, b) pairs (question-disjointness)")

    build("D_translate_dose", ("+", "-"), blocked, args.out)
    build("D_translate_dose_add", ("+",), blocked, args.out)
    print("[evt] pin the two order_hashes in ts1b_op_bridge{,_add}.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
