"""Context-augmented op-install sets (Table-11 replication track).

fewshot_diag measured that the single-example-per-sequence install collapses
under ANY in-context example (bare_op EM 0.72 -> 0.00 at k=2): every training
sequence held one example at position 0, so prepended context is maximally
OOD. The paper's intervention is pre-training-style (packed sequences), so
for their model a few-shot prompt IS the training distribution.

Two derivations from the frozen ``D_algo_op`` (each source row used at most
once per set, frozen order, eval disjointness inherited):

- ``D_algo_op_ctx``  — k uniform 0..16 exemplars above each query, loss span
  on the final answer. Outcome (2026-08-31, evt-ts1b-op-install2): CURED the
  few-shot collapse (EM stable across k) but EM fell 0.73 -> 0.19 with
  median relative error ~0.02 — robust but approximate; only ~580K
  supervised answers reached the loss (vs 3.5M in the plain install).
- ``D_algo_op_ctx2`` — 50% k=0 single examples (exactness supervision) +
  50% k uniform 1..16 (robustness), ~1.7x denser answer supervision.

    907 + 4 = 911\n23 - 45 = -22\n9881 + 38 = 9919
                                            ^^^^ label span

Usage:
    python3 make_op_ctx.py --out ../data/full
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from geode.arith import order_hash

ALGO_OP_PIN = "d92600148fb9b3b3f3637f1afe14dac04053c2fc9154c8c7b05808d89a4757bb"
SEED = 20260717
K_MAX = 16


def build(name: str, rows: list[dict], out: Path) -> str:
    rng = random.Random(f"{SEED}:{'op_ctx' if name == 'D_algo_op_ctx' else name}")
    records = []
    i = 0
    while i < len(rows):
        if name == "D_algo_op_ctx2":
            k = 0 if rng.random() < 0.5 else rng.randint(1, K_MAX)
        else:
            k = rng.randint(0, K_MAX)
        if i + k + 1 > len(rows):
            break
        exemplars, query = rows[i : i + k], rows[i + k]
        i += k + 1
        prefix = "".join(r["full_text"] + "\n" for r in exemplars)
        cs = len(prefix) + int(query["answer_char_start"])
        ce = len(prefix) + int(query["answer_char_end"])
        full = prefix + query["full_text"]
        records.append(
            {
                "idx": len(records),
                "dataset": name,
                "a": int(query["a"]),
                "b": int(query["b"]),
                "op": str(query["op"]),
                "x_digits": int(query["x_digits"]),
                "y_digits": int(query["y_digits"]),
                "format": "bare_op_ctx",
                "label_mode": "correct",
                "shown_answer": int(query["shown_answer"]),
                "k_shots": k,
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
    ks = pd.Series([r["k_shots"] for r in records])
    print(f"[evt] wrote {path}  n={len(records):,}  order_hash={h}")
    print(f"[evt]   k distribution: mean {ks.mean():.2f}, k=0 share {(ks == 0).mean():.3f}")
    print(f"[evt]   row 0 (k={records[0]['k_shots']}): {records[0]['full_text'][:80]!r}...")
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src = pd.read_parquet(args.out / "D_algo_op.parquet")
    if order_hash(src.to_dict("records")) != ALGO_OP_PIN:
        raise SystemExit("D_algo_op.parquet does not match its frozen pin; refusing")
    rows = src.to_dict("records")
    build("D_algo_op_ctx", rows, args.out)
    build("D_algo_op_ctx2", rows, args.out)
    print("[evt] pin the order_hashes in ts1b_op_install{2,3}.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
