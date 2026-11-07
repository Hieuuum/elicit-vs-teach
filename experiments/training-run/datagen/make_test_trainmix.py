"""Build ``D_algo_eval_bare_trainmix_n20000.parquet``: the ts1b redesign's test
set, reshaped so its problem-size mix equals the training set's (owner,
2026-11-06).

Why. ``D_algo_eval_bare`` (100K word add/sub test problems) spreads 10% over
each of 10 operand-length cells, but the children's training set
``D_algo_bare_fresh4m`` holds 1.43% in each of 1x4, 2x3, 3x2, 4x1 and 15.7% in
each of 2x4, 3x3, 3x4, 4x2, 4x3, 4x4 (those four cells are nearly used up by
the symbol install and the test set, so training cannot move toward 10%).
So the test set moves: a seeded sample of ``D_algo_eval_bare`` with
per-(cell, op) counts proportional to the training set's (largest-remainder
rounding). Every row is an unchanged test row; none is in training.

Deterministic. Writes ``<stem>.report.json`` beside the parquet.

Usage:
    python3 make_test_trainmix.py --out ../data/full [--n 20000] [--seed 20261106]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from geode.arith import order_hash  # noqa: E402

TRAIN_STEM = "D_algo_bare_fresh4m"
TRAIN_PIN = "32dd6f241a9e10a9c42ed56a13fc4fdcf8e0da15a28f80a561eb57dc9d2e96d5"
TEST_STEM = "D_algo_eval_bare"
TEST_PIN = "e419baa213bbe07dfeb50f46fe17b464056cd18c2a7302238a66682d7c594631"
Stratum = tuple[int, int, str]  # (x_digits, y_digits, op)


def strata_counts(df: pd.DataFrame) -> Counter:
    """Row count per (x_digits, y_digits, op)."""
    return Counter(
        zip(df["x_digits"].astype(int).tolist(), df["y_digits"].astype(int).tolist(), df["op"])
    )


def allocate(weights: dict, n: int) -> dict:
    """Split ``n`` over the keys of ``weights`` proportionally, largest-remainder
    rounding (ties broken by key order). Sums to exactly ``n``; zero weight gets 0."""
    total = sum(weights.values())
    if total <= 0 or n < 0:
        raise ValueError("allocate: need positive total weight and n >= 0")
    keys = sorted(weights)
    exact = {k: n * weights[k] / total for k in keys}
    out = {k: int(exact[k]) for k in keys}
    by_rem = sorted(keys, key=lambda k: (-(exact[k] - out[k]), keys.index(k)))
    for k in by_rem[: n - sum(out.values())]:
        out[k] += 1
    return out


def sample_strata(df: pd.DataFrame, alloc: dict, seed: int) -> pd.DataFrame:
    """Seeded sample of ``alloc[s]`` rows from each stratum ``s`` of ``df``,
    then one seeded shuffle of the result. Raises if a stratum is short."""
    rng = random.Random(seed)
    keys = list(
        zip(df["x_digits"].astype(int).tolist(), df["y_digits"].astype(int).tolist(), df["op"])
    )
    pools: dict = {}
    for i, k in enumerate(keys):
        pools.setdefault(k, []).append(i)
    picked: list[int] = []
    for s in sorted(alloc):
        pool = pools.get(s, [])
        if alloc[s] > len(pool):
            raise ValueError(f"sample_strata: stratum {s} has {len(pool)} rows, need {alloc[s]}")
        picked += rng.sample(pool, alloc[s])
    rng.shuffle(picked)
    return df.iloc[picked]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True, help="dir holding the frozen parquets")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()

    train = pd.read_parquet(args.out / f"{TRAIN_STEM}.parquet")
    if order_hash(train.to_dict("records")) != TRAIN_PIN:
        raise SystemExit(f"{TRAIN_STEM}.parquet does not match its pin; refusing")
    test = pd.read_parquet(args.out / f"{TEST_STEM}.parquet")
    if order_hash(test.to_dict("records")) != TEST_PIN:
        raise SystemExit(f"{TEST_STEM}.parquet does not match its pin; refusing")

    train_mix = strata_counts(train)
    alloc = allocate(dict(train_mix), args.n)
    out = sample_strata(test, alloc, args.seed).copy()
    out["src_idx"] = out["idx"]
    out["idx"] = range(len(out))
    stem = f"{TEST_STEM}_trainmix_n{args.n}"
    out["dataset"] = stem
    out = out.reset_index(drop=True)

    t_train = set(zip(train["a"].astype(int), train["op"], train["b"].astype(int)))
    t_out = list(zip(out["a"].astype(int), out["op"], out["b"].astype(int)))
    got = strata_counts(out)
    checks = {
        "n_rows": len(out) == args.n,
        "all_unique": len(set(t_out)) == len(t_out),
        "disjoint_train": not (set(t_out) & t_train),
        "rows_are_test_rows": set(out["src_idx"]).issubset(set(test["idx"])),
        "counts_match_allocation": dict(got) == {k: v for k, v in alloc.items() if v},
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise SystemExit(f"[trainmix] checks FAILED: {failed}")

    path = args.out / f"{stem}.parquet"
    out.to_parquet(path, index=False)
    pin = order_hash(out.to_dict("records"))
    n_train = sum(train_mix.values())
    cells_train = Counter()
    cells_test = Counter()
    for (x, y, _), c in train_mix.items():
        cells_train[f"{x}x{y}"] += c / n_train
    for (x, y, _), c in got.items():
        cells_test[f"{x}x{y}"] += c / args.n
    rep = {
        "final": path.name,
        "n_rows": len(out),
        "order_hash": pin,
        "seed": args.seed,
        "from": {
            "test": TEST_STEM,
            "test_pin": TEST_PIN,
            "mix_of": TRAIN_STEM,
            "mix_pin": TRAIN_PIN,
        },
        "allocation": {f"{x}x{y}{op}": c for (x, y, op), c in sorted(alloc.items())},
        "cell_share": {
            k: {"train": round(cells_train[k], 5), "test": round(cells_test[k], 5)}
            for k in sorted(cells_train)
        },
        "checks": checks,
    }
    (args.out / f"{stem}.report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"[trainmix] wrote {path}: {len(out):,} rows, order_hash={pin}")
    for k, v in rep["cell_share"].items():
        print(f"  {k}: train {v['train']:.4f}  test {v['test']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
