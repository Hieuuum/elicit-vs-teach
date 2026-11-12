"""Build ``D_algo_bare_hard4m.parquet``: a 4M word add/sub set weighted toward
hard problems, for continuing the two matched ts1b children (owner, 2026-11-11).

Why. The children (``evt-ts1b-elicit-ft-fresh-b256-c500``,
``evt-ts1b-teach-ft-fresh-b256``) match overall (test EM 0.953 / 0.952) but
4x4 addition lags (0.83 / 0.91) and both are still improving. The owner chose
the "hard-heavy" mix below and to continue the current children on it.

Mix (share of rows per cell, split evenly between + and -):

    4x4 40% | 3x3, 3x4, 4x3 15% each | 2x4, 4x2 5% each | 1x4, 2x3, 3x2, 4x1 1.25% each

Blocked (never in this set):
    D_algo (symbol install) ∪ D_algo_eval ∪ probe (test sets) ∪ D_translate_mix
    (bridge) ∪ EAP-IG problems (both operand orders) ∪ a new reserve of 3x4
    addition pairs for future circuit folds (written to
    ``D_eapig_reserve_3x4.parquet``, both orders blocked here)
    and, in every cell except the four small ones, ``D_algo_bare_fresh4m``
    (what the children already trained on).

The small cells (1x4, 2x3, 3x2, 4x1) have only ~10K unseen problems each, fewer
than their 50K share: every unseen one is taken, the rest are repeats from
``D_algo_bare_fresh4m`` (owner accepted repeats there). The report counts them.

The current 20K test set is kept unchanged (owner); this set does not reshape it.

Deterministic. CPU only (no GPU, so no ``--confirm-cost``).

Usage:
    python3 make_algo_hard4m.py --out ../data/full [--n 4000000] [--seed 20261111]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from make_algo_fresh4m import (  # noqa: E402
    BRIDGE_PIN,
    BRIDGE_STEM,
    COLS,
    addition_pair_triples,
    load_eapig_metas,
    triples_of,
)
from make_data import DIGIT_BANDS, _frozen_triples, _record, _seed_int  # noqa: E402

from geode.arith import digits, order_hash, true_answer  # noqa: E402

Triple = tuple[int, str, int]
Cell = tuple[int, int]

FRESH_STEM = "D_algo_bare_fresh4m"
FRESH_PIN = "32dd6f241a9e10a9c42ed56a13fc4fdcf8e0da15a28f80a561eb57dc9d2e96d5"
FINAL_STEM = "D_algo_bare_hard4m"
RESERVE_STEM = "D_eapig_reserve_3x4"
RESERVE_CELL: Cell = (3, 4)
RESERVE_N = 50_000  # unordered 3x4 addition pairs; 10 folds need 15,360
OPS = ("+", "-")
MIX: dict[Cell, float] = {
    (4, 4): 0.40,
    (3, 3): 0.15,
    (3, 4): 0.15,
    (4, 3): 0.15,
    (2, 4): 0.05,
    (4, 2): 0.05,
    (1, 4): 0.0125,
    (2, 3): 0.0125,
    (3, 2): 0.0125,
    (4, 1): 0.0125,
}
SMALL_CELLS: frozenset[Cell] = frozenset({(1, 4), (2, 3), (3, 2), (4, 1)})


def allocation(n: int, mix: dict[Cell, float], ops: tuple[str, ...] = OPS) -> dict:
    """Integer rows per (cell, op), summing to exactly ``n``.

    Largest-remainder rounding over the (cell, op) shares; ties broken by
    sorted key so the result is deterministic.
    """
    if abs(sum(mix.values()) - 1.0) > 1e-9:
        raise ValueError(f"mix shares sum to {sum(mix.values())}, not 1")
    keys = sorted((c, op) for c in mix for op in ops)
    exact = {k: n * mix[k[0]] / len(ops) for k in keys}
    out = {k: int(exact[k]) for k in keys}
    short = n - sum(out.values())
    for k in sorted(keys, key=lambda k: (-(exact[k] - out[k]), k))[:short]:
        out[k] += 1
    return out


def sample_new(
    cell: Cell, k: int, blocked: set[tuple[int, int]], rng: random.Random
) -> list[tuple[int, int]]:
    """``k`` distinct (a, b) pairs from ``cell``, none in ``blocked`` (rejection
    sampling; only used where k is far below the free room)."""
    (lx, hx), (ly, hy) = DIGIT_BANDS[cell[0]], DIGIT_BANDS[cell[1]]
    free = (hx - lx + 1) * (hy - ly + 1) - sum(
        1 for a, b in blocked if lx <= a <= hx and ly <= b <= hy
    )
    if k > free:
        raise ValueError(f"cell {cell}: asked {k} > free {free}")
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    while len(out) < k:
        p = (rng.randint(lx, hx), rng.randint(ly, hy))
        if p in blocked or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def sample_small(
    cell: Cell,
    k: int,
    hard_blocked: set[tuple[int, int]],
    seen_before: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], int]:
    """``k`` distinct pairs from a small cell: every pair not in ``hard_blocked``
    and not ``seen_before`` first (enumerated), then repeats drawn from
    ``seen_before`` (still never ``hard_blocked``). Returns (pairs, n_repeats)."""
    (lx, hx), (ly, hy) = DIGIT_BANDS[cell[0]], DIGIT_BANDS[cell[1]]
    allowed = [
        (a, b) for a in range(lx, hx + 1) for b in range(ly, hy + 1) if (a, b) not in hard_blocked
    ]
    unseen = [p for p in allowed if p not in seen_before]
    repeats = [p for p in allowed if p in seen_before]
    if k > len(allowed):
        raise ValueError(f"cell {cell}: asked {k} > allowed {len(allowed)}")
    if k <= len(unseen):
        return rng.sample(unseen, k), 0
    extra = rng.sample(repeats, k - len(unseen))
    return unseen + extra, len(extra)


def reserve_pairs(
    n: int, cell: Cell, blocked_add: set[Triple], rng: random.Random
) -> list[tuple[int, int]]:
    """``n`` distinct addition pairs (a, b) from ``cell`` whose triple in
    neither operand order is in ``blocked_add``."""
    (lx, hx), (ly, hy) = DIGIT_BANDS[cell[0]], DIGIT_BANDS[cell[1]]
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    while len(out) < n:
        a, b = rng.randint(lx, hx), rng.randint(ly, hy)
        if (a, b) in seen or (a, "+", b) in blocked_add or (b, "+", a) in blocked_add:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out


def pairs_by_cell_op(triples: set[Triple]) -> dict[tuple[Cell, str], set[tuple[int, int]]]:
    out: dict[tuple[Cell, str], set[tuple[int, int]]] = {}
    for a, op, b in triples:
        out.setdefault(((digits(a), digits(b)), op), set()).add((a, b))
    return out


def build_triples(
    n: int,
    seed: int,
    hard_blocked: set[Triple],
    fresh: set[Triple],
    mix: dict[Cell, float] = MIX,
) -> tuple[list[Triple], dict, dict]:
    """Sample every (cell, op) block and shuffle. Returns (triples, alloc, repeats)."""
    alloc = allocation(n, mix)
    hard_by = pairs_by_cell_op(hard_blocked)
    fresh_by = pairs_by_cell_op(fresh)
    triples: list[Triple] = []
    repeats: dict = {}
    for (cell, op), k in sorted(alloc.items()):
        rng = random.Random(_seed_int(f"{FINAL_STEM}:{seed}:{cell[0]}x{cell[1]}:{op}"))
        hb = hard_by.get((cell, op), set())
        fb = fresh_by.get((cell, op), set())
        if cell in SMALL_CELLS:
            pairs, rep = sample_small(cell, k, hb, fb, rng)
        else:
            pairs, rep = sample_new(cell, k, hb | fb, rng), 0
        repeats[f"{cell[0]}x{cell[1]}{op}"] = rep
        triples.extend((a, op, b) for a, b in pairs)
    random.Random(_seed_int(f"{FINAL_STEM}-order:{seed}")).shuffle(triples)
    return triples, alloc, repeats


def write_parquet(triples: list[Triple], path: Path, name: str, chunk: int = 50_000) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    schema = None
    try:
        for start in range(0, len(triples), chunk):
            rows = [
                _record(start + i, a, b, op, true_answer(a, b, op), name, "bare_nl", "correct")
                for i, (a, op, b) in enumerate(triples[start : start + chunk])
            ]
            table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(path, schema)
            elif table.schema != schema:
                table = table.cast(schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True, help="dir holding the frozen parquets")
    ap.add_argument("--n", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=20261111)
    args = ap.parse_args()
    out: Path = args.out

    fresh_df = pd.read_parquet(out / f"{FRESH_STEM}.parquet", columns=COLS)
    if order_hash(fresh_df.to_dict("records")) != FRESH_PIN:
        raise SystemExit(f"{FRESH_STEM}.parquet does not match its pin; refusing")
    fresh = set(triples_of(fresh_df))
    del fresh_df
    bridge = pd.read_parquet(out / f"{BRIDGE_STEM}.parquet")
    if order_hash(bridge.to_dict("records")) != BRIDGE_PIN:
        raise SystemExit(f"{BRIDGE_STEM}.parquet does not match its pin; refusing")
    bridge_t = set(triples_of(bridge))
    report = json.loads((out / "report.json").read_text())
    algo = _frozen_triples(out, ("D_algo",), report)
    tests = _frozen_triples(out, ("D_algo_eval", "probe"), report)
    metas = load_eapig_metas()
    eapig_t = addition_pair_triples(metas)
    base_blocked = algo | tests | bridge_t | eapig_t

    # 1. reserve 3x4 addition pairs for future circuit folds (disjoint from
    #    everything above and from what the children trained on)
    res_rng = random.Random(_seed_int(f"{RESERVE_STEM}:{args.seed}"))
    reserve = reserve_pairs(RESERVE_N, RESERVE_CELL, base_blocked | fresh, res_rng)
    reserve_t = {(a, "+", b) for a, b in reserve} | {(b, "+", a) for a, b in reserve}
    res_df = pd.DataFrame({"a": [a for a, _ in reserve], "b": [b for _, b in reserve]})
    res_df["op"] = "+"
    res_path = out / f"{RESERVE_STEM}.parquet"
    res_df.to_parquet(res_path, index=False)
    res_pin = order_hash(
        [
            {
                "a": a,
                "b": b,
                "op": "+",
                "shown_answer": a + b,
                "format": "pair",
                "label_mode": "n/a",
            }
            for a, b in reserve
        ]
    )
    print(f"[hard] reserve: {len(reserve):,} 3x4 addition pairs -> {res_path.name} ({res_pin[:8]})")

    # 2. the training set
    hard_blocked = base_blocked | reserve_t
    print(
        f"[hard] blocked: D_algo {len(algo):,} | tests {len(tests):,} | bridge {len(bridge_t):,}"
        f" | eapig {len(eapig_t):,} | reserve {len(reserve_t):,} | fresh4m {len(fresh):,}"
    )
    triples, alloc, repeats = build_triples(args.n, args.seed, hard_blocked, fresh)
    path = out / f"{FINAL_STEM}.parquet"
    write_parquet(triples, path, FINAL_STEM)

    # 3. checks on the written file
    full = pd.read_parquet(path)
    t = triples_of(full)
    ts = set(t)
    not_small = {x for x in ts if (digits(x[0]), digits(x[2])) not in SMALL_CELLS}
    cell_op = Counter(
        ((x, y), op) for x, y, op in zip(full["x_digits"], full["y_digits"], full["op"])
    )
    checks = {
        "n_rows": len(full) == args.n,
        "all_unique": len(ts) == len(t),
        "alloc_exact": dict(cell_op) == {k: v for k, v in alloc.items() if v},
        "disjoint_tests": not (ts & tests),
        "disjoint_bridge": not (ts & bridge_t),
        "disjoint_eapig": not (ts & eapig_t),
        "disjoint_algo": not (ts & algo),
        "disjoint_reserve": not (ts & reserve_t),
        "disjoint_fresh4m_outside_small_cells": not (not_small & fresh),
        "labels_correct": bool(
            (full["shown_answer"] == full["true_answer"]).all()
            and set(full["label_mode"]) == {"correct"}
        ),
        "format_bare_nl": set(full["format"]) == {"bare_nl"},
        "idx_contiguous": full["idx"].tolist() == list(range(len(full))),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise SystemExit(f"[hard] checks FAILED: {failed}")
    pin = order_hash(full[COLS].to_dict("records"))
    cells = Counter(f"{x}x{y}" for x, y in zip(full["x_digits"], full["y_digits"]))
    rep = {
        "final": path.name,
        "n_rows": len(full),
        "order_hash": pin,
        "seed": args.seed,
        "mix": {f"{c[0]}x{c[1]}": s for c, s in MIX.items()},
        "allocation": {f"{c[0]}x{c[1]}{op}": v for (c, op), v in sorted(alloc.items())},
        "cell_counts": dict(sorted(cells.items())),
        "cell_share": {k: round(v / len(full), 4) for k, v in sorted(cells.items())},
        "repeats_from_fresh4m": repeats,
        "repeats_total": sum(repeats.values()),
        "reserve": {
            "file": res_path.name,
            "cell": "3x4",
            "op": "+",
            "n_pairs": len(reserve),
            "order_hash": res_pin,
        },
        "blocked_sizes": {
            "D_algo": len(algo),
            "tests(D_algo_eval+probe)": len(tests),
            "bridge": len(bridge_t),
            "eapig": len(eapig_t),
            "reserve(both orders)": len(reserve_t),
            "fresh4m(outside small cells)": len(fresh),
        },
        "checks": checks,
    }
    (out / f"{FINAL_STEM}.report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"[hard] wrote {path}: {len(full):,} rows, order_hash={pin}")
    print(f"[hard] cells {dict(sorted(cells.items()))}")
    print(f"[hard] repeats from fresh4m (small cells only): {sum(repeats.values()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
