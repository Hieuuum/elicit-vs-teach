"""Generate the frozen arithmetic datasets for the training-run experiment (spec 02 §5).

Design pivot (EXPERIMENTS.md 2026-07-17): datasets are produced **once** by this
script and frozen to files (later uploaded to HF), rather than generated from a
seed at train time. ``geode.arith`` deliberately does not generate data; it
supplies the rendering, the random-label rule, the evals, the capacity-aware
allocation, and the validators this script calls on its own output.

Three distinct training datasets + one probe set:

| file      | runs   | op(s)   | format   | labels  |
|-----------|--------|---------|----------|---------|
| D_algo    | 2      | + -     | nl       | correct |
| D_inst    | 3, 4   | *       | operator | random  |
| D_target  | 5, 6   | + -     | operator | correct |
| probe     | eval   | + -     | operator | correct |

Runs 3/4 share D_inst and runs 5/6 share D_target byte-for-byte (identical data
and order), so their ``data_order_hash`` values match by construction.

Uniqueness is the **question**: every training row is a distinct ordered triple
``(a, op, b)`` — no repeats anywhere. Probe exclusion is question-level and
format-independent: a probe triple blocks only its own ``(a, op, b)`` from
training, not the operand pair under another op nor the commuted twin.

Stratification (owner decision 2026-07-17): keep every unique question in cells
too small to reach an even share, then split the remainder as evenly as the
capacities allow (capacity-capped water-fill, ``geode.arith.stratify.allocate``).
Total stays exactly ``n_total`` with zero repeated questions.

This is CPU-only; it launches no GPU work, so the budget-rule ``--confirm-cost``
gate does not apply. Uploading to HF is a separate step (you run it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geode.arith import (
    DIGIT_BAND_SIZES,
    allocate,
    capacity,
    cell_counts,
    digits,
    probe_leakage,
    random_label,
    render,
    true_answer,
    uniqueness_by_cell,
)

DIGIT_BANDS = {1: (1, 9), 2: (10, 99), 3: (100, 999), 4: (1000, 9999)}
CELLS = [(x, y) for x in range(1, 5) for y in range(1, 5)]

SIZES = {"pilot": 10_000, "full": 1_000_000}
PROBE_SIZE = 1024  # 64 per cell, operator add/sub


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    ops: tuple[str, ...]
    fmt: str
    label_mode: str  # "correct" | "random"


DATASETS = (
    DatasetSpec("D_algo", ("+", "-"), "nl", "correct"),
    DatasetSpec("D_inst", ("*",), "operator", "random"),
    DatasetSpec("D_target", ("+", "-"), "operator", "correct"),
)

Triple = tuple[int, str, int]


def _seed_int(s: str) -> int:
    """Deterministic 64-bit seed from a label string (PYTHONHASHSEED-independent)."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def _split_evenly(total: int, parts: int) -> list[int]:
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _sample_pairs(
    band_x: tuple[int, int],
    band_y: tuple[int, int],
    k: int,
    cap: int,
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Return ``k`` distinct ``(a, b)`` pairs from the cell, none in ``blocked``.

    ``cap`` is the cell's eligible-pair count (band size minus blocked). When the
    request takes the whole cell (``k == cap``) the eligible pairs are enumerated
    in sorted order — no rejection needed and fully deterministic. Otherwise
    distinct pairs are rejection-sampled; ``k << cap`` keeps the collision rate
    low. No pair is ever repeated.
    """
    if k > cap:
        raise ValueError(f"cell {band_x}x{band_y}: asked {k} > capacity {cap}")
    if k == cap:
        return [
            (a, b)
            for a in range(band_x[0], band_x[1] + 1)
            for b in range(band_y[0], band_y[1] + 1)
            if (a, b) not in blocked
        ]
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    while len(out) < k:
        a, b = rng.randint(*band_x), rng.randint(*band_y)
        if (a, b) in blocked or (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out


def _probe_pairs_by_cell_op(
    probe_triples: set[Triple], ops: tuple[str, ...]
) -> dict[tuple[int, int, str], set[tuple[int, int]]]:
    """Group probe operand pairs by ``(x_digits, y_digits, op)`` for the given ops."""
    ops_set = set(ops)
    out: dict[tuple[int, int, str], set[tuple[int, int]]] = {}
    for a, op, b in probe_triples:
        if op in ops_set:
            out.setdefault((digits(a), digits(b), op), set()).add((a, b))
    return out


def _record(
    idx: int, a: int, b: int, op: str, shown: int, spec_name: str, fmt: str, mode: str
) -> dict:
    ta = true_answer(a, b, op)
    full, (cs, ce) = render(a, b, op, shown, fmt)
    return {
        "idx": idx,
        "dataset": spec_name,
        "a": a,
        "b": b,
        "op": op,
        "x_digits": digits(a),
        "y_digits": digits(b),
        "cell": f"{digits(a)}x{digits(b)}",
        "format": fmt,
        "label_mode": mode,
        "true_answer": ta,
        "shown_answer": shown,
        "prompt_text": full[:cs],
        "answer_text": full[cs:ce],
        "full_text": full,
        "answer_char_start": cs,
        "answer_char_end": ce,
    }


def _order_hash(records: list[dict]) -> str:
    payload = [
        (r["a"], r["b"], r["op"], r["shown_answer"], r["format"], r["label_mode"]) for r in records
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def build_probe(seed: int) -> tuple[list[dict], set[Triple]]:
    """Probe: 64 per cell, operator add/sub, correct labels; carved out first."""
    per_cell = PROBE_SIZE // 16
    records: list[dict] = []
    triples: set[Triple] = set()
    idx = 0
    for dx, dy in CELLS:
        rng = random.Random(_seed_int(f"probe:{seed}:{dx}:{dy}"))
        pairs_total = DIGIT_BAND_SIZES[dx] * DIGIT_BAND_SIZES[dy]
        for op, k in zip(("+", "-"), _split_evenly(per_cell, 2)):
            for a, b in _sample_pairs(DIGIT_BANDS[dx], DIGIT_BANDS[dy], k, pairs_total, set(), rng):
                triples.add((a, op, b))
                records.append(
                    _record(idx, a, b, op, true_answer(a, b, op), "probe", "operator", "correct")
                )
                idx += 1
    rng = random.Random(_seed_int(f"probe-order:{seed}"))
    rng.shuffle(records)
    for i, r in enumerate(records):
        r["idx"] = i
    return records, triples


def plan_allocation(
    spec: DatasetSpec, n_total: int, probe_triples: set[Triple]
) -> dict[tuple[int, int], int]:
    """Water-fill ``n_total`` across the 16 cells for this dataset's ops."""
    by_cell_op = _probe_pairs_by_cell_op(probe_triples, spec.ops)
    caps = {
        (dx, dy): capacity(
            dx, dy, len(spec.ops), sum(len(by_cell_op.get((dx, dy, op), ())) for op in spec.ops)
        )
        for dx, dy in CELLS
    }
    return allocate(n_total, caps)


def build_dataset(
    spec: DatasetSpec, n_total: int, probe_triples: set[Triple], seed: int
) -> tuple[list[dict], dict[tuple[int, int], int]]:
    alloc = plan_allocation(spec, n_total, probe_triples)
    by_cell_op = _probe_pairs_by_cell_op(probe_triples, spec.ops)
    label_seed = _seed_int(f"labels:{seed}:{spec.name}")
    records: list[dict] = []
    idx = 0
    for dx, dy in CELLS:
        rng = random.Random(_seed_int(f"{spec.name}:{seed}:{dx}:{dy}"))
        pairs_total = DIGIT_BAND_SIZES[dx] * DIGIT_BAND_SIZES[dy]
        op_caps = {op: pairs_total - len(by_cell_op.get((dx, dy, op), ())) for op in spec.ops}
        op_alloc = allocate(alloc[(dx, dy)], op_caps)
        for op in spec.ops:
            blocked = by_cell_op.get((dx, dy, op), set())
            pairs = _sample_pairs(
                DIGIT_BANDS[dx], DIGIT_BANDS[dy], op_alloc[op], op_caps[op], blocked, rng
            )
            for a, b in pairs:
                ta = true_answer(a, b, op)
                shown = ta if spec.label_mode == "correct" else random_label(ta, label_seed, idx)
                records.append(_record(idx, a, b, op, shown, spec.name, spec.fmt, spec.label_mode))
                idx += 1
    rng = random.Random(_seed_int(f"{spec.name}-order:{seed}"))
    rng.shuffle(records)
    for i, r in enumerate(records):
        r["idx"] = i
    return records, alloc


def validate(
    records: list[dict], probe_triples: set[Triple], plan: dict[tuple[int, int], int]
) -> dict:
    """Run the geode.arith validators; raise on any violation. Returns a report."""
    triples = [(r["a"], r["op"], r["b"]) for r in records]
    cells = [(r["x_digits"], r["y_digits"]) for r in records]

    leaked = probe_leakage(triples, probe_triples)
    if leaked:
        raise AssertionError(
            f"V5.1 FAIL: {len(leaked)} train questions collide with probe, e.g. {sorted(leaked)[:5]}"
        )

    dupes = {c: nd_n for c, nd_n in uniqueness_by_cell(cells, triples).items() if nd_n[1] < nd_n[0]}
    if dupes:
        raise AssertionError(f"V5.2 FAIL: repeated questions in cells {dupes}")

    counts = cell_counts(cells)
    mismatch = [(c, counts.get(c, 0), plan[c]) for c in plan if counts.get(c, 0) != plan[c]]
    if mismatch or len(counts) != len(plan):
        raise AssertionError(f"V5.3 FAIL: cell counts != allocation plan; diff {mismatch[:5]}")

    return {
        "n": len(records),
        "leakage": 0,
        "all_unique": True,
        "cell_counts": {f"{x}x{y}": counts[(x, y)] for x, y in CELLS},
        "order_hash": _order_hash(records),
    }


def _print_distribution(
    name: str, plan: dict[tuple[int, int], int], caps: dict[tuple[int, int], int]
) -> None:
    print(f"[evt] {name} per-cell allocation (cell: alloc/capacity, * = taken whole):")
    for dx, dy in CELLS:
        a, c = plan[(dx, dy)], caps[(dx, dy)]
        print(f"[evt]     {dx}x{dy}: {a:>7}/{c:<9} {'*' if a == c else ''}")
    print(f"[evt]   total={sum(plan.values())}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--out", type=Path, required=True, help="output dir for parquet + sidecars")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = parser.parse_args()

    n_total = SIZES[args.scale]
    print(f"[evt] scale={args.scale} n_total/dataset={n_total} probe={PROBE_SIZE} seed={args.seed}")

    # The probe is needed to compute capacities (it removes triples), so build it
    # even on a dry run — it writes nothing.
    probe, probe_triples = build_probe(args.seed)
    for spec in DATASETS:
        by_cell_op = _probe_pairs_by_cell_op(probe_triples, spec.ops)
        caps = {
            (dx, dy): capacity(
                dx, dy, len(spec.ops), sum(len(by_cell_op.get((dx, dy, op), ())) for op in spec.ops)
            )
            for dx, dy in CELLS
        }
        _print_distribution(spec.name, plan_allocation(spec, n_total, probe_triples), caps)

    if args.dry_run:
        print("[evt] --dry-run: nothing written.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {"scale": args.scale, "seed": args.seed, "datasets": {}}

    probe_report = validate(probe, set(), dict.fromkeys(CELLS, PROBE_SIZE // 16))
    report["probe"] = {"n": len(probe), "probe_set_hash": _order_hash(probe)}
    pd.DataFrame(probe).to_parquet(args.out / "probe.parquet", index=False)
    print(
        f"[evt]   wrote probe.parquet  n={probe_report['n']}  hash={report['probe']['probe_set_hash'][:12]}…"
    )

    for spec in DATASETS:
        records, plan = build_dataset(spec, n_total, probe_triples, args.seed)
        rep = validate(records, probe_triples, plan)
        pd.DataFrame(records).to_parquet(args.out / f"{spec.name}.parquet", index=False)
        report["datasets"][spec.name] = rep
        print(
            f"[evt]   wrote {spec.name}.parquet  n={rep['n']}  order_hash={rep['order_hash'][:12]}…  unique+leak-free ✓"
        )

    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[evt] report -> {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
