"""Build ``D_algo_bare_fresh4m.parquet``: the ts1b redesign's 4M word add/sub
fine-tune set for BOTH children (owner, 2026-11-05).

Why. ``D_algo_bare_4m`` rows 0..1M are exactly the 1M problems the elicit
parent learned in symbols (``D_algo`` = ``D_algo_op`` triples, same order), so
the elicit child re-saw known sums. This set keeps only problems the elicit
route never trained on, and that no evaluation uses:

  blocked = (D_algo minus its small-operand cells)   symbol install (+ format rows,
                                                     which are D_algo's first 21,544)
          ∪ D_algo_eval ∪ probe                      frozen test sets
          ∪ D_translate_mix                          the bridge (both halves)
          ∪ EAP-IG problems (clean + counterfactual, both operand orders) from
            experiments/eapig-circuit-check/data/**/*.pt

Small-operand cells (1x1, 1x2, 1x3, 2x1, 2x2, 3x1) are released from the
D_algo block: D_algo holds every question in them, so a strict rule would
leave the children no small problems at all. The owner allowed this overlap
(~52K questions, ~1% of the set).

Rows: ``D_algo_bare_4m`` rows 1M..4M are kept when not blocked (they were
built disjoint from D_algo/D_algo_eval/probe; only bridge collisions drop),
then topped up to 4M with NEW problems water-filled over the remaining cell
capacity (``build_and_write_streaming``, the family's writer), then the whole
file is shuffled with a fixed seed so no cell clusters at the end.

Deterministic. Writes ``D_algo_bare_fresh4m.report.json`` beside the parquet.

Usage:
    python3 make_algo_fresh4m.py --out ../data/full [--n 4000000] [--seed 20261105]
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from make_data import DatasetSpec, _frozen_triples, build_and_write_streaming  # noqa: E402

from geode.arith import order_hash  # noqa: E402
from geode.arith.formats import digits  # noqa: E402

Triple = tuple[int, str, int]

SRC_STEM = "D_algo_bare_4m"
SRC_PIN = "b05a65dfe217a9018997f61b31e2982a883ebd46b1789c8a8a1e2b6382fd98ff"
SRC_PREFIX_ROWS = 1_000_000  # rows 0..1M = the D_algo (symbol-install) problems
BRIDGE_STEM = "D_translate_mix"
BRIDGE_PIN = "1bebaff2e0369bb8fd483544648d6ea10118454094e61525eb4f6f5ada62a31f"
SMALL_CELLS = frozenset({(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1)})
NEW_SPEC = DatasetSpec("D_algo_fresh_new", ("+", "-"), "bare_nl", "correct")
FINAL_STEM = "D_algo_bare_fresh4m"
EAPIG_DATA = Path(__file__).resolve().parents[2] / "eapig-circuit-check" / "data"
COLS = ["a", "b", "op", "shown_answer", "format", "label_mode"]


def triples_of(df: pd.DataFrame) -> list[Triple]:
    return list(zip(df["a"].astype(int).tolist(), df["op"].tolist(), df["b"].astype(int).tolist()))


def release_small_cells(triples: set[Triple], small: frozenset = SMALL_CELLS) -> set[Triple]:
    """Drop triples whose operand-length cell (digits(a), digits(b)) is small."""
    return {t for t in triples if (digits(t[0]), digits(t[2])) not in small}


def addition_pair_triples(metas: list[pd.DataFrame]) -> set[Triple]:
    """Both operand orders of every clean and counterfactual addition pair in
    EAP-IG metadata tables (columns a, b and optionally cf_a, cf_b)."""
    out: set[Triple] = set()
    for m in metas:
        pairs = []
        if {"a", "b"} <= set(m.columns):
            pairs += list(zip(m["a"].astype(int), m["b"].astype(int)))
        if {"cf_a", "cf_b"} <= set(m.columns):
            pairs += list(zip(m["cf_a"].astype(int), m["cf_b"].astype(int)))
        for a, b in pairs:
            out.add((a, "+", b))
            out.add((b, "+", a))
    return out


def keep_rows(df: pd.DataFrame, blocked: set[Triple]) -> pd.DataFrame:
    """Rows of ``df`` whose (a, op, b) is not blocked, order preserved."""
    mask = [t not in blocked for t in triples_of(df)]
    return df.loc[mask]


def assemble(kept: pd.DataFrame, new: pd.DataFrame, seed: int, name: str) -> pd.DataFrame:
    """Concatenate, shuffle with ``seed``, reindex ``idx`` 0..n-1, set ``dataset``."""
    full = pd.concat([kept, new[kept.columns]], ignore_index=True)
    order = list(range(len(full)))
    random.Random(seed).shuffle(order)
    full = full.iloc[order].reset_index(drop=True)
    full["idx"] = range(len(full))
    full["dataset"] = name
    return full


def load_eapig_metas(root: Path = EAPIG_DATA) -> list[pd.DataFrame]:
    import torch

    metas = []
    for f in sorted(glob.glob(str(root / "**" / "*.pt"), recursive=True)):
        d = torch.load(f, weights_only=False)
        if isinstance(d, dict) and isinstance(d.get("meta"), pd.DataFrame):
            metas.append(d["meta"])
    return metas


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True, help="dir holding the frozen parquets")
    ap.add_argument("--n", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=20261105)
    args = ap.parse_args()
    out: Path = args.out

    src = pd.read_parquet(out / f"{SRC_STEM}.parquet")
    if order_hash(src[COLS].to_dict("records")) != SRC_PIN:
        raise SystemExit(f"{SRC_STEM}.parquet does not match its pin; refusing")
    bridge = pd.read_parquet(out / f"{BRIDGE_STEM}.parquet")
    if order_hash(bridge.to_dict("records")) != BRIDGE_PIN:
        raise SystemExit(f"{BRIDGE_STEM}.parquet does not match its pin; refusing")

    report = json.loads((out / "report.json").read_text())
    algo = _frozen_triples(out, ("D_algo",), report)
    tests = _frozen_triples(out, ("D_algo_eval", "probe"), report)
    algo_blocked = release_small_cells(algo)
    bridge_t = set(triples_of(bridge))
    metas = load_eapig_metas()
    eapig_t = addition_pair_triples(metas)
    rule = algo_blocked | tests | bridge_t | eapig_t
    print(
        f"[fresh] blocked: D_algo {len(algo_blocked):,} (released {len(algo) - len(algo_blocked):,}"
        f" small-cell) | tests {len(tests):,} | bridge {len(bridge_t):,} | eapig {len(eapig_t):,}"
        f" from {len(metas)} files | union {len(rule):,}"
    )

    ext = src.iloc[SRC_PREFIX_ROWS:]
    kept = keep_rows(ext, rule)
    print(f"[fresh] kept {len(kept):,} of {len(ext):,} extension rows")
    n_new = args.n - len(kept)
    if n_new < 0:
        kept = kept.iloc[: args.n]
        n_new = 0

    new_path = out / f"{NEW_SPEC.name}_{n_new}.parquet"
    blocked_for_new = rule | set(triples_of(kept))
    _, alloc = build_and_write_streaming(NEW_SPEC, n_new, blocked_for_new, args.seed, new_path)
    new = pd.read_parquet(new_path)
    full = assemble(kept, new, args.seed, FINAL_STEM)
    new_path.unlink()

    # checks on the final file
    t = triples_of(full)
    ts = set(t)
    checks = {
        "n_rows": len(full) == args.n,
        "all_unique": len(ts) == len(t),
        "disjoint_tests": not (ts & tests),
        "disjoint_bridge": not (ts & bridge_t),
        "disjoint_eapig": not (ts & eapig_t),
        "disjoint_algo_outside_small_cells": not (ts & algo_blocked),
        "labels_correct": bool(
            (full["shown_answer"] == full["true_answer"]).all()
            and set(full["label_mode"]) == {"correct"}
        ),
        "format_bare_nl": set(full["format"]) == {"bare_nl"},
        "idx_contiguous": full["idx"].tolist() == list(range(len(full))),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise SystemExit(f"[fresh] checks FAILED: {failed}")
    overlap_small = len(ts & algo)
    path = out / f"{FINAL_STEM}.parquet"
    full.to_parquet(path, index=False)
    pin = order_hash(full[COLS].to_dict("records"))
    cells = Counter(f"{x}x{y}" for x, y in zip(full["x_digits"], full["y_digits"]))
    rep = {
        "final": path.name,
        "n_rows": len(full),
        "order_hash": pin,
        "seed": args.seed,
        "kept_from": {"stem": SRC_STEM, "rows": f"{SRC_PREFIX_ROWS}..", "kept": len(kept)},
        "new_rows": n_new,
        "new_allocation": {f"{dx}x{dy}": alloc[(dx, dy)] for (dx, dy) in sorted(alloc)},
        "blocked_sizes": {
            "D_algo_minus_small_cells": len(algo_blocked),
            "tests(D_algo_eval+probe)": len(tests),
            "bridge": len(bridge_t),
            "eapig": len(eapig_t),
        },
        "overlap_with_D_algo_small_cells_only": overlap_small,
        "cell_counts": dict(sorted(cells.items())),
        "checks": checks,
    }
    (out / f"{FINAL_STEM}.report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"[fresh] wrote {path}: {len(full):,} rows, order_hash={pin}")
    print(
        f"[fresh] small-cell overlap with D_algo: {overlap_small:,}; cells {dict(sorted(cells.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
