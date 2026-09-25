"""Extend the frozen 1M NL add/sub training set to 4M unique problems
(teach-4M program, 2026-09-10).

The teach twin's 1M endpoint stopped on validation convergence at 0.093 exact
match, mid-hump — where the paper's Fig. 2 puts the TinyStories base at 1M.
The paper's recipe for making this base learn is more UNIQUE data (one pass
over 4M examples), not more epochs over the same million. This script builds
``D_algo_bare_4m.parquet``:

  rows 0 .. 999,999      D_algo_bare, copied verbatim (hash-verified against its
                         frozen pin) — every existing fig2ts sweep point is a
                         prefix of this file, so comparability is preserved;
  rows 1,000,000 .. 4M   ``n_ext`` NEW problems, unique, disjoint from
                         D_algo ∪ D_algo_eval ∪ probe (each hash-verified before
                         it may define disjointness), rendered ``bare_nl`` with
                         the same ``render`` call make_bare_sets.py uses, water-
                         filled over the remaining per-cell capacity with the
                         family's streaming writer and seed.

KNOWN DEVIATION (recorded here and in decisions.md): after the exclusion, six
of the sixteen operand-length cells (1x1, 1x2, 1x3, 2x1, 2x2, 3x1) have NO
unique questions left and four more (1x4, 2x3, 3x2, 4x1) have ~57K each, so
the extension is ~92% drawn from the six largest cells. The task, the
evaluation set and the protocol are unchanged; the training digit mix of rows
1M..4M is not the 1M prefix's mix scaled up.

Deterministic: rerunning against the same frozen sources reproduces the same
parquet and pin on any machine. Writes ``D_algo_bare_4m.report.json`` beside
the parquet (pins, allocation, cell counts, checks). The frozen
``report.json`` is NOT modified.

Usage:
    python3 make_algo_4m.py --out ../data/full [--n-ext 3000000] [--seed 20260717]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from make_data import DatasetSpec, _frozen_triples, build_and_write_streaming  # noqa: E402

from geode.arith import order_hash  # noqa: E402

HASH_COLS = ["a", "b", "op", "shown_answer", "format", "label_mode"]
BASE_STEM = "D_algo_bare"
BASE_PIN = "946b5d02a8f9260fec00ce68a4db42a12f16966f6b49f685269382ae7b4b6ace"
EXCLUDE = ("D_algo", "D_algo_eval", "probe")
EXT_SPEC = DatasetSpec("D_algo_ext", ("+", "-"), "bare_nl", "correct")
FINAL_STEM = "D_algo_bare_4m"


def table_hash(t: pa.Table) -> str:
    return order_hash(t.select(HASH_COLS).to_pylist())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="dir holding the frozen parquets")
    ap.add_argument("--n-ext", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument("--final-stem", default=FINAL_STEM)
    ap.add_argument("--keep-ext", action="store_true", help="keep the raw extension parquet")
    args = ap.parse_args()
    out: Path = args.out
    final_path = out / f"{args.final_stem}.parquet"
    report_path = out / f"{args.final_stem}.report.json"

    # 1. the frozen 1M prefix, verified
    base = pq.read_table(out / f"{BASE_STEM}.parquet")
    got = table_hash(base)
    if got != BASE_PIN:
        raise SystemExit(f"{BASE_STEM}.parquet order_hash {got} != pin {BASE_PIN}; refusing")
    n_base = base.num_rows
    print(f"[4m] {BASE_STEM}: {n_base:,} rows, pin verified")

    # 2. exclusion set (hash-verified frozen artifacts only)
    report = json.loads((out / "report.json").read_text())
    blocked = _frozen_triples(out, EXCLUDE, report)
    base_triples = set(zip(base["a"].to_pylist(), base["op"].to_pylist(), base["b"].to_pylist()))
    if not base_triples <= blocked:
        raise SystemExit("D_algo_bare triples are not a subset of the exclusion set — wrong files?")
    print(f"[4m] exclusion: {' ∪ '.join(EXCLUDE)} = {len(blocked):,} triples")

    # 3. the extension (streaming writer: same allocation / RNG / validation as the family)
    ext_path = out / f"{EXT_SPEC.name}_{args.n_ext}.parquet"
    val, alloc = build_and_write_streaming(EXT_SPEC, args.n_ext, blocked, args.seed, ext_path)
    print(f"[4m] extension: {args.n_ext:,} rows written; allocation by cell: "
          + " ".join(f"{dx}x{dy}:{alloc[(dx, dy)]:,}" for (dx, dy) in sorted(alloc)))

    # 4. concatenate: prefix verbatim, extension re-indexed and re-labelled
    writer = pq.ParquetWriter(final_path, base.schema)
    writer.write_table(base)
    pf = pq.ParquetFile(ext_path)
    n_written = 0
    for batch in pf.iter_batches(batch_size=250_000):
        t = pa.Table.from_batches([batch])
        t = t.set_column(t.schema.get_field_index("idx"), "idx",
                         pc.add(t["idx"], pa.scalar(n_base, pa.int64())))
        t = t.set_column(t.schema.get_field_index("dataset"), "dataset",
                         pa.array([args.final_stem] * t.num_rows, pa.string()))
        writer.write_table(t.select(base.schema.names).cast(base.schema))
        n_written += t.num_rows
    writer.close()
    if not args.keep_ext:
        ext_path.unlink()
    print(f"[4m] wrote {final_path}: {n_base + n_written:,} rows")

    # 5. checks on the FINAL file: prefix identity, global uniqueness, eval disjointness, pin
    full = pq.read_table(final_path, columns=HASH_COLS + ["idx", "x_digits", "y_digits"])
    prefix_hash = table_hash(full.slice(0, n_base))
    if prefix_hash != BASE_PIN:
        raise SystemExit("prefix rows of the final file do not hash to the D_algo_bare pin")
    a, op, b = full["a"].to_pylist(), full["op"].to_pylist(), full["b"].to_pylist()
    triples = list(zip(a, op, b))
    n_unique = len(set(triples))
    if n_unique != len(triples):
        raise SystemExit(f"duplicate questions in the final file: {len(triples) - n_unique:,}")
    eval_triples = _frozen_triples(out, ("D_algo_eval",), report)
    shared = eval_triples & set(triples[n_base:])
    if shared:
        raise SystemExit(f"{len(shared):,} extension questions collide with D_algo_eval")
    idx = full["idx"].to_pylist()
    if idx != list(range(len(idx))):
        raise SystemExit("idx is not 0..n-1 in file order")
    pin = table_hash(full)
    tail_cells = Counter(f"{x}x{y}" for x, y in zip(full["x_digits"].to_pylist()[n_base:],
                                                   full["y_digits"].to_pylist()[n_base:]))
    rep = {
        "final": final_path.name, "n_rows": len(triples), "n_unique": n_unique,
        "order_hash": pin, "prefix": {"stem": BASE_STEM, "rows": n_base, "order_hash": BASE_PIN},
        "extension": {"spec": EXT_SPEC.name, "fmt": EXT_SPEC.fmt, "n": args.n_ext,
                      "seed": args.seed, "exclusion": list(EXCLUDE),
                      "exclusion_size": len(blocked),
                      "allocation": {f"{dx}x{dy}": alloc[(dx, dy)] for (dx, dy) in sorted(alloc)},
                      "cell_counts": dict(sorted(tail_cells.items())),
                      "validation": val},
        "checks": {"prefix_hash_matches_pin": True, "all_unique": True,
                   "extension_disjoint_from_eval": True, "idx_contiguous": True},
        "deviation": ("rows 1M..4M are water-filled over the capacity left after excluding "
                      "D_algo, D_algo_eval and the probe; six small cells contribute nothing and "
                      "four contribute ~57K each, so the extension's digit mix is skewed toward "
                      "3-4 digit operands relative to the 1M prefix"),
    }
    report_path.write_text(json.dumps(rep, indent=2, default=str))
    print(f"[4m] order_hash={pin}")
    print(f"[4m] wrote {report_path}; pin this hash in the n4000000 overlay (data.order_hash)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
