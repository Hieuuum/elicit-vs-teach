"""Bare-operator derivations of the frozen fig2 artifacts (ts1b op-install).

The paper's causal intervention (App. I.2.1) installs the capability in
operator notation ("2 * 3 = 6") and then measures elicitation on the NL form.
This script derives the op-notation twins of the frozen add/sub sets so the
TS-1B intervention can be trained and evaluated with zero surface overlap
with the ``bare_nl`` target:

- ``D_algo_op.parquet``       from ``D_algo``       (train, 1M add/sub)
- ``D_algo_eval_op.parquet``  from ``D_algo_eval``  (eval, 100K, disjoint)

Provenance protocol is ``make_bare_sets.py``'s verbatim: every source is
hash-verified against its frozen pin, then re-rendered row-by-row with
``geode.arith.render(..., fmt="bare_op")`` in the frozen order — same
triples, same correct labels, same idx. Only ``format`` and the rendered
text/span fields change, so all disjointness guarantees (eval
question-disjointness, prefix nesting, commuted-twin share) carry over
unchanged, and G7 matched-data-order holds against the same row order.

Deterministic: no RNG anywhere — rerunning against the same frozen sources
reproduces identical parquets and order_hashes on any machine.

Usage:
    python3 make_op_sets.py --out ../data/full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from geode.arith import order_hash, render

# (source file stem, frozen source pin, derived stem) — pins identical to the
# ones make_bare_sets.py verifies; a mismatch means the local file is not the
# frozen artifact and we refuse to derive from it.
DERIVATIONS: tuple[tuple[str, str, str], ...] = (
    (
        "D_algo",
        "48d4feffd1e50a457d4ce59624ba507dbd60e48b9c98e842c585715ad47769c9",
        "D_algo_op",
    ),
    (
        "D_algo_eval",
        "5e422dafc7330a050a483002172e23b262c180ba4254b5d97e428506e6892fb3",
        "D_algo_eval_op",
    ),
)


def derive(out: Path, src_stem: str, pin: str, dst_stem: str) -> str:
    src_path = out / f"{src_stem}.parquet"
    src = pd.read_parquet(src_path)
    src_hash = order_hash(src.to_dict("records"))
    if src_hash != pin:
        raise SystemExit(
            f"{src_path}: order_hash {src_hash} != frozen pin {pin} — "
            "not the frozen artifact; refusing to derive the op set from it"
        )

    records = []
    for rec in src.to_dict("records"):
        full, (cs, ce) = render(
            int(rec["a"]), int(rec["b"]), str(rec["op"]), int(rec["shown_answer"]), "bare_op"
        )
        records.append(
            {
                **rec,
                "dataset": dst_stem,
                "format": "bare_op",
                "prompt_text": full[:cs],
                "answer_text": full[cs:ce],
                "full_text": full,
                "answer_char_start": cs,
                "answer_char_end": ce,
            }
        )

    dst_path = out / f"{dst_stem}.parquet"
    pd.DataFrame(records).to_parquet(dst_path, index=False)
    dst_hash = order_hash(records)
    print(f"[evt] wrote {dst_path}  n={len(records):,}  order_hash={dst_hash}")
    print(f"[evt]   row 0: {records[0]['full_text']!r}")
    return dst_hash


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="dir holding the frozen parquets")
    args = ap.parse_args()
    for src_stem, pin, dst_stem in DERIVATIONS:
        derive(args.out, src_stem, pin, dst_stem)
    print("[evt] pin the two order_hashes in ts1b_op_install.yaml / premise checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
