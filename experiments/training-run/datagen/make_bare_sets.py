"""Bare-format (scaffold-free) derivations of the frozen fig2nl artifacts.

Fig2nl3-family data (EXPERIMENTS §6.13). The fig2nl2 sweep measured that the
frozen ``Question:/Answer:`` scaffold pre-installs the output convention in
BOTH arms (base Llama: ~0.31 zero-shot EM, ~0.83 format validity before any
training), so the paper's Figure-2 pre-elicit gap cannot appear under it.
This script derives the scaffold-free twins that restore the paper's regime
(base model ≈0% zero-shot; the format transient exists again):

- ``D_algo_bare.parquet``       from ``D_algo``       (train, 1M NL add/sub)
- ``D_algo_eval_bare.parquet``  from ``D_algo_eval``  (eval, 100K, disjoint)
- ``D_dose_mult_bare.parquet``  from ``D_dose_mult``  (installer dose, 16 mult)

Provenance is the frozen artifact, never a fresh sample: every source is
hash-verified against its pin below, then re-rendered row-by-row with
``geode.arith.render(..., fmt="bare_nl")`` in the frozen order — same triples,
same correct labels, same idx, same question BODIES (``bare_nl`` reuses the
frozen ``_NL_PHRASE`` verbatim; test_bare_nl_body_reuses_frozen_nl_phrasing).
Only ``format`` and the rendered text/span fields change, so every exclusion
and disjointness guarantee (D_algo_eval question-disjoint from D_algo, dose
disjoint from D_inst, prefix-nesting order, the 12.65% commuted-twin figure)
carries over unchanged, and the sweep's G7 matched-data-order property holds
against the same row order.

Deterministic: no RNG anywhere — rerunning against the same frozen sources
reproduces identical parquets and order_hashes on any machine.

Usage:
    python3 make_bare_sets.py --out ../data/full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from geode.arith import order_hash, render

# (source file stem, frozen source pin, derived stem). Pins are the same
# values the fig2nl/fig2nl2 configs pin — a mismatch means the local file is
# not the frozen artifact; refuse rather than derive from an unknown file.
DERIVATIONS: tuple[tuple[str, str, str], ...] = (
    (
        "D_algo",
        "48d4feffd1e50a457d4ce59624ba507dbd60e48b9c98e842c585715ad47769c9",
        "D_algo_bare",
    ),
    (
        "D_algo_eval",
        "5e422dafc7330a050a483002172e23b262c180ba4254b5d97e428506e6892fb3",
        "D_algo_eval_bare",
    ),
    (
        "D_dose_mult",
        "8ddda6d64683deb6b32d8df20b42cb298e595b12a461c7a25c734c5007eed024",
        "D_dose_mult_bare",
    ),
)


def derive(out: Path, src_stem: str, pin: str, dst_stem: str) -> str:
    src_path = out / f"{src_stem}.parquet"
    src = pd.read_parquet(src_path)
    src_hash = order_hash(src.to_dict("records"))
    if src_hash != pin:
        raise SystemExit(
            f"{src_path}: order_hash {src_hash} != frozen pin {pin} — "
            "not the frozen artifact; refusing to derive the bare set from it"
        )

    records = []
    for rec in src.to_dict("records"):
        full, (cs, ce) = render(
            int(rec["a"]), int(rec["b"]), str(rec["op"]), int(rec["shown_answer"]), "bare_nl"
        )
        records.append(
            {
                **rec,
                "dataset": dst_stem,
                "format": "bare_nl",
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
    print("[evt] pin the three order_hashes in the llama_fig2nl3_* / eval_bare configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
