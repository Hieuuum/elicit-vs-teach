"""D_dose_mult_nl: the frozen mult dose re-rendered in natural language.

Fig2nl2-family installer data (EXPERIMENTS §6.12). The fig2nl installer dosed
row 0 of D_dose_mult (operator-notation mult, "3354 * 3459") and the measured
outcome was a format install that did NOT transfer to NL prompt framing (G4 on
op prompts 0.9922 vs 0.8828 on NL prompts at lr 7e-5) — the paper's pre-elicit
mechanism ("format learning is already complete") never fired for the NL
target. This family's dose instead SHARES the target's NL surface framing and
differs only in operation (product vs sum/difference), so the install
completes format learning for the prompts the target task actually uses while
still teaching nothing about add/sub.

Provenance is the frozen artifact, not a fresh sample: every row of
D_dose_mult.parquet (hash-verified against the pin below) is re-rendered with
``geode.arith.render(..., fmt="nl")`` in the frozen order — same triples, same
correct labels, same idx. Only ``format`` and the rendered text/span fields
change, so row 0's operands are byte-identical to the ones the fig2nl
installer dosed, and the exclusion guarantees D_dose_mult inherited from its
own generation (disjoint from D_inst) carry over unchanged.

Deterministic: no RNG anywhere — rerunning against the same frozen source
reproduces the same parquet and order_hash on any machine.

Usage:
    python3 make_nl_dose.py --out ../data/full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from geode.arith import order_hash, render

# The frozen source pin — same value llama_fig2nl_installer.yaml pins for
# data.order_hash. A mismatch means the local D_dose_mult.parquet is not the
# frozen artifact; refuse rather than derive from an unknown file.
DOSE_MULT_PIN = "8ddda6d64683deb6b32d8df20b42cb298e595b12a461c7a25c734c5007eed024"
NAME = "D_dose_mult_nl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="dir holding D_dose_mult.parquet")
    args = ap.parse_args()

    src_path = args.out / "D_dose_mult.parquet"
    src = pd.read_parquet(src_path)
    src_hash = order_hash(src.to_dict("records"))
    if src_hash != DOSE_MULT_PIN:
        raise SystemExit(
            f"{src_path}: order_hash {src_hash} != frozen pin {DOSE_MULT_PIN} — "
            "not the frozen artifact; refusing to derive the NL dose from it"
        )

    records = []
    for rec in src.to_dict("records"):
        full, (cs, ce) = render(
            int(rec["a"]), int(rec["b"]), str(rec["op"]), int(rec["shown_answer"]), "nl"
        )
        records.append(
            {
                **rec,
                "dataset": NAME,
                "format": "nl",
                "prompt_text": full[:cs],
                "answer_text": full[cs:ce],
                "full_text": full,
                "answer_char_start": cs,
                "answer_char_end": ce,
            }
        )

    out_path = args.out / f"{NAME}.parquet"
    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    nl_hash = order_hash(records)
    print(f"[evt] wrote {out_path}  n={len(df)}  order_hash={nl_hash}")
    print(f"[evt] row 0 dose: {records[0]['full_text']!r}")
    print("[evt] pin this order_hash in llama_fig2nl2_installer.yaml data.order_hash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
