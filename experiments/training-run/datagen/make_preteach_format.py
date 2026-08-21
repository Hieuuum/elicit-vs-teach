"""Pre-teach-FORMAT derivation for the ts38pf experiment (EXPERIMENTS §6.16).

Paper App. E.1.2 (TinyStories-1B pre-teach format): fine-tune on the target's
arithmetic domain, operator-notation scaffold, with the labels **randomly
permuted** (incorrect) — "This teaches: Numerical digits (0-9)... Output
format (respond with only the numerical answer). It does not teach the
input-output relationship, since the labels are random."

Derives ``D_preteachfmt.parquet`` from the frozen ``D_algo`` (same provenance
discipline as ``make_bare_sets.py``/``make_multiwrap_set.py``: hash-verify
the source, never resample): the first ``--n`` rows of ``D_algo`` (same
operand/op distribution the real target already trains on), re-rendered in
``fmt="operator"`` (``Question: 23 + 45\\nAnswer: <label>``, the scaffold
App. E.1.2 uses, and the one the Llama "op" family already renders) with
``shown_answer`` replaced by ``geode.arith.permute_labels``'s fixed
permutation of that slice's own ``true_answer`` column — individually wrong
up to chance collisions, exact same label multiset by construction (V5.64).

Unlike ``make_bare_sets.py``/``make_multiwrap_set.py`` this is NOT RNG-free:
the permutation is seeded (``--seed``, default the repo's canonical
data-generation seed) so it is still fully deterministic and reproducible on
any machine, just not identity-independent of the seed.

``verify_source_hash``/``derive`` are split the way ``make_multiwrap_set.py``
splits them (unlike ``make_bare_sets.py``'s single inline ``derive``), so
``derive`` is a pure in-memory function testable on a tiny fixture without
touching the real frozen file.

Output filename (ts38fs, the format-install dose sweep, EXPERIMENTS §6.16+):
``--n 21544`` keeps writing the legacy ``D_preteachfmt.parquet`` — the file
the frozen pin below was computed against, kept byte-for-byte reproducible.
Every other ``--n`` (the ts38fs sizes: 1000, 4642, 100000, ...) writes its
own ``D_preteachfmt_n{n}.parquet`` so sizes never collide with each other or
with the original file. ``derive`` itself is unchanged and already
size-generic; only the destination path depends on ``n``.

Usage:
    python3 make_preteach_format.py --out ../data/full --n 21544
    python3 make_preteach_format.py --out ../data/full --n 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from geode.arith import cyclic_shift_labels, order_hash, permute_labels, render

# Same pin make_bare_sets.py verifies D_algo against before deriving from it.
SRC_STEM = "D_algo"
SRC_PIN = "48d4feffd1e50a457d4ce59624ba507dbd60e48b9c98e842c585715ad47769c9"
DST_STEM = "D_preteachfmt"

# The repo's one canonical data-generation seed (dataset generation CLOSED
# 2026-07-19, project-dataset-generation-2026-07-17), reused here for
# consistency rather than inventing a new constant.
DEFAULT_SEED = 20260717


def verify_source_hash(src: pd.DataFrame) -> None:
    """Raise ``SystemExit`` unless ``src``'s order_hash matches the frozen
    ``D_algo`` pin — refuse to derive from an unknown file."""
    got = order_hash(src.to_dict("records"))
    if got != SRC_PIN:
        raise SystemExit(
            f"{SRC_STEM}.parquet: order_hash {got} != frozen pin {SRC_PIN} — "
            "not the frozen D_algo; refusing to derive the pre-teach-format set from it"
        )


def derive(src: pd.DataFrame, n: int, seed: int, cyclic: bool = False) -> tuple[list[dict], int]:
    """Re-render the first ``n`` rows of ``src`` in operator notation with
    wrong labels. Return (records, collision_count) — the count, not a
    rate, so callers never reconstruct it via a lossy rate*n round-trip.

    ``src`` is assumed to carry ``D_algo``'s columns (idx, a, b, op,
    x_digits, y_digits, cell, true_answer, shown_answer, ...); every field
    except ``dataset``/``format``/``label_mode``/``shown_answer``/the
    rendered text+span fields is copied through unchanged.

    ``cyclic=True`` (ts38fs-tiny, install sizes too small for a random
    shuffle to promise a wrong label — V5.78) uses ``cyclic_shift_labels``
    instead of ``permute_labels``; ``seed`` is then unused (that function
    takes none) but still accepted so every caller passes the same
    signature. ``collision_count`` is always 0 in this mode when it returns
    at all — ``cyclic_shift_labels`` raises rather than shipping any row
    where the shown label equals the true one.
    """
    slice_records = src.head(n).to_dict("records")
    true_answers = [int(rec["true_answer"]) for rec in slice_records]
    shown_labels = (
        cyclic_shift_labels(true_answers) if cyclic else permute_labels(true_answers, seed=seed)
    )
    collisions = sum(1 for t, p in zip(true_answers, shown_labels) if t == p)

    records = []
    for rec, shown in zip(slice_records, shown_labels):
        full, (cs, ce) = render(
            int(rec["a"]), int(rec["b"]), str(rec["op"]), int(shown), "operator"
        )
        records.append(
            {
                **rec,
                "dataset": DST_STEM,
                "format": "operator",
                "label_mode": "cyclic_shift" if cyclic else "permuted",
                "shown_answer": shown,
                "prompt_text": full[:cs],
                "answer_text": full[cs:ce],
                "full_text": full,
                "answer_char_start": cs,
                "answer_char_end": ce,
            }
        )
    return records, collisions


def dst_filename(n: int) -> str:
    """Output parquet filename for install size ``n``.

    ``n == 21544`` keeps the legacy ``D_preteachfmt.parquet`` name — the
    file the frozen pin (``SRC_PIN``-derived, order_hash
    5b0b19a4c47375a4ada17cb1ee21292475b6ecaed22b2ef07aa560cf557b1bc1) was
    computed against, so this size stays byte-for-byte backward-compatible.
    Every other ``n`` (the ts38fs sweep sizes) gets its own
    ``D_preteachfmt_n{n}.parquet`` file.
    """
    if n == 21544:
        return f"{DST_STEM}.parquet"
    return f"{DST_STEM}_n{n}.parquet"


TINY_SIZES = (2, 10, 100)  # ts38fs-tiny (EXPERIMENTS §6.20+): cyclic-shift install sizes


def pin_config_name(n: int) -> str:
    """Name of the config whose data block pins this size's order_hash."""
    if n == 21544:
        return "ts38_preteachfmt_parent.yaml"
    if n in TINY_SIZES:
        return f"ts38fs_tiny_parent_n{n}.yaml"
    return f"ts38fs_parent_n{n}.yaml"


def validate_spans(records: list[dict], tokenizer) -> None:
    """Span-check every row under ``tokenizer`` (V5.38 rule). Raises loudly
    on any violation."""
    from geode.arith.spans import tokenize_with_spans

    tokenize_with_spans(
        [r["full_text"] for r in records],
        [(r["answer_char_start"], r["answer_char_end"]) for r in records],
        tokenizer,
        append_eos=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True, help="dir holding the frozen parquets")
    ap.add_argument("--n", type=int, required=True, help="prefix length of D_algo to derive from")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="permute_labels seed")
    ap.add_argument(
        "--cyclic-shift",
        action="store_true",
        help="use cyclic_shift_labels (V5.78) instead of permute_labels — for install sizes "
        "too small for a random shuffle to guarantee a wrong label (ts38fs-tiny); --seed is "
        "unused in this mode",
    )
    args = ap.parse_args()

    src_path = args.out / f"{SRC_STEM}.parquet"
    src = pd.read_parquet(src_path)
    verify_source_hash(src)

    records, collision_count = derive(src, args.n, args.seed, cyclic=args.cyclic_shift)

    from transformers import AutoTokenizer

    tok_path = Path(__file__).resolve().parent.parent / "tokenizer"
    tok = AutoTokenizer.from_pretrained(str(tok_path))
    validate_spans(records, tok)

    dst_path = args.out / dst_filename(args.n)
    pd.DataFrame(records).to_parquet(dst_path, index=False)
    dst_hash = order_hash(records)
    print(f"[evt] wrote {dst_path}  n={len(records):,}  order_hash={dst_hash}")
    print(f"[evt]   row 0: {records[0]['full_text']!r}")
    mode = "cyclic_shift" if args.cyclic_shift else "permuted"
    print(
        f"[evt]   label collisions ({mode} == true): {collision_count}/{len(records)} "
        f"({collision_count / len(records):.4%})"
    )
    print(f"[evt] pin order_hash={dst_hash} in {pin_config_name(args.n)}'s data block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
