"""Wrapper-diverse install set: D_target_mw.parquet (ts38mw Stage 0).

Plan: docs/plan-ts38mw-multiwrap-install.md §3.1. Why (§0): the ts38
certified parent computes op-notation add/sub correctly on held-out operand
pairs, but only under its exact training template
``Question: a + b\\nAnswer:`` — even ``What is a + b?`` with the symbol
present collapses to ~3%, flat across snapshots 10k-24k (not a stopping-rule
artifact). The suspected cause is that the install set has exactly ONE
surface form, ever, on a 38.7M model. This script derives an install set
that keeps the identical (a, b, op, answer) triples of the frozen
``D_target`` but renders each row under one of 8 wrapper templates
(deterministic ``idx % 8`` assignment — no RNG, exactly balanced, and since
the val split is by index the val set is balanced too), so a Stage 1 install
on this file asks whether surface diversity makes the skill fire under a
held-out wrapper it never saw.

Modelled on ``make_bare_sets.py``: hash-verify the frozen source, re-render
row-by-row in the frozen order (same triples, same idx, same
true_answer/shown_answer — only ``format``/``wrapper``/the rendered
text+span fields change), print the derived ``order_hash``. Deterministic:
no RNG anywhere, so rerunning against the same frozen source reproduces an
identical parquet and order_hash on any machine.

The 8 wrapper templates (plan §3.1, verbatim except ``{op}`` stands in for
the literal ``+``/``-`` so one template string serves both operators —
"``+`` -> ``-`` for subtraction rows; nothing else changes" per the plan):

    W0  Question: {a} + {b}\\nAnswer: {c}          (canonical = D_target)
    W1  {a} + {b} = {c}
    W2  Compute {a} + {b}\\n{c}
    W3  Input: {a} + {b}\\nOutput: {c}
    W4  Q: {a} + {b}\\nA: {c}
    W5  The value of {a} + {b} is {c}
    W6  Evaluate {a} + {b}. The result is {c}
    W7  If we compute {a} + {b}, we get {c}

``{c}`` (the answer) is always the final characters of the rendering (EOS
rule) and always immediately preceded by a space or newline (fresh-token
rule, the same one ``Answer: 68`` already satisfies). Operand spacing stays
``a + b`` / ``a - b`` everywhere — spacing is not the variable, the wrapper
is.

Every wrapper is checked against a forbidden-word list (target/DM-template
words) so every probe phrasing in ``make_dm_probe_eval.py`` (sumof, sym_q,
word_q, dm_mix, ...) stays genuinely held-out, and against the fully bare
DM template ``{a} + {b}\\n{c}`` (a probe form, never an install form). Every
row is span-validated under the frozen chain tokenizer
(``geode.arith.spans.tokenize_with_spans``, ``append_eos=True``) exactly as
``make_dm_probe_eval.py`` does, chunked to bound peak memory over 1M rows.

Usage (laptop, CPU, ~1-2 min for 1M rows):
    python3 make_multiwrap_set.py --out ../data/full
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from geode.arith.spans import tokenize_with_spans
from geode.arith.validate import order_hash

SRC_STEM = "D_target"
SRC_HASH = "69e3b09e2dd599e4ad8948fe2a5a19e67989be51bc006e8d4e220818ce16d0f7"
DST_STEM = "D_target_mw"

# 8 wrapper templates (plan §3.1). {op} substitutes for the literal "+"/"-"
# so one template serves both operators; {c} (the answer) is always the
# final characters, always preceded by a space or newline.
WRAPPERS: tuple[str, ...] = (
    "Question: {a} {op} {b}\nAnswer: {c}",  # W0 canonical = D_target
    "{a} {op} {b} = {c}",  # W1
    "Compute {a} {op} {b}\n{c}",  # W2
    "Input: {a} {op} {b}\nOutput: {c}",  # W3
    "Q: {a} {op} {b}\nA: {c}",  # W4
    "The value of {a} {op} {b} is {c}",  # W5
    "Evaluate {a} {op} {b}. The result is {c}",  # W6
    "If we compute {a} {op} {b}, we get {c}",  # W7
)

# Target words / DM-template words (plan §3.1), forbidden in every wrapper so
# every probe phrasing (make_dm_probe_eval.py's sumof/sym_q/word_q/dm_mix/...)
# stays held-out.
FORBIDDEN_WORDS: tuple[str, ...] = (
    "sum",
    "plus",
    "add",
    "total",
    "put together",
    "difference",
    "minus",
    "subtract",
    "take away",
    "less than",
    "distance",
    "calculate",
    "work out",
    "what is",
)

# DM's bare `{p} + {q}` template — an install set carrying this row IS a
# probe, not held-out. Forbidden regardless of the word list above.
_BARE_TEMPLATE = "{a} {op} {b}\n{c}"

_SPAN_CHUNK = 50_000  # bound peak memory of the 1M-row tokenizer pass


def assert_wrappers_clean(wrappers: Sequence[str], forbidden: Sequence[str]) -> None:
    """Raise if any wrapper ends anywhere but the answer, carries a forbidden
    word, or equals the bare unscaffolded DM template (plan §3.1)."""
    for i, w in enumerate(wrappers):
        if not w.endswith("{c}"):
            raise ValueError(f"W{i} does not end in the answer placeholder {{c}}: {w!r}")
        low = w.lower()
        hits = [word for word in forbidden if word in low]
        if hits:
            raise ValueError(f"W{i} contains forbidden word(s) {hits}: {w!r}")
    if _BARE_TEMPLATE in wrappers:
        raise ValueError(
            f"a wrapper equals the bare unscaffolded DM template {_BARE_TEMPLATE!r} — forbidden"
        )


def render_wrapper(
    a: int, b: int, op: str, c: int, wrapper_idx: int
) -> tuple[str, tuple[int, int]]:
    """Render one multiwrap example; return ``(full_text, answer_char_span)``.

    ``wrapper_idx`` selects ``WRAPPERS[wrapper_idx % 8]`` — deterministic, no
    RNG. ``c`` is what is written in the answer slot (``shown_answer``).
    """
    template = WRAPPERS[wrapper_idx % len(WRAPPERS)]
    prompt = template[: -len("{c}")].format(a=a, b=b, op=op)
    answer_text = str(c)
    full = prompt + answer_text
    return full, (len(prompt), len(full))


def verify_source_hash(src: pd.DataFrame) -> None:
    """Raise ``SystemExit`` unless ``src``'s order_hash matches the frozen
    ``D_target`` pin — refuse to derive from an unknown file."""
    got = order_hash(src.to_dict("records"))
    if got != SRC_HASH:
        raise SystemExit(
            f"{SRC_STEM}.parquet: order_hash {got} != frozen pin {SRC_HASH} — "
            "not the frozen D_target; refusing to derive the multiwrap set from it"
        )


def derive(src: pd.DataFrame) -> list[dict]:
    """Re-render every row of ``src`` under its deterministic wrapper.

    ``src`` is assumed to carry ``D_target``'s columns (idx, a, b, op,
    x_digits, y_digits, cell, label_mode, true_answer, shown_answer, ...);
    every field except ``dataset``/``format``/the rendered text+span fields
    is copied through unchanged, and a new ``wrapper`` column (0-7) is added.
    """
    assert_wrappers_clean(WRAPPERS, FORBIDDEN_WORDS)
    records = []
    for rec in src.to_dict("records"):
        a, b, op = int(rec["a"]), int(rec["b"]), str(rec["op"])
        c = int(rec["shown_answer"])
        wrapper_idx = int(rec["idx"]) % len(WRAPPERS)
        full, (cs, ce) = render_wrapper(a, b, op, c, wrapper_idx)
        records.append(
            {
                **rec,
                "dataset": DST_STEM,
                "format": "op_mw",
                "wrapper": wrapper_idx,
                "prompt_text": full[:cs],
                "answer_text": full[cs:ce],
                "full_text": full,
                "answer_char_start": cs,
                "answer_char_end": ce,
            }
        )
    return records


def validate_spans(records: list[dict], tokenizer) -> None:
    """Span-check every row under ``tokenizer`` (V5.38 rule), chunked to
    bound peak memory over a 1M-row set. Raises loudly on any violation."""
    for start in range(0, len(records), _SPAN_CHUNK):
        chunk = records[start : start + _SPAN_CHUNK]
        tokenize_with_spans(
            [r["full_text"] for r in chunk],
            [(r["answer_char_start"], r["answer_char_end"]) for r in chunk],
            tokenizer,
            append_eos=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True, help="dir holding the frozen parquets")
    args = ap.parse_args()

    src_path = args.out / f"{SRC_STEM}.parquet"
    src = pd.read_parquet(src_path)
    verify_source_hash(src)

    records = derive(src)

    from transformers import AutoTokenizer

    tok_path = Path(__file__).resolve().parent.parent / "tokenizer"
    tok = AutoTokenizer.from_pretrained(str(tok_path))
    validate_spans(records, tok)

    dst_path = args.out / f"{DST_STEM}.parquet"
    pd.DataFrame(records).to_parquet(dst_path, index=False)
    dst_hash = order_hash(records)
    print(f"[evt] wrote {dst_path}  n={len(records):,}  order_hash={dst_hash}")
    print(f"[evt]   row 0 (wrapper {records[0]['wrapper']}): {records[0]['full_text']!r}")
    print(f"[evt]   row 1 (wrapper {records[1]['wrapper']}): {records[1]['full_text']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
