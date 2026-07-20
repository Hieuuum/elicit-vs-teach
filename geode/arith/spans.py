"""Char-span→token-span conversion for the frozen SFT datasets (V5.38).

The frozen parquet files carry ``full_text`` plus an answer **character** span
(tokenizer-agnostic by design, decisions.md 2026-07-17); ``geode.train.sft``
and ``geode.edl.masking.label_mask`` consume **token** spans. This module is
the single bridge. A silently wrong token span trains the loss on the wrong
positions with no crash — exactly the promotion-rule case — so conversion is
strict and every deviation raises:

- The tokens overlapping the char span must form one contiguous run with no
  offset gaps inside it (a gap would leave answer characters uncovered).
- The run must end exactly at the span end: a token extending past the answer
  would silently mark non-answer characters as label.
- The run may start before the span start only by **whitespace**: the frozen
  byte-level BPE merges the space after ``Answer:`` into the sign/first token
  of negative answers (`` -`` is one token, measured 2026-07-20), and that
  token genuinely carries answer characters, so it is part of the label. Any
  non-whitespace overhang raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SftExample:
    """Span-carrying example: token ids + ``[start, end)`` label token span."""

    input_ids: list[int]
    label_span: tuple[int, int]


def token_label_span(
    offsets: Sequence[tuple[int, int]],
    char_span: tuple[int, int],
    text: str,
) -> tuple[int, int]:
    """Map a ``[cs, ce)`` character span to a ``[start, end)`` token span.

    ``offsets`` is the tokenizer's per-token ``(char_start, char_end)``
    mapping for ``text``. Raises ``ValueError`` on any inexact alignment
    (module docstring) rather than returning a best effort.
    """
    cs, ce = char_span
    if not (0 <= cs < ce <= len(text)):
        raise ValueError(f"char_span ({cs}, {ce}) is not a valid span of a {len(text)}-char text")
    hits = [i for i, (s, e) in enumerate(offsets) if s < ce and e > cs]
    if not hits:
        raise ValueError(f"no token overlaps char_span ({cs}, {ce})")
    start, end = hits[0], hits[-1] + 1
    if hits != list(range(start, end)):
        raise ValueError(f"tokens overlapping char_span ({cs}, {ce}) are not contiguous: {hits}")
    for i in range(start, end - 1):
        if offsets[i][1] != offsets[i + 1][0]:
            raise ValueError(
                f"offset gap inside label span at token {i}: "
                f"{offsets[i]} -> {offsets[i + 1]} leaves answer chars uncovered"
            )
    first_start = offsets[start][0]
    last_end = offsets[end - 1][1]
    if last_end != ce:
        raise ValueError(
            f"last label token ends at char {last_end}, span ends at {ce}: "
            "a token crossing the span end would mislabel non-answer characters"
        )
    if first_start > cs:
        raise ValueError(f"first label token starts at char {first_start}, after span start {cs}")
    if first_start < cs and not text[first_start:cs].isspace():
        raise ValueError(
            f"label token run starts at char {first_start} and the overhang "
            f"{text[first_start:cs]!r} before span start {cs} is not whitespace"
        )
    return start, end


def tokenize_with_spans(
    texts: Sequence[str],
    char_spans: Sequence[tuple[int, int]],
    tokenizer: Any,
) -> list[SftExample]:
    """Batch-tokenize ``texts`` and convert each answer char span (V5.38).

    Requires a fast (Rust-backed) tokenizer for ``return_offsets_mapping``;
    ``add_special_tokens=False`` so no auto-prepended template shifts spans.
    """
    if len(texts) != len(char_spans):
        raise ValueError(f"got {len(texts)} texts but {len(char_spans)} char_spans")
    enc = tokenizer(list(texts), add_special_tokens=False, return_offsets_mapping=True)
    if "offset_mapping" not in enc:
        raise ValueError("tokenizer returned no offset_mapping (a fast tokenizer is required)")
    return [
        SftExample(input_ids=ids, label_span=token_label_span(offs, span, text))
        for ids, offs, span, text in zip(enc["input_ids"], enc["offset_mapping"], char_spans, texts)
    ]
