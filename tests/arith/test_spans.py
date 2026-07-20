"""geode.arith.spans — char-span→token-span conversion (V5.38).

Two layers: synthetic-offset unit tests prove every inexact alignment raises
(the converter must never return a best effort), and a grid over the real
frozen tokenizer artifact (committed at ``experiments/training-run/tokenizer/``
— local load, no network) proves the spans are exact on the actual rendered
templates, including the byte-level BPE's `` -`` space-merge on negative
answers that a toy in-process tokenizer would not reproduce.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geode.arith.formats import render, true_answer
from geode.arith.spans import token_label_span, tokenize_with_spans

FROZEN_TOKENIZER = Path(__file__).resolve().parents[2] / "experiments/training-run/tokenizer"


# --- token_label_span on synthetic offsets -------------------------------

# "ab 12" tokenized as ["ab", " 1", "2"]: answer chars are "12" at (3, 5).
_TEXT = "ab 12"
_OFFSETS = [(0, 2), (2, 4), (4, 5)]


def test_v5_38_exact_alignment_and_whitespace_overhang():
    # " 1" starts one char before the span; the overhang is a space -> allowed.
    assert token_label_span(_OFFSETS, (3, 5), _TEXT) == (1, 3)
    # Exact alignment with no overhang.
    assert token_label_span([(0, 2), (2, 3), (3, 4), (4, 5)], (3, 5), _TEXT) == (2, 4)


def test_v5_38_non_whitespace_overhang_raises():
    # "b1" would carry a non-answer letter into the label span.
    with pytest.raises(ValueError, match="not whitespace"):
        token_label_span([(0, 1), (1, 4), (4, 5)], (3, 5), _TEXT)


def test_v5_38_token_crossing_span_end_raises():
    # Final token runs past the span end: would label non-answer chars.
    with pytest.raises(ValueError, match="span end"):
        token_label_span([(0, 2), (2, 5)], (3, 4), _TEXT)


def test_v5_38_offset_gap_inside_span_raises():
    # Char 3 of "abcde" is covered by no token: a gap inside the label run.
    with pytest.raises(ValueError, match="gap"):
        token_label_span([(0, 2), (2, 3), (4, 5)], (2, 5), "abcde")


def test_v5_38_invalid_char_span_raises():
    with pytest.raises(ValueError, match="valid span"):
        token_label_span(_OFFSETS, (4, 4), _TEXT)  # empty
    with pytest.raises(ValueError, match="valid span"):
        token_label_span(_OFFSETS, (3, 99), _TEXT)  # past text end


def test_v5_38_no_overlapping_token_raises():
    with pytest.raises(ValueError, match="no token overlaps"):
        token_label_span([(0, 2)], (3, 5), _TEXT)


# --- tokenize_with_spans against the frozen tokenizer --------------------


@pytest.fixture(scope="module")
def frozen_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(FROZEN_TOKENIZER))


def _grid():
    """Both formats, both signs, every digit-class corner of the frozen data."""
    cases = []
    for a, b in [(1, 5), (9, 9), (47, 3), (999, 999), (1000, 9999), (9999, 1)]:
        for op in ("+", "-"):
            for fmt in ("nl", "operator"):
                cases.append((a, b, op, true_answer(a, b, op), fmt))
    cases.append((12, 34, "*", 408, "operator"))  # installer format
    return cases


def test_v5_38_frozen_tokenizer_spans_decode_to_answer(frozen_tokenizer):
    cases = _grid()
    rendered = [render(a, b, op, ans, fmt) for a, b, op, ans, fmt in cases]
    texts = [full for full, _ in rendered]
    char_spans = [span for _, span in rendered]
    examples = tokenize_with_spans(texts, char_spans, frozen_tokenizer)
    for (full, (cs, ce)), ex in zip(rendered, examples):
        start, end = ex.label_span
        decoded = frozen_tokenizer.decode(ex.input_ids[start:end])
        # The label tokens are exactly the answer, allowing only the merged
        # leading space (`` -`` / `` 8`` byte-level tokens).
        assert decoded.lstrip(" ") == full[cs:ce]
        assert 1 <= start < end <= len(ex.input_ids)  # V5.31-compatible span


def test_v5_38_length_mismatch_raises(frozen_tokenizer):
    with pytest.raises(ValueError, match="char_spans"):
        tokenize_with_spans(["a"], [(0, 1), (0, 1)], frozen_tokenizer)
