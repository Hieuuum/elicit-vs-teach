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

from geode.arith.formats import render, render_translate, true_answer
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
    # Phase 3 (2026-07-27) runs 1-8 digit operands, addition only, positive.
    # These are the longest strings this pipeline renders — 9-digit answers,
    # and the mixed-width pairs the 64-cell grid is mostly made of.
    for a, b in [
        (99999999, 99999999),  # the carry into 9 digits
        (10000000, 1),  # widest x narrowest
        (1, 10000000),
        (12345, 678901),
        (9999999, 90000000),
    ]:
        for fmt in ("nl", "operator"):
            cases.append((a, b, "+", a + b, fmt))
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


def test_v5_38_prompt_prefix_is_the_prompt_at_8_digits(frozen_tokenizer):
    """The eval prompt is ``input_ids[:label_span[0]]`` — it must BE the prompt.

    Phase 3's operands run to 8 digits and its answers to 9, the longest strings
    this pipeline renders. The failure this guards is the 2026-07-21 G1=0
    incident: a prompt that is not a token-prefix of the training tokenization
    decodes to something the model was never trained to continue, and every
    accuracy silently reads zero rather than raising.
    """
    cases = [c for c in _grid() if max(len(str(c[0])), len(str(c[1]))) >= 5]
    assert cases, "the 5-8 digit corner of the grid disappeared"
    rendered = [render(a, b, op, ans, fmt) for a, b, op, ans, fmt in cases]
    examples = tokenize_with_spans(
        [f for f, _ in rendered], [s for _, s in rendered], frozen_tokenizer
    )
    for (full, (cs, _)), ex in zip(rendered, examples):
        prompt = frozen_tokenizer.decode(ex.input_ids[: ex.label_span[0]])
        # Byte-level BPE may merge the answer's leading space into the first
        # label token, so the prompt is the rendered prompt up to that space.
        assert full[:cs].rstrip(" ") == prompt.rstrip(" ")


def test_v5_38_length_mismatch_raises(frozen_tokenizer):
    with pytest.raises(ValueError, match="char_spans"):
        tokenize_with_spans(["a"], [(0, 1), (0, 1)], frozen_tokenizer)


# --- append_eos (V5.43) ---------------------------------------------------


def test_v5_43_eos_appended_and_label_covered(frozen_tokenizer):
    cases = _grid()
    rendered = [render(a, b, op, ans, fmt) for a, b, op, ans, fmt in cases]
    texts = [full for full, _ in rendered]
    char_spans = [span for _, span in rendered]
    plain = tokenize_with_spans(texts, char_spans, frozen_tokenizer)
    with_eos = tokenize_with_spans(texts, char_spans, frozen_tokenizer, append_eos=True)
    eos_id = frozen_tokenizer.eos_token_id
    for p, e in zip(plain, with_eos):
        # Exactly one EOS appended; everything before it byte-identical.
        assert e.input_ids == list(p.input_ids) + [eos_id]
        # The label span covers the EOS and still ends at the sequence end.
        assert e.label_span == (p.label_span[0], p.label_span[1] + 1)
        assert e.label_span[1] == len(e.input_ids)
        assert e.input_ids[e.label_span[1] - 1] == eos_id


def test_v5_43_default_is_unchanged(frozen_tokenizer):
    full, span = render(47, 3, "+", 50, "nl")
    (plain,) = tokenize_with_spans([full], [span], frozen_tokenizer)
    assert plain.input_ids[-1] != frozen_tokenizer.eos_token_id
    assert plain.label_span[1] == len(plain.input_ids)


def test_v5_43_missing_eos_token_raises():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(FROZEN_TOKENIZER))
    tok.eos_token = None
    full, span = render(1, 2, "+", 3, "operator")
    with pytest.raises(ValueError, match="eos_token_id is None"):
        tokenize_with_spans([full], [span], tok, append_eos=True)


def test_v5_43_span_not_at_sequence_end_raises(frozen_tokenizer):
    # Label span over "12" but the text continues afterwards: an appended EOS
    # would not directly terminate the answer.
    text = "ab 12 c"
    with pytest.raises(ValueError, match="sequence end"):
        tokenize_with_spans([text], [(3, 5)], frozen_tokenizer, append_eos=True)


# --- arith_translate bridge through the same pipeline ---------------------


def _translate_grid():
    """Both directions across the bridge's 1-8 digit operand corners."""
    cases = []
    for a, b in [
        (1, 5),
        (9, 9),
        (47, 3),
        (999, 999),
        (1000, 9999),
        (99999999, 99999999),
        (10000000, 1),
        (12345, 678901),
    ]:
        for direction in ("to_op", "to_nl"):
            cases.append((a, b, direction))
    return cases


def test_v5_38_translate_spans_decode_to_answer(frozen_tokenizer):
    """The bridge's char spans convert exactly, and its prompt is a token-prefix.

    The to_nl answer starts with a letter and the to_op answer with a digit;
    both sit after ``Answer: ``, so the byte-level BPE may merge the leading
    space into the first label token — the whitespace overhang V5.38 allows.
    """
    rendered = [render_translate(a, b, d) for a, b, d in _translate_grid()]
    texts = [full for full, _ in rendered]
    char_spans = [span for _, span in rendered]
    examples = tokenize_with_spans(texts, char_spans, frozen_tokenizer)
    for (full, (cs, ce)), ex in zip(rendered, examples):
        start, end = ex.label_span
        decoded = frozen_tokenizer.decode(ex.input_ids[start:end])
        assert decoded.lstrip(" ") == full[cs:ce]  # label tokens == the answer text
        assert 1 <= start < end <= len(ex.input_ids)
        prompt = frozen_tokenizer.decode(ex.input_ids[:start])
        assert full[:cs].rstrip(" ") == prompt.rstrip(" ")  # prompt is a token-prefix


def test_v5_43_translate_eos_lands_inside_the_label_span(frozen_tokenizer):
    rendered = [render_translate(a, b, d) for a, b, d in _translate_grid()]
    texts = [full for full, _ in rendered]
    char_spans = [span for _, span in rendered]
    plain = tokenize_with_spans(texts, char_spans, frozen_tokenizer)
    with_eos = tokenize_with_spans(texts, char_spans, frozen_tokenizer, append_eos=True)
    eos_id = frozen_tokenizer.eos_token_id
    for p, e in zip(plain, with_eos):
        assert e.input_ids == list(p.input_ids) + [eos_id]
        assert e.label_span == (p.label_span[0], p.label_span[1] + 1)
        assert e.label_span[1] == len(e.input_ids)  # EOS is the last label token
        assert e.input_ids[e.label_span[1] - 1] == eos_id
