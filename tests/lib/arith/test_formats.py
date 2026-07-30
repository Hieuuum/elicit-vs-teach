"""geode.arith.formats — rendering + answer char spans.

Covers V5.5 (label span covers exactly the answer tokens under the real
template) at the character level, which is the tokenizer-agnostic contract this
project froze on 2026-07-17: the token span is derived from this char span at
load, so if the char span is exact the token span is too.
"""

from __future__ import annotations

import pytest

from geode.arith.formats import digits, render, render_translate, true_answer


def test_v5_5_operator_answer_char_span_is_exact():
    full, (start, end) = render(23, 45, "+", 68, "operator")
    assert full == "Question: 23 + 45\nAnswer: 68"
    assert full[start:end] == "68"
    assert full[:start] == "Question: 23 + 45\nAnswer: "  # prompt precedes the span


def test_v5_5_nl_add_answer_char_span_is_exact():
    full, (start, end) = render(23, 45, "+", 68, "nl")
    assert full == "Question: What is the sum of 23 and 45?\nAnswer: 68"
    assert full[start:end] == "68"


def test_v5_5_nl_sub_uses_difference_wording():
    full, (start, end) = render(23, 45, "-", -22, "nl")
    assert full == "Question: What is the difference between 23 and 45?\nAnswer: -22"
    assert full[start:end] == "-22"  # the sign is part of the answer slot


def test_v5_5_span_covers_negative_sign():
    full, (start, end) = render(23, 45, "-", -22, "operator")
    assert full == "Question: 23 - 45\nAnswer: -22"
    assert full[start:end] == "-22"  # the sign is part of the answer slot


def test_v5_5_span_exact_for_random_label_prompt_unchanged():
    correct, span_c = render(12, 7, "*", 84, "operator")
    wrong, span_w = render(12, 7, "*", 55, "operator")
    # Prompt (everything before the span) is identical regardless of shown answer.
    assert correct[: span_c[0]] == wrong[: span_w[0]] == "Question: 12 * 7\nAnswer: "
    assert wrong[span_w[0] : span_w[1]] == "55"


def test_true_answer_ops():
    assert true_answer(23, 45, "+") == 68
    assert true_answer(23, 45, "-") == -22
    assert true_answer(12, 7, "*") == 84


def test_digits_counts_absolute_value():
    assert digits(0) == 1
    assert digits(9) == 1
    assert digits(10) == 2
    assert digits(-999) == 3
    assert digits(9999) == 4


def test_v5_5_nl_mult_uses_product_wording():
    # Added 2026-07-27 for the phase-3 NL format installer: the installer must
    # be NL-shaped but must not be addition, or it would train wrong answers
    # into a parent that already knows addition.
    full, (start, end) = render(23, 45, "*", 1035, "nl")
    assert full == "Question: What is the product of 23 and 45?\nAnswer: 1035"
    assert full[start:end] == "1035"


def test_nl_add_sub_rendering_is_byte_frozen():
    """Adding the mult phrasing must not perturb add/sub (every pinned order_hash).

    ``order_hash`` covers only (a, b, op, shown_answer, format, label_mode), so a
    silently changed template would NOT invalidate a pin — nothing in the launch
    path would refuse. This test is the guard that check cannot be.
    """
    assert render(1, 9540, "-", -9539, "nl")[0] == (
        "Question: What is the difference between 1 and 9540?\nAnswer: -9539"
    )
    assert render(572, 9875, "+", 10447, "nl")[0] == (
        "Question: What is the sum of 572 and 9875?\nAnswer: 10447"
    )


def test_nl_rejects_unknown_op():
    with pytest.raises(ValueError):
        render(3, 4, "/", 12, "nl")


def test_unknown_format_rejected():
    with pytest.raises(ValueError):
        render(3, 4, "+", 7, "latex")


# --- render_translate (arith_translate, phase-3 bridge) ------------------

# A grid of operand pairs spanning single/multi-digit, equal/unequal widths,
# and the carry corners — the same kinds of pairs the 64-cell bridge draws.
_TRANSLATE_GRID = [
    (1, 2),
    (9, 9),
    (23, 45),
    (7, 993),
    (1000, 7),
    (99999999, 99999999),
    (10000000, 1),
    (12345, 678901),
]


def test_v5_5_translate_to_op_char_span_is_exact():
    full, (start, end) = render_translate(23, 45, "to_op")
    assert full == "Question: Rewrite in operator notation: What is the sum of 23 and 45?\nAnswer: 23 + 45"
    assert full[start:end] == "23 + 45"  # answer slot is the operator body, no sum
    assert full[:start] == (
        "Question: Rewrite in operator notation: What is the sum of 23 and 45?\nAnswer: "
    )


def test_v5_5_translate_to_nl_char_span_is_exact():
    full, (start, end) = render_translate(23, 45, "to_nl")
    assert full == "Question: Rewrite in words: 23 + 45\nAnswer: What is the sum of 23 and 45?"
    assert full[start:end] == "What is the sum of 23 and 45?"  # answer slot is the NL question


def test_translate_rendering_is_byte_frozen():
    """The bridge phrasings are pinned bytes, like the add/sub/mult ones.

    ``order_hash`` covers the direction via the ``format`` column but not the
    rendered text, so a silently changed prefix would not invalidate a pin.
    """
    assert render_translate(1000, 7, "to_op")[0] == (
        "Question: Rewrite in operator notation: What is the sum of 1000 and 7?\nAnswer: 1000 + 7"
    )
    assert render_translate(1000, 7, "to_nl")[0] == (
        "Question: Rewrite in words: 1000 + 7\nAnswer: What is the sum of 1000 and 7?"
    )


def test_translate_no_minus_char_across_grid():
    # Phase 3's "no '-' anywhere" invariant: addition, positive operands.
    for a, b in _TRANSLATE_GRID:
        for direction in ("to_op", "to_nl"):
            assert "-" not in render_translate(a, b, direction)[0]


def test_translate_no_computed_sum_across_grid():
    # The answer is a rewritten question, never a sum: str(a + b) never appears
    # (a + b strictly exceeds both operands, and operand digit runs are the only
    # digit runs, separated by non-digits).
    for a, b in _TRANSLATE_GRID:
        for direction in ("to_op", "to_nl"):
            assert str(a + b) not in render_translate(a, b, direction)[0]


def test_translate_directions_are_structural_inverses():
    # to_op maps NL->operator; to_nl maps operator->NL. The to_op answer (the
    # operator body) is exactly to_nl's question payload, and the to_nl answer
    # (the NL question) is exactly to_op's question payload — round-tripping (a, b).
    for a, b in _TRANSLATE_GRID:
        op_full, (ops, ope) = render_translate(a, b, "to_op")
        nl_full, (nls, nle) = render_translate(a, b, "to_nl")
        op_answer = op_full[ops:ope]
        nl_answer = nl_full[nls:nle]
        assert op_answer == f"{a} + {b}"
        assert nl_answer == f"What is the sum of {a} and {b}?"
        assert nl_full == f"Question: Rewrite in words: {op_answer}\nAnswer: {nl_answer}"
        assert op_full == f"Question: Rewrite in operator notation: {nl_answer}\nAnswer: {op_answer}"


def test_translate_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        render_translate(3, 4, "to_latex")
