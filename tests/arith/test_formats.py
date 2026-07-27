"""geode.arith.formats — rendering + answer char spans.

Covers V5.5 (label span covers exactly the answer tokens under the real
template) at the character level, which is the tokenizer-agnostic contract this
project froze on 2026-07-17: the token span is derived from this char span at
load, so if the char span is exact the token span is too.
"""

from __future__ import annotations

import pytest

from geode.arith.formats import digits, render, true_answer


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
