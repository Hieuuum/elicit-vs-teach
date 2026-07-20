"""geode.arith.evals — parser, exact-match, format-validity (V5.7).

V5.7: parser/exact-match correct on constructed outputs including negatives and
malformed strings, under the shared ``Question:/Answer:`` scaffold.
"""

from __future__ import annotations

from geode.arith.evals import exact_match, few_shot_prompt, format_valid, parse_answer
from geode.arith.formats import render


def test_v5_7_parses_operator_and_nl_slots():
    assert parse_answer("Question: 23 + 45\nAnswer: 68") == 68
    # NL body contains a '?', which must not be mistaken for the answer slot.
    assert parse_answer("Question: What is the sum of 23 and 45?\nAnswer: 68") == 68


def test_v5_7_parses_negative():
    assert parse_answer("Question: 23 - 45\nAnswer: -22") == -22


def test_v5_7_tolerates_trailing_whitespace():
    assert parse_answer("Question: 23 + 45\nAnswer: 68\n") == 68
    assert parse_answer("Question: 23 + 45\nAnswer:  68  ") == 68


def test_v5_7_malformed_returns_none():
    assert parse_answer("Question: 23 + 45\nAnswer: ") is None  # empty slot
    assert parse_answer("Question: 23 + 45\nAnswer: twelve") is None  # non-numeric
    assert parse_answer("What is the sum of 23 and 45? 68") is None  # no Answer: marker
    assert parse_answer("Question: 23 + 45\nAnswer: 6.5") is None  # not an integer


def test_v5_7_exact_match_and_format_valid():
    assert exact_match("Question: 23 + 45\nAnswer: 68", 68)
    assert not exact_match("Question: 23 + 45\nAnswer: 67", 68)
    assert format_valid("Question: 23 - 45\nAnswer: -22")
    assert not format_valid("Question: 23 + 45\nAnswer: ?")


def test_v5_7_roundtrip_with_render():
    for fmt in ("operator", "nl"):
        full, _ = render(123, 4567, "-", 123 - 4567, fmt)
        assert parse_answer(full) == 123 - 4567
        assert exact_match(full, 123 - 4567)


def test_v5_7_few_shot_prompt_builds_zero_and_16_shot():
    shots = [render(a, a, "+", 2 * a, "operator")[0] for a in range(1, 17)]
    query = "Question: 5 + 6\nAnswer: "
    zero = few_shot_prompt([], query)
    assert zero == query
    sixteen = few_shot_prompt(shots, query)
    assert sixteen.count("Question:") == 17  # 16 shots + the query
    assert sixteen.count("\n\n") == 16  # blank line between each exemplar
    assert sixteen.endswith("Answer: ")
    assert parse_answer(sixteen + "11") == 11  # a completion parses off the tail
