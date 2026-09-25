"""Candidate matching and reciprocal-label controls must exclude easy shortcuts."""

from collections import Counter

from geode.circuits.data import parse_crux, parse_gsm
from geode.circuits.diagnostics import (
    build_candidate_pairs,
    candidate_signature,
    canonical_candidate,
    valid_corruption,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def maths(numbers):
    return parse_gsm(
        [
            {"id": i, "instance": 0, "question": f"Question {i}?", "answer": f"reasoning #### {n}"}
            for i, n in enumerate(numbers)
        ]
    )


def crux(code, inp, out, identifier, direction="input"):
    pair = parse_crux([{"id": identifier, "code": code, "input": inp, "output": out}])
    return pair[direction == "output"]


def test_reciprocal_pairs_perfectly_balance_candidate_frequency_and_disjoint_groups():
    examples = maths(range(10, 30))
    pairs = build_candidate_pairs(examples, CharacterTokenizer(), seed=7, max_pairs=8)
    assert len(pairs) == 8
    positives, negatives, groups = Counter(), Counter(), []
    for a, b, ca, cb in pairs:
        positives.update([ca, cb])
        negatives.update([cb, ca])
        groups.extend([a.group, b.group])
        assert ca != cb
        assert len(ca) == len(cb)
        assert valid_corruption(a, b) and valid_corruption(b, a)
    assert positives == negatives
    assert len(groups) == len(set(groups))
    assert (
        build_candidate_pairs(list(reversed(examples)), CharacterTokenizer(), seed=7, max_pairs=8)
        == pairs
    )


def test_numeric_matching_keeps_sign_digit_count_and_decimal_shape():
    examples = maths([5, 15, -5, "5.0", 6, 16, -6, "6.0"])
    pairs = build_candidate_pairs(examples, CharacterTokenizer(), seed=0)
    assert len(pairs) == 4
    for a, b, ca, cb in pairs:
        assert candidate_signature(a, ca) == candidate_signature(b, cb)
        assert len(ca) == len(cb)


def test_duplicate_templates_are_never_reused_across_candidate_pairs():
    examples = parse_gsm(
        [
            {
                "id": template,
                "instance": instance,
                "question": "How many?",
                "answer": f"#### {template + 10}",
            }
            for template in range(6)
            for instance in range(4)
        ]
    )
    pairs = build_candidate_pairs(examples, CharacterTokenizer(), seed=2)
    assert len(pairs) == 3
    groups = [e.group for a, b, _, _ in pairs for e in (a, b)]
    assert len(groups) == len(set(groups))


def test_canonical_crux_candidate_excludes_prompt_known_assertion_half():
    a = crux("def f(x):\n    return x + 1", "7", "8", "sample")
    b = crux("def f(x):\n    return x + 1", "7", "8", "sample", "output")
    assert canonical_candidate(a) == "f(7)"
    assert canonical_candidate(b) == "8"


def test_alternative_valid_input_cannot_be_used_as_a_negative_or_corruption():
    a = crux("def f(x):\n    return abs(x)", "3", "3", "a")
    b = crux("def f(x):\n    return abs(x)", "-3", "3", "b")
    assert not valid_corruption(a, b)
    assert not valid_corruption(b, a)
    assert build_candidate_pairs([a, b], CharacterTokenizer()) == []


def test_runtime_errors_cannot_be_used_as_easy_negative_inputs():
    a = crux("def f(x):\n    return 1 / x", "1", "1.0", "a")
    b = crux("def f(x):\n    return x + 1", "0", "1", "b")
    assert not valid_corruption(b, a)  # f(0) raises on source a.
    assert build_candidate_pairs([a, b], CharacterTokenizer()) == []


def test_input_arity_and_output_python_types_must_match():
    a = crux("def f(x):\n    return x", "1", "1", "a")
    b = crux("def f(x,y):\n    return x+y", "1, 2", "3", "b")
    assert candidate_signature(a, canonical_candidate(a)) != candidate_signature(
        b, canonical_candidate(b)
    )
    integer = crux("def f(x):\n    return x", "123", "123", "integer", "output")
    string = crux("def f(x):\n    return x", "'a'", "'a'", "string", "output")
    assert len(canonical_candidate(integer)) == len(canonical_candidate(string))
    assert build_candidate_pairs([integer, string], CharacterTokenizer()) == []


def test_wrong_benchmark_reference_is_excluded_from_probe_pool():
    a = crux("def f(x):\n    return x", "1", "9", "a")
    b = crux("def f(x):\n    return x", "2", "2", "b")
    assert build_candidate_pairs([a, b], CharacterTokenizer()) == []


def test_well_formed_wrong_input_reciprocal_pair_is_retained():
    a = crux("def f(x):\n    return x+1", "1", "2", "a")
    b = crux("def f(x):\n    return x+1", "2", "3", "b")
    assert len(build_candidate_pairs([a, b], CharacterTokenizer())) == 1
