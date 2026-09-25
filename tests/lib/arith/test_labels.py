"""geode.arith.labels — random-label rule (V5.6).

V5.6: random-label mode is statistically independent of the operands. Under the
OPEN(6) default the label depends on the operands *only* through the true
answer's digit count and sign, so the honest, non-flaky form of the property is
conditional independence: at a fixed position, any two same-shape true answers
get the same label regardless of which operands produced them.
"""

from __future__ import annotations

from collections import Counter

import pytest

from geode.arith.formats import digits, true_answer
from geode.arith.labels import cyclic_shift_labels, permute_labels, random_label


def test_v5_6_label_matches_true_answer_digit_count_and_sign():
    for ta in (7, 84, 512, 9999, -3, -22, -100):
        lab = random_label(ta, seed=0, index=0)
        assert digits(lab) == digits(ta)
        assert (lab < 0) == (ta < 0)


def test_v5_6_independent_of_operands_given_answer_shape():
    # Two different operand pairs, same true answer shape (3-digit positive),
    # at the same index -> identical label. The operands never enter beyond
    # digits+sign, so nothing about (a, b) leaks into the label.
    ta1 = true_answer(100, 400, "+")  # 500
    ta2 = true_answer(250, 250, "+")  # 500
    assert digits(ta1) == digits(ta2)
    for index in range(20):
        assert random_label(ta1, seed=7, index=index) == random_label(ta2, seed=7, index=index)


def test_v5_6_labels_vary_across_positions_and_seeds():
    seq_a = [random_label(500, seed=1, index=i) for i in range(200)]
    seq_b = [random_label(500, seed=2, index=i) for i in range(200)]
    assert len(set(seq_a)) > 1  # not a constant
    assert seq_a != seq_b  # seed changes the sequence


def test_v5_6_distribution_roughly_uniform_over_digit_band():
    # 3-digit band is [100, 999]; sampling many positions should spread out.
    labels = [random_label(500, seed=3, index=i) for i in range(3000)]
    assert all(100 <= v <= 999 for v in labels)
    # Coverage of the hundreds bucket is broad (weak, deterministic sanity).
    buckets = Counter(v // 100 for v in labels)
    assert len(buckets) == 9  # buckets 1..9 all hit


def test_v5_64_permuted_labels_preserve_marginal_exactly():
    # Duplicates and negatives included: the shown-label multiset must equal
    # the true-answer multiset exactly (the shape prior is preserved by
    # construction, not approximately).
    answers = [7, 84, -3, 500, 500, -22, 9999, 100, -100, 7]
    out = permute_labels(answers, seed=11)
    assert sorted(out) == sorted(answers)


def test_v5_64_permutation_deterministic_and_seed_sensitive():
    answers = list(range(-100, 100))
    assert permute_labels(answers, seed=4) == permute_labels(answers, seed=4)
    assert permute_labels(answers, seed=4) != permute_labels(answers, seed=5)


def test_v5_64_mapping_destroyed_on_distinct_answers():
    # With all-distinct answers, a label is correct only at a fixed point of
    # the permutation. E[fixed points] = 1 for a uniform shuffle; 25 would
    # mean a broken rng, not bad luck (seeded, deterministic).
    answers = list(range(1, 501))
    out = permute_labels(answers, seed=5)
    assert sorted(out) == sorted(answers)
    assert sum(a == o for a, o in zip(answers, out)) < 25


def test_v5_78_cyclic_shift_zero_fixed_points_on_distinct_values():
    for n in (2, 3, 10, 100):
        answers = list(range(n))  # all distinct -> rotation can never collide
        out = cyclic_shift_labels(answers)
        assert all(o != a for o, a in zip(out, answers))


def test_v5_78_cyclic_shift_preserves_multiset():
    # Duplicates included (1/2/3 each appear twice), chosen so the rotation
    # itself doesn't collide -- collision behavior has its own test below.
    answers = [1, 2, 3, 1, 2, 3]
    out = cyclic_shift_labels(answers)
    assert sorted(out) == sorted(answers)


def test_v5_78_cyclic_shift_is_a_rotation_by_one():
    answers = [10, 20, 30, 40]
    assert cyclic_shift_labels(answers) == [20, 30, 40, 10]


def test_v5_78_cyclic_shift_refuses_n_below_2():
    with pytest.raises(ValueError):
        cyclic_shift_labels([])
    with pytest.raises(ValueError):
        cyclic_shift_labels([42])


def test_v5_78_cyclic_shift_raises_on_duplicate_value_collision():
    # n=2 with both positions sharing a value: rotation necessarily leaves
    # each position showing its own value (multiset is {5, 5}). Must raise,
    # not silently ship a leaky label.
    with pytest.raises(ValueError):
        cyclic_shift_labels([5, 5])


def test_v5_78_cyclic_shift_not_seeded():
    # No seed parameter at all -- same input always gives the same output,
    # there is nothing to vary across calls.
    answers = [1, 2, 3, 4, 5]
    assert cyclic_shift_labels(answers) == cyclic_shift_labels(answers)
