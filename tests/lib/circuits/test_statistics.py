"""Offline mathematical checks for node ranks and clustered uncertainty."""

import itertools
import json

import numpy as np
import pytest

from geode.circuits.statistics import (
    jaccard,
    paired_metric_ci,
    paired_overlap_ci,
    random_jaccard_baseline,
    split_half_stability,
    topk_nodes,
)


def test_ranking_is_absolute_mean_signed_not_mean_absolute():
    scores = np.array([[100, 3, -4], [-100, 3, -4]])
    assert topk_nodes(scores, ["canceled", "positive", "negative"], 2) == ["negative", "positive"]


def test_ranking_ties_use_node_name_and_ignore_column_order():
    names = ["head.9", "head.1", "mlp.0"]
    scores = np.array([2, -2, 1])
    assert topk_nodes(scores, names, 2) == ["head.1", "head.9"]
    assert topk_nodes(scores[::-1], names[::-1], 2) == ["head.1", "head.9"]


def test_jaccard_is_symmetric_set_overlap():
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert jaccard(["a", "a"], ["a"]) == 1
    assert jaccard([], []) == 1
    assert jaccard(["a"], []) == 0


def test_random_null_matches_exact_enumerated_distribution():
    n, k = 6, 2
    sets = list(itertools.combinations(map(str, range(n)), k))
    exact_mean = np.mean([jaccard(a, b) for a in sets for b in sets])
    result = random_jaccard_baseline(n, k, n_draws=100_000, seed=11)
    assert result["mean"] == pytest.approx(exact_mean, abs=0.003)
    assert result["mean_intersection"] == pytest.approx(k * k / n, abs=0.01)
    # E[J] is not E[intersection] / E[union].
    assert abs(result["mean"] - (k * k / n) / (2 * k - k * k / n)) > 0.025


def test_null_full_universe_and_seed_reproducibility():
    assert random_jaccard_baseline(16, 16)["reference_interval_95"] == [1.0, 1.0]
    assert random_jaccard_baseline(50, 16, seed=2) == random_jaccard_baseline(50, 16, seed=2)


def test_paired_bootstrap_cancels_common_noise_exactly():
    rng = np.random.default_rng(8)
    a = rng.normal(size=80)
    result = paired_metric_ci(a, a + 2, np.repeat(np.arange(20), 4), n_bootstrap=200)
    assert result["difference_b_minus_a"] == pytest.approx(2)
    assert result["ci_difference"] == pytest.approx([2, 2])
    assert result["ci_a"][1] > result["ci_a"][0]


def test_repeated_variants_do_not_artificially_shrink_metric_uncertainty():
    a = np.arange(20, dtype=float)
    groups = list(map(str, range(20)))
    original = paired_metric_ci(a, a**2, groups, n_bootstrap=200)
    replicated = paired_metric_ci(
        np.repeat(a, 8), np.repeat(a**2, 8), np.repeat(groups, 8), n_bootstrap=200
    )
    for key in ("ci_a", "ci_b", "ci_difference", "mean_a", "mean_b"):
        assert replicated[key] == original[key]
    assert replicated["n_groups"] == original["n_groups"] == 20


def test_uneven_group_sizes_still_receive_equal_weight():
    # Nine copies of one source do not outvote the other independent sources.
    a = np.array([1.0] * 9 + [0, 0, 0])
    result = paired_metric_ci(a, a, ["a"] * 9 + ["b", "c", "d"], n_bootstrap=100)
    assert result["mean_a"] == 0.25


def test_ethics_strata_macro_average_ignores_task_size():
    a = np.array([1.0] * 4 + [0.0] * 12)
    result = paired_metric_ci(
        a, a, np.arange(16), strata=["small"] * 4 + ["large"] * 12, n_bootstrap=100
    )
    assert result["mean_a"] == 0.5


def test_group_reordering_preserves_bootstrap_and_overlap():
    rng = np.random.default_rng(9)
    a, b = rng.normal(size=(2, 24, 8))
    groups = np.repeat(np.arange(12), 2)
    order = rng.permutation(len(groups))
    names = list(map(str, range(8)))
    original = paired_overlap_ci(a, b, names, groups, k=3, n_bootstrap=100)
    reordered = paired_overlap_ci(a[order], b[order], names, groups[order], k=3, n_bootstrap=100)
    assert original == reordered


def test_top_membership_is_recomputed_inside_bootstrap():
    # Full-set score has an exact tie; resamples must break it differently.
    a = np.array([[4, 0]] * 4 + [[0, 4]] * 4, dtype=float)
    b = np.array([[4, 0]] * 8, dtype=float)
    result = paired_overlap_ci(a, b, ["a", "b"], np.arange(8), k=1, n_bootstrap=200)
    assert result["jaccard"] == 1
    assert result["ci"] == [0, 1]


def test_identical_checkpoint_overlap_is_one_in_every_paired_resample():
    scores = np.random.default_rng(3).normal(size=(20, 40))
    result = paired_overlap_ci(
        scores, scores, list(map(str, range(40))), np.arange(20), n_bootstrap=100
    )
    assert result["jaccard"] == 1
    assert result["ci"] == [1, 1]
    json.dumps(result, allow_nan=False)


def test_stable_planted_circuit_has_perfect_split_half_reliability():
    scores = np.tile([8, -7, 0.1, 0], (40, 1))
    result = split_half_stability(scores, list("abcd"), np.repeat(np.arange(20), 2), k=2)
    assert result["split_jaccards"] == [1] * 20
    assert "not a strict ceiling" in result["interpretation"]


def test_split_half_keeps_replicated_variants_together():
    scores = np.random.default_rng(11).normal(size=(20, 8))
    groups = np.arange(20)
    a = split_half_stability(scores, list("abcdefgh"), groups, k=2)
    b = split_half_stability(
        np.repeat(scores, 8, axis=0), list("abcdefgh"), np.repeat(groups, 8), k=2
    )
    assert a == b


def test_rejects_insufficient_independent_groups_and_nonfinite_scores():
    with pytest.raises(ValueError, match="four independent"):
        paired_metric_ci(np.ones(80), np.ones(80), np.repeat([0, 1], 40))
    with pytest.raises(ValueError, match="finite"):
        topk_nodes(np.array([np.nan]), ["a"], 1)
    with pytest.raises(ValueError, match="uniquely"):
        topk_nodes(np.ones(2), ["a", "a"], 1)
    with pytest.raises(ValueError, match="multiple strata"):
        paired_metric_ci(np.ones(8), np.ones(8), np.repeat(np.arange(4), 2), strata=np.arange(8))
