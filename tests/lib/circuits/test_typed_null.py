"""Mathematical checks for node-type-preserving circuit overlap sensitivity."""

from itertools import combinations, product

import numpy as np
import pytest

from geode.circuits.statistics import (
    jaccard,
    node_type_counts,
    node_type_jaccard_baseline,
    random_jaccard_baseline,
    sample_type_matched_nodes,
)


def universe(n_mlp, n_heads):
    return [f"layer.{i}.mlp" for i in range(n_mlp)] + [f"layer.0.attn.{i}" for i in range(n_heads)]


def test_typed_sampling_matches_exhaustive_small_universe_not_ratio_of_expectations():
    mlps, heads = universe(3, 0), universe(0, 4)
    options_a = [
        tuple(a) + tuple(b) for a, b in product(combinations(mlps, 2), combinations(heads, 1))
    ]
    options_b = [
        tuple(a) + tuple(b) for a, b in product(combinations(mlps, 1), combinations(heads, 2))
    ]
    exact = [jaccard(a, b) for a, b in product(options_a, options_b)]
    result = node_type_jaccard_baseline(
        options_a[0], options_b[0], mlps + heads, n_draws=100_000, seed=17
    )
    assert result["mean"] == pytest.approx(np.mean(exact), abs=0.003)
    assert result["std"] == pytest.approx(np.std(exact), abs=0.003)
    assert result["expected_intersection"] == pytest.approx(2 / 3 + 2 / 4)
    assert result["mean_intersection"] == pytest.approx(result["expected_intersection"], abs=0.005)
    naive = result["expected_intersection"] / (6 - result["expected_intersection"])
    assert abs(result["mean"] - naive) > 0.01


def test_one_type_exactly_reduces_to_existing_uniform_baseline():
    names = [f"node-{i}" for i in range(100)]
    typed = node_type_jaccard_baseline(names[:16], names[-16:], names, seed=8)
    uniform = random_jaccard_baseline(100, 16, seed=8)
    for field in ("mean", "std", "reference_interval_95", "mean_intersection"):
        assert typed[field] == uniform[field]
    assert typed["universe_type_counts"] == {"other": 100}


def test_whole_mlp_concentration_can_force_overlap_without_task_specificity():
    names = universe(16, 256)
    result = node_type_jaccard_baseline(names[:16], names[:16], names)
    assert result["mean"] == 1
    assert result["reference_interval_95"] == [1, 1]
    assert result["type_counts_a"] == {"attention_head": 0, "mlp": 16}
    assert random_jaccard_baseline(len(names), 16)["mean"] < 0.04


def test_disjoint_node_types_force_zero_overlap_even_with_unequal_set_sizes():
    names = universe(4, 8)
    result = node_type_jaccard_baseline(names[:3], names[4:6], names)
    assert result["k_a"] == 3 and result["k_b"] == 2
    assert result["mean"] == 0
    assert result["reference_interval_95"] == [0, 0]


def test_null_preserves_counts_not_specific_node_identities_and_is_order_invariant():
    names = universe(4, 8)
    first = node_type_jaccard_baseline(
        names[:2] + names[4:6], names[1:3] + names[6:8], names, seed=3
    )
    other = node_type_jaccard_baseline(
        names[2:4] + names[8:10], names[:2] + names[10:12], names[::-1], seed=3
    )
    assert first == other


def test_empty_selection_conventions_are_finite():
    names = universe(2, 4)
    assert node_type_jaccard_baseline([], [], names)["mean"] == 1
    assert node_type_jaccard_baseline([], names[:2], names)["mean"] == 0


@pytest.mark.parametrize(
    "a,b,names",
    [
        (["layer.0.mlp", "layer.0.mlp"], [], universe(2, 4)),
        (["absent"], [], universe(2, 4)),
        ([], [], ["duplicate", "duplicate"]),
        ([], [], []),
    ],
)
def test_invalid_node_selection_cannot_change_null_population_silently(a, b, names):
    with pytest.raises(ValueError):
        node_type_jaccard_baseline(a, b, names)


def test_intervention_sampler_preserves_size_type_counts_and_uniqueness():
    names = universe(16, 256)
    reference = names[:10] + names[16:22]
    for seed in range(100):
        sampled = sample_type_matched_nodes(reference, names, seed=seed)
        assert len(sampled) == len(set(sampled)) == len(reference)
        assert set(sampled).issubset(names)
        assert node_type_counts(sampled) == {"attention_head": 6, "mlp": 10}


def test_intervention_sampler_is_seeded_order_invariant_and_does_not_exclude_reference():
    names = universe(3, 4)
    reference = names[:2] + names[3:5]
    assert sample_type_matched_nodes(reference, names, seed=12) == sample_type_matched_nodes(
        reference[::-1], names[::-1], seed=12
    )
    assert sample_type_matched_nodes(names[:3], names, seed=12) == sorted(names[:3])
    assert sample_type_matched_nodes([], names, seed=12) == []
    assert (
        len({tuple(sample_type_matched_nodes(reference, names, seed=seed)) for seed in range(20)})
        > 1
    )


def test_intervention_sampler_is_uniform_within_types_in_tiny_exhaustive_space():
    names = universe(2, 4)
    reference = [names[0], names[2]]
    possibilities = {tuple(sorted(pair)) for pair in product(names[:2], names[2:])}
    counts = {pair: 0 for pair in possibilities}
    for seed in range(4096):
        counts[tuple(sample_type_matched_nodes(reference, names, seed=seed))] += 1
    # All 2*4 equally likely sets are represented near their expectation 512.
    assert set(counts) == possibilities
    assert all(abs(count - 512) < 85 for count in counts.values())


@pytest.mark.parametrize(
    "reference,names",
    [
        (["layer.0.mlp", "layer.0.mlp"], universe(2, 4)),
        (["missing"], universe(2, 4)),
        ([], ["same", "same"]),
    ],
)
def test_intervention_sampler_rejects_ambiguous_or_foreign_reference_nodes(reference, names):
    with pytest.raises(ValueError):
        sample_type_matched_nodes(reference, names)
