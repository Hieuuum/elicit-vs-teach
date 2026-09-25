"""Offline leakage, separability and null properties of grouped linear probes."""

import json

import numpy as np
import pytest

from geode.circuits.probes import (
    _shuffle_group_labels,
    evaluate_layerwise_probes,
    evaluate_linear_probe,
    grouped_split,
    standardize_from_train,
)


def paired_fixture(n_groups=100, hidden=12, seed=7):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), 2)
    labels = np.tile([0, 1], n_groups)
    features = rng.normal(size=(len(labels), hidden))
    features[:, 0] = (2 * labels - 1) * 4 + rng.normal(scale=0.1, size=len(labels))
    return features, labels, groups


def test_group_split_has_exact_ratio_and_no_candidate_or_variant_leakage():
    _, labels, groups = paired_fixture()
    split = grouped_split(groups, labels, seed=14)
    assert {key: len(rows) for key, rows in split.items()} == {
        "train": 120,
        "validation": 40,
        "test": 40,
    }
    sets = [set(groups[rows]) for rows in split.values()]
    assert not sets[0] & sets[1] and not sets[1] & sets[2] and not sets[0] & sets[2]
    assert len(set.union(*sets)) == 100


def test_group_split_is_invariant_to_input_order():
    _, labels, groups = paired_fixture()
    perm = np.random.default_rng(5).permutation(len(labels))
    a = grouped_split(groups, labels, seed=8)
    b = grouped_split(groups[perm], labels[perm], seed=8)
    assert all(set(groups[a[key]]) == set(groups[perm][b[key]]) for key in a)


def test_planted_semantic_direction_generalizes_and_shuffle_control_falls():
    x, y, groups = paired_fixture(n_groups=200)
    result = evaluate_linear_probe(x, y, groups, seed=5, n_shuffles=8)
    assert result["test_accuracy"] > 0.98
    assert 0.3 < result["shuffled_test_accuracy_mean"] < 0.7
    assert result["majority_test_accuracy"] == 0.5
    json.dumps(result, allow_nan=False)


def test_random_features_and_labels_remain_near_chance():
    rng = np.random.default_rng(33)
    x = rng.normal(size=(1000, 10))
    y = np.tile([0, 1], 500)
    result = evaluate_linear_probe(x, y, np.repeat(np.arange(500), 2), seed=2, n_shuffles=2)
    assert 0.4 < result["test_accuracy"] < 0.6


def test_standardization_never_uses_validation_or_test_moments():
    train = np.array([[0, 4], [2, 4]], dtype=float)
    val = np.array([[100, 100]], dtype=float)
    test = np.array([[-100, -100]], dtype=float)
    a, b, c, mean, scale = standardize_from_train(train, val, test)
    np.testing.assert_array_equal(mean, [1, 4])
    np.testing.assert_array_equal(scale, [1, 1])
    np.testing.assert_array_equal(a, [[-1, 0], [1, 0]])
    np.testing.assert_array_equal(b, [[99, 96]])
    np.testing.assert_array_equal(c, [[-101, -104]])


def test_test_feature_changes_cannot_change_scaling_or_selected_alpha():
    x, y, groups = paired_fixture()
    split = grouped_split(groups, y, seed=6)
    original = evaluate_linear_probe(x, y, groups, seed=6, n_shuffles=1, splits=split)
    shifted = x.copy()
    shifted[split["test"]] += 1000
    changed = evaluate_linear_probe(shifted, y, groups, seed=6, n_shuffles=1, splits=split)
    for key in ("training_mean", "training_scale", "regularization", "validation_accuracy"):
        assert changed[key] == original[key]


def test_test_label_changes_cannot_select_alpha_or_fit_scaling():
    x, y, groups = paired_fixture()
    split = grouped_split(groups, y, seed=6)
    original = evaluate_linear_probe(x, y, groups, n_shuffles=1, splits=split)
    changed_y = y.copy()
    changed_y[split["test"]] = 1 - changed_y[split["test"]]
    changed = evaluate_linear_probe(x, changed_y, groups, n_shuffles=1, splits=split)
    assert original["regularization"] == changed["regularization"]
    assert original["predictions"] == changed["predictions"]
    assert original["test_accuracy"] + changed["test_accuracy"] == 1


def test_train_scaling_is_invariant_to_positive_feature_units():
    x, y, groups = paired_fixture()
    a = evaluate_linear_probe(x, y, groups, n_shuffles=1)
    b = evaluate_linear_probe(x * np.arange(1, x.shape[1] + 1) + 42, y, groups, n_shuffles=1)
    assert a["predictions"] == b["predictions"]
    assert a["regularization"] == b["regularization"]


def test_ridge_dual_form_recovers_direction_when_width_exceeds_rows():
    x, y, groups = paired_fixture(n_groups=50, hidden=100)
    # All dimensions redundantly carry the same planted signal.
    x[:] = (2 * y[:, None] - 1) * 4 + np.random.default_rng(2).normal(scale=0.1, size=x.shape)
    result = evaluate_linear_probe(x, y, groups, n_shuffles=1)
    assert result["test_accuracy"] == 1


def test_layerwise_probes_share_splits_and_detect_only_planted_layer():
    x, y, groups = paired_fixture(n_groups=200)
    noise = np.random.default_rng(5).normal(size=x.shape)
    result = evaluate_layerwise_probes(np.stack([noise, x], axis=1), y, groups, n_shuffles=1)
    first, second = result["layers"]
    assert first["split_indices"] == second["split_indices"]
    assert 0.3 < first["test_accuracy"] < 0.7
    assert second["test_accuracy"] == 1


def test_shuffled_ethics_variants_keep_one_label_per_source():
    groups = np.repeat(np.arange(40), 8)
    y = np.repeat(np.tile([0, 1], 20), 8)
    shuffled = _shuffle_group_labels(y, groups, np.random.default_rng(5))
    assert all(len(set(shuffled[groups == group])) == 1 for group in range(40))
    assert shuffled.sum() == y.sum()
    assert not np.array_equal(shuffled, y)


def test_shuffled_candidate_pairs_keep_balanced_labels_but_destroy_association():
    groups = np.repeat(np.arange(100), 2)
    y = np.tile([0, 1], 100)
    shuffled = _shuffle_group_labels(y, groups, np.random.default_rng(5))
    assert all(shuffled[groups == group].sum() == 1 for group in range(100))
    assert 0.35 < (shuffled == y).mean() < 0.65


def test_rejects_impossible_class_coverage_and_group_leakage():
    with pytest.raises(ValueError, match="three source groups"):
        grouped_split(np.arange(20), [1] + [0] * 19)
    x, y, groups = paired_fixture()
    split = {
        "train": np.arange(120),
        "validation": np.arange(120, 160),
        "test": np.arange(160, 200),
    }
    split["train"][-1], split["validation"][0] = 120, 119
    with pytest.raises(ValueError, match="leak"):
        evaluate_linear_probe(x, y, groups, splits=split)


def test_multiclass_probe_and_constant_features_are_well_defined():
    groups = np.repeat(np.arange(100), 3)
    y = np.tile(np.arange(3), 100)
    x = np.column_stack([np.eye(3)[y], np.ones(len(y))])
    result = evaluate_linear_probe(x, y, groups, n_shuffles=1)
    assert result["test_accuracy"] == 1
    assert result["classes"] == [0, 1, 2]
    assert result["training_scale"][-1] == 1


def test_hyperparameter_ties_prefer_more_regularization():
    x, y, groups = paired_fixture()
    result = evaluate_linear_probe(x, y, groups, regularizations=(0.01, 0.1, 1), n_shuffles=1)
    assert result["regularization"] == 1


@pytest.mark.parametrize("hidden", [5, 60])
def test_ridge_matches_independent_augmented_normal_equations(hidden):
    x, y, groups = paired_fixture(n_groups=30, hidden=hidden)
    split = grouped_split(groups, y, seed=4)
    alpha = 0.3
    result = evaluate_linear_probe(
        x, y, groups, splits=split, regularizations=(alpha,), n_shuffles=1
    )
    train, val, test, _, _ = standardize_from_train(
        x[split["train"]], x[split["validation"]], x[split["test"]]
    )
    design = np.column_stack([np.ones(len(train)), train])
    penalty = np.eye(hidden + 1) * alpha
    penalty[0, 0] = 0  # Intercept is unpenalized.
    targets = np.eye(2)[y[split["train"]]]
    coefficients = np.linalg.solve(
        design.T @ design / len(train) + penalty, design.T @ targets / len(train)
    )
    predicted = (np.column_stack([np.ones(len(test)), test]) @ coefficients).argmax(axis=1)
    val_predicted = (np.column_stack([np.ones(len(val)), val]) @ coefficients).argmax(axis=1)
    assert result["predictions"] == predicted.tolist()
    assert result["validation_accuracy"] == (val_predicted == y[split["validation"]]).mean()
