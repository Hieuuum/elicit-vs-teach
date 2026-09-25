"""CPU linear diagnostics with grouped held-out evaluation and leakage controls."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from geode.circuits.statistics import _finite_array, _group_keys


def grouped_split(
    group_ids: Sequence[Any], labels: Sequence[Any], *, seed: int = 0
) -> dict[str, np.ndarray]:
    """Deterministic 60/20/20 source-group split containing all classes per split.

    Sorting source identifiers before randomization makes assignments invariant
    to row order. Label coverage is a feasibility guard, never feature-based
    split selection. Integer rounding gives the remainder to training.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional vector")
    unique, inverse = _group_keys(group_ids, len(labels))
    classes = np.unique(labels)
    if len(unique) < 10 or len(classes) < 2:
        raise ValueError("Need at least ten source groups and two label classes")
    for label in classes:
        if len(np.unique(inverse[labels == label])) < 3:
            raise ValueError("Every class must occur in at least three source groups")
    n_eval = len(unique) // 5
    rng = np.random.default_rng(seed)
    for _ in range(512):
        shuffled = rng.permutation(len(unique))
        assignments = {
            "train": shuffled[2 * n_eval :],
            "validation": shuffled[:n_eval],
            "test": shuffled[n_eval : 2 * n_eval],
        }
        indices = {key: np.flatnonzero(np.isin(inverse, ids)) for key, ids in assignments.items()}
        if all(len(np.unique(labels[rows])) == len(classes) for rows in indices.values()):
            return indices
    raise ValueError("Could not form grouped splits with every class; expand the diagnostic pool")


def standardize_from_train(
    train: np.ndarray, validation: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform all splits using training moments only; constant features have scale 1."""
    train, validation, test = (_finite_array(x, 2) for x in (train, validation, test))
    if train.shape[1] != validation.shape[1] or train.shape[1] != test.shape[1]:
        raise ValueError("All feature matrices must have the same width")
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    return (train - mean) / scale, (validation - mean) / scale, (test - mean) / scale, mean, scale


def _group_accuracy(predicted: np.ndarray, labels: np.ndarray, groups: Sequence[Any]) -> float:
    unique, inverse = _group_keys(groups, len(labels))
    correct = predicted == labels
    return float(np.mean([correct[inverse == i].mean() for i in range(len(unique))]))


def _shuffle_group_labels(
    labels: np.ndarray, group_ids: Sequence[Any], rng: np.random.Generator
) -> np.ndarray:
    """Preserve homogeneous variant groups; randomize candidate labels within pairs.

    Homogeneous groups exchange labels, so replicated control renderings never
    become independent shuffled examples. Mixed-label groups shuffle their
    label vector internally, appropriate for paired valid/invalid candidates.
    """
    unique, inverse = _group_keys(group_ids, len(labels))
    result = labels.copy()
    homogeneous = []
    for i in range(len(unique)):
        rows = np.flatnonzero(inverse == i)
        if len(np.unique(labels[rows])) == 1:
            homogeneous.append(i)
        else:
            result[rows] = rng.permutation(labels[rows])
    values = [labels[np.flatnonzero(inverse == i)[0]] for i in homogeneous]
    for i, label in zip(homogeneous, rng.permutation(values), strict=True):
        result[inverse == i] = label
    return result


def evaluate_linear_probe(
    features: np.ndarray,
    labels: Sequence[Any],
    group_ids: Sequence[Any],
    *,
    seed: int = 0,
    regularizations: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    n_shuffles: int = 5,
    splits: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Tune a ridge linear classifier on validation groups, then evaluate once.

    Loss is mean squared one-hot residual plus alpha*||W||² with an unpenalized
    intercept. Train examples have equal group weight. The final classifier
    remains fit on TRAIN ONLY (no train+validation refit), ensuring identical
    standardization and training semantics across layers. Validation selects
    alpha by group accuracy; ties choose the larger alpha. Test labels never
    enter preprocessing or hyperparameter selection. Shuffled-label controls
    refit and tune on the same split; test truth remains unchanged.
    """
    features = _finite_array(features, 2)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("labels must have one entry per feature row")
    group_ids = list(group_ids)
    _group_keys(group_ids, len(features))
    classes, encoded = np.unique(labels, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("At least two classes are required")
    alphas = sorted(set(float(a) for a in regularizations), reverse=True)
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("regularizations must contain finite positive values")
    if n_shuffles < 1:
        raise ValueError("At least one shuffled-label control is required")
    split = grouped_split(group_ids, encoded, seed=seed) if splits is None else splits
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("splits must contain train, validation and test")
    flat = np.concatenate([np.asarray(split[key], dtype=int) for key in split])
    if len(flat) != len(features) or sorted(flat.tolist()) != list(range(len(features))):
        raise ValueError("Supplied splits must partition every example exactly once")
    # Canonical keys avoid treating string '1' and integer 1 as the same source.
    _, inverse = _group_keys(group_ids, len(features))
    group_sets = [set(inverse[split[key]]) for key in ("train", "validation", "test")]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Source groups leak across supplied splits")
    if any(len(groups) < 2 for groups in group_sets):
        raise ValueError("Each split requires at least two independent source groups")
    if any(len(np.unique(encoded[split[key]])) != len(classes) for key in split):
        raise ValueError("Every split must contain all label classes")
    train_idx, val_idx, test_idx = (
        np.asarray(split[key], dtype=int) for key in ("train", "validation", "test")
    )
    train, val, test, mean, scale = standardize_from_train(
        features[train_idx], features[val_idx], features[test_idx]
    )
    train_groups = [group_ids[i] for i in train_idx]
    val_groups = [group_ids[i] for i in val_idx]
    test_groups = [group_ids[i] for i in test_idx]
    unique_train, inv_train = _group_keys(train_groups, len(train_idx))
    counts = np.bincount(inv_train)
    weights = 1.0 / (len(unique_train) * counts[inv_train])
    # Weighted centering makes the fitted intercept unpenalized, even when
    # variable-size source groups give the standardized train a nonzero mean.
    center = weights @ train
    centered = train - center
    weighted = centered * np.sqrt(weights)[:, None]
    # Work in the smaller of sample and feature dimensions, one factorization
    # shared by all alphas and shuffled baselines.
    dual = weighted.shape[0] < weighted.shape[1]
    gram = weighted @ weighted.T if dual else weighted.T @ weighted
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0)

    def fit(train_labels: np.ndarray) -> tuple[float, float, np.ndarray]:
        targets = np.eye(len(classes))[train_labels]
        intercept = weights @ targets
        response = (targets - intercept) * np.sqrt(weights)[:, None]
        rhs = eigenvectors.T @ (response if dual else weighted.T @ response)
        best: tuple[float, float, np.ndarray] | None = None
        for alpha in alphas:
            solution = eigenvectors @ (rhs / (eigenvalues[:, None] + alpha))
            coefficients = weighted.T @ solution if dual else solution
            val_prediction = ((val - center) @ coefficients + intercept).argmax(axis=1)
            accuracy = _group_accuracy(val_prediction, encoded[val_idx], val_groups)
            if best is None or accuracy > best[1]:
                test_prediction = ((test - center) @ coefficients + intercept).argmax(axis=1)
                best = (alpha, accuracy, test_prediction)
        assert best is not None
        return best

    alpha, val_accuracy, predicted = fit(encoded[train_idx])
    rng = np.random.default_rng(seed)
    shuffled = []
    for _ in range(n_shuffles):
        shuffled_labels = _shuffle_group_labels(encoded[train_idx], train_groups, rng)
        _, _, null_predicted = fit(shuffled_labels)
        shuffled.append(_group_accuracy(null_predicted, encoded[test_idx], test_groups))
    majority = int(np.argmax(weights @ np.eye(len(classes))[encoded[train_idx]]))
    return {
        "test_accuracy": _group_accuracy(predicted, encoded[test_idx], test_groups),
        "validation_accuracy": val_accuracy,
        "regularization": alpha,
        "regularizations": alphas,
        "shuffled_test_accuracies": shuffled,
        "shuffled_test_accuracy_mean": float(np.mean(shuffled)),
        "majority_test_accuracy": _group_accuracy(
            np.full(len(test_idx), majority), encoded[test_idx], test_groups
        ),
        "classes": classes.tolist(),
        "predictions": classes[predicted].tolist(),
        "test_labels": labels[test_idx].tolist(),
        "test_group_ids": [g.item() if isinstance(g, np.generic) else g for g in test_groups],
        "split_indices": {key: np.asarray(rows).tolist() for key, rows in split.items()},
        "split_n_examples": {key: len(rows) for key, rows in split.items()},
        "split_n_groups": dict(
            zip(("train", "validation", "test"), map(len, group_sets), strict=True)
        ),
        "training_mean": mean.tolist(),
        "training_scale": scale.tolist(),
        "seed": seed,
        "classifier": "ridge one-hot linear classifier with unpenalized intercept",
        "weighting": "equal source groups",
        "control": "shuffle training labels within mixed groups / across homogeneous groups; unchanged heldout truth",
    }


def evaluate_layerwise_probes(
    features: np.ndarray,
    labels: Sequence[Any],
    group_ids: Sequence[Any],
    *,
    seed: int = 0,
    regularizations: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    n_shuffles: int = 5,
) -> dict[str, Any]:
    """Evaluate [example, layer, hidden] activations with identical source splits."""
    features = _finite_array(features, 3)
    split = grouped_split(group_ids, labels, seed=seed)
    return {
        "layers": [
            {
                "layer": layer,
                **evaluate_linear_probe(
                    features[:, layer, :],
                    labels,
                    group_ids,
                    seed=seed,
                    regularizations=regularizations,
                    n_shuffles=n_shuffles,
                    splits=split,
                ),
            }
            for layer in range(features.shape[1])
        ],
        "seed": seed,
    }
