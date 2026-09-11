"""Grouped uncertainty and node-overlap diagnostics for checkpoint comparisons.

The sampling unit is a source problem, template, or native ETHICS group. All
rows in a group stay together; groups receive equal weight. Optional strata
receive equal weight (e.g. ETHICS tasks), with resampling inside each stratum.
Intervals describe dataset sampling uncertainty, never training-seed variation.
"""

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np


def _group_keys(group_ids: Sequence[Any], n: int) -> tuple[list[str], np.ndarray]:
    if len(group_ids) != n:
        raise ValueError("group_ids must have one entry per example")
    keys = [f"{type(g).__name__}:{g!r}" for g in group_ids]
    unique = sorted(set(keys))
    lookup = {key: i for i, key in enumerate(unique)}
    return unique, np.array([lookup[key] for key in keys], dtype=int)


def _finite_array(values: Any, ndim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim or not array.size or not np.isfinite(array).all():
        raise ValueError(f"Expected nonempty finite {ndim}-dimensional array")
    return array


def _groups(
    values: np.ndarray, group_ids: Sequence[Any], strata: Sequence[Any] | None
) -> tuple[np.ndarray, list[np.ndarray]]:
    unique, inverse = _group_keys(group_ids, len(values))
    if len(unique) < 4:
        raise ValueError("At least four independent source groups are required")
    means = np.stack([values[inverse == i].mean(axis=0) for i in range(len(unique))])
    if strata is None:
        return means, [np.arange(len(unique))]
    _, row_strata = _group_keys(strata, len(values))
    group_strata = []
    for i in range(len(unique)):
        candidates = np.unique(row_strata[inverse == i])
        if len(candidates) != 1:
            raise ValueError("A source group cannot span multiple strata")
        group_strata.append(candidates[0])
    group_strata = np.array(group_strata)
    partitions = [np.flatnonzero(group_strata == s) for s in np.unique(group_strata)]
    if any(len(partition) < 4 for partition in partitions):
        raise ValueError("Each stratum needs at least four independent source groups")
    return means, partitions


def _aggregate(means: np.ndarray, partitions: list[np.ndarray]) -> np.ndarray:
    return np.stack([means[indices].mean(axis=0) for indices in partitions]).mean(axis=0)


def _resample(partitions: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    return [rng.choice(indices, size=len(indices), replace=True) for indices in partitions]


def _interval(values: Sequence[float], confidence: float) -> list[float]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    return np.quantile(values, [(1 - confidence) / 2, (1 + confidence) / 2]).tolist()


def topk_nodes(scores: np.ndarray, node_names: Sequence[str], k: int = 16) -> list[str]:
    """Rank abs(mean(signed scores)), breaking ties lexicographically by name."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 2:
        scores = _finite_array(scores, 2).mean(axis=0)
    else:
        scores = _finite_array(scores, 1)
    names = list(node_names)
    if len(names) != len(scores) or len(set(names)) != len(names):
        raise ValueError("Node names must uniquely identify every score column")
    if not all(isinstance(name, str) for name in names):
        raise ValueError("Node names must be strings")
    if not isinstance(k, int) or not 1 <= k <= len(names):
        raise ValueError("k must lie between one and the node universe size")
    return [
        names[i] for i in sorted(range(len(names)), key=lambda i: (-abs(scores[i]), names[i]))[:k]
    ]


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Set overlap; two empty sets have overlap one by convention."""
    left, right = set(a), set(b)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def random_jaccard_baseline(
    n_nodes: int, k: int = 16, *, n_draws: int = 100_000, seed: int = 0
) -> dict[str, Any]:
    """Exact sampling law of two independent uniform k-subsets.

    Their intersection is Hypergeometric(N, k, k). Its Jaccard distribution
    is sampled directly, avoiding an incorrect ratio-of-expectations shortcut.
    The reported interval is a null reference interval, not a mean's CI.
    """
    if not isinstance(k, int) or not isinstance(n_nodes, int) or not 1 <= k <= n_nodes:
        raise ValueError("Require integers 1 <= k <= n_nodes")
    if n_draws < 100:
        raise ValueError("Use at least 100 random draws")
    overlap = np.random.default_rng(seed).hypergeometric(k, n_nodes - k, k, size=n_draws)
    values = overlap / (2 * k - overlap)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "reference_interval_95": _interval(values, 0.95),
        "mean_intersection": float(overlap.mean()),
        "n_nodes": n_nodes,
        "k": k,
        "n_draws": n_draws,
        "seed": seed,
        "method": "hypergeometric independent uniform node subsets",
    }


def node_type(name: str) -> str:
    """Classify the declared node granularity; unfamiliar names are 'other'."""
    if not isinstance(name, str):
        raise ValueError("Node names must be strings")
    if name.endswith(".mlp"):
        return "mlp"
    if ".attn." in name:
        return "attention_head"
    return "other"


def node_type_counts(nodes: Sequence[str]) -> dict[str, int]:
    """Count declared node types, omitting types with zero selected nodes."""
    return dict(sorted(Counter(node_type(name) for name in nodes).items()))


def sample_type_matched_nodes(
    reference_nodes: Sequence[str], node_universe: Sequence[str], *, seed: int = 0
) -> list[str]:
    """Draw a uniform node set conditional on the reference set's type counts.

    Sampling is without replacement within each type. The reference set is
    eligible to be sampled: excluding it would change the intended null.
    Sorting the universe before sampling makes the seeded result invariant to
    input order. Returned names are lexical, not ranked by importance.
    """
    reference, universe = list(reference_nodes), list(node_universe)
    if (
        not universe
        or any(not isinstance(name, str) for name in reference + universe)
        or len(reference) != len(set(reference))
        or len(universe) != len(set(universe))
    ):
        raise ValueError("Node universe and reference must contain unique string names")
    if not set(reference).issubset(universe):
        raise ValueError("Reference nodes must belong to the declared node universe")
    counts = node_type_counts(reference)
    rng = np.random.default_rng(seed)
    sampled = []
    for kind, count in counts.items():
        candidates = sorted(name for name in universe if node_type(name) == kind)
        sampled.extend(rng.choice(candidates, size=count, replace=False).tolist())
    return sorted(sampled)


def node_type_jaccard_baseline(
    nodes_a: Sequence[str],
    nodes_b: Sequence[str],
    node_universe: Sequence[str],
    *,
    n_draws: int = 100_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Conditional random-overlap reference preserving each map's node-type counts.

    Whole MLP blocks and individual heads have different granularity; a shared
    tendency to rank whole MLPs highly can exceed the uniform-node baseline
    without task-specific reuse. Within each type, draw independent uniform
    subsets of the observed sizes at A and B. The total intersection is a sum
    of independent hypergeometrics. Transform EACH draw to Jaccard before
    averaging, avoiding a ratio-of-expectations approximation.

    OLMo names ending in '.mlp' identify whole MLP blocks; '.attn.' identifies
    attention heads. All remaining declared names form an explicit 'other'
    type, allowing this sensitivity to reduce to the ordinary uniform null.
    The interval is a conditional null reference range, not an uncertainty CI
    for observed overlap, and this sensitivity does not replace the primary
    uniform baseline or establish task specificity by itself.
    """
    left, right, universe = list(nodes_a), list(nodes_b), list(node_universe)
    if (
        not universe
        or any(not isinstance(name, str) for name in left + right + universe)
        or len(set(universe)) != len(universe)
        or len(set(left)) != len(left)
        or len(set(right)) != len(right)
    ):
        raise ValueError("Node universe and selections must contain unique string names")
    if not set(left).issubset(universe) or not set(right).issubset(universe):
        raise ValueError("Selected nodes must belong to the declared node universe")
    if not isinstance(n_draws, int) or n_draws < 100:
        raise ValueError("Use at least 100 integer random draws")

    types = sorted({node_type(name) for name in universe})
    population = {label: sum(node_type(name) == label for name in universe) for label in types}
    count_a = {label: sum(node_type(name) == label for name in left) for label in types}
    count_b = {label: sum(node_type(name) == label for name in right) for label in types}
    rng = np.random.default_rng(seed)
    overlap = np.zeros(n_draws, dtype=np.int64)
    for label in types:
        overlap += rng.hypergeometric(
            count_a[label], population[label] - count_a[label], count_b[label], size=n_draws
        )
    union = len(left) + len(right) - overlap
    values = np.divide(overlap, union, out=np.ones(n_draws), where=union != 0)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "reference_interval_95": _interval(values, 0.95),
        "mean_intersection": float(overlap.mean()),
        "expected_intersection": sum(count_a[t] * count_b[t] / population[t] for t in types),
        "n_nodes": len(universe),
        "k_a": len(left),
        "k_b": len(right),
        "type_counts_a": count_a,
        "type_counts_b": count_b,
        "universe_type_counts": population,
        "n_draws": n_draws,
        "seed": seed,
        "method": "independent uniform subsets within each observed node type",
        "interpretation": "conditional node-type-preserving null reference; not a sampling CI",
    }


def paired_overlap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    node_names: Sequence[str],
    group_ids: Sequence[Any],
    *,
    k: int = 16,
    n_bootstrap: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
    strata: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Paired cluster bootstrap, recomputing both top-k memberships every draw.

    Both score matrices must use the SAME example order, node order, and groups.
    The caller must join checkpoint records by stable item IDs before calling.
    """
    a, b = _finite_array(scores_a, 2), _finite_array(scores_b, 2)
    if a.shape != b.shape:
        raise ValueError("Paired checkpoints must have identical score shapes")
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    means, partitions = _groups(np.stack([a, b], axis=1), group_ids, strata)
    aggregate = _aggregate(means, partitions)
    top_a, top_b = (topk_nodes(row, node_names, k) for row in aggregate)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        sample = _aggregate(means, _resample(partitions, rng))
        draws.append(
            jaccard(topk_nodes(sample[0], node_names, k), topk_nodes(sample[1], node_names, k))
        )
    return {
        "jaccard": jaccard(top_a, top_b),
        "ci": _interval(draws, confidence),
        "top_a": top_a,
        "top_b": top_b,
        "n_groups": len(means),
        "n_examples": len(a),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "seed": seed,
        "weighting": "equal strata, equal source groups within stratum",
    }


def split_half_stability(
    scores: np.ndarray,
    node_names: Sequence[str],
    group_ids: Sequence[Any],
    *,
    k: int = 16,
    repeats: int = 20,
    seed: int = 0,
    strata: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Independent grouped halves; descriptive reliability, not a strict ceiling."""
    means, partitions = _groups(_finite_array(scores, 2), group_ids, strata)
    if repeats < 2:
        raise ValueError("At least two split-half repetitions are required")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        shuffled = [rng.permutation(indices) for indices in partitions]
        left = _aggregate(means, [indices[: len(indices) // 2] for indices in shuffled])
        right = _aggregate(means, [indices[len(indices) // 2 :] for indices in shuffled])
        draws.append(jaccard(topk_nodes(left, node_names, k), topk_nodes(right, node_names, k)))
    return {
        "mean_jaccard": float(np.mean(draws)),
        "split_jaccards": draws,
        "reference_interval_95": _interval(draws, 0.95),
        "repeats": repeats,
        "n_groups": len(means),
        "seed": seed,
        "interpretation": "within-checkpoint reliability reference, not a strict ceiling or CI",
    }


def paired_metric_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    group_ids: Sequence[Any],
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
    strata: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Equal-source-group metric estimates and paired CI for B minus A.

    Supply native group scores (e.g. all-correct ETHICS group indicators), not
    item accuracy, when the benchmark defines an exact group scoring rule.
    """
    a, b = _finite_array(values_a, 1), _finite_array(values_b, 1)
    if a.shape != b.shape:
        raise ValueError("Paired metric vectors must have the same shape")
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    means, partitions = _groups(np.stack([a, b], axis=1), group_ids, strata)
    point = _aggregate(means, partitions)
    rng = np.random.default_rng(seed)
    draws = np.stack([_aggregate(means, _resample(partitions, rng)) for _ in range(n_bootstrap)])
    return {
        "mean_a": float(point[0]),
        "mean_b": float(point[1]),
        "difference_b_minus_a": float(point[1] - point[0]),
        "ci_a": _interval(draws[:, 0], confidence),
        "ci_b": _interval(draws[:, 1], confidence),
        "ci_difference": _interval(draws[:, 1] - draws[:, 0], confidence),
        "n_groups": len(means),
        "n_examples": len(a),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "seed": seed,
        "weighting": "equal strata, equal source groups within stratum",
    }
