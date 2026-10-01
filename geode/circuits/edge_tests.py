"""Model-free statistics for the EAP-IG edge-circuit validation plan.

Every function here takes already-realized numbers (means, per-example
scores, random-draw arrays) and returns a verdict or a derived statistic. No
function trains, runs, or scores a model, and none draws its own randomness:
callers own the random circuits/baselines and pass the resulting arrays in.
See `experiments/eapig-circuit-check/PLAN.md` ("Frozen decisions") for the
thresholds and conventions these functions implement.

Edge sets are plain integer index arrays into the fixed 195,865-edge graph
(`experiments/eapig-circuit-check/PLAN.md`, "Edge graph" row). Node names are
strings; `a{l}.h{i}` and `a{l}.kv{g}` denote heads, everything else
(`embed`, `m{l}`, `logits`) does not.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import binomtest, hypergeom, t as student_t

__all__ = [
    "faithfulness",
    "log_size_mean_f",
    "tost_equivalence",
    "random_baseline_test",
    "topk_edges",
    "size_to_k",
    "consistency",
    "consistency_pass",
    "specificity",
    "jaccard",
    "chance_jaccard",
    "change_rate",
    "topk_overlap",
    "layer_histogram",
    "nodes_touched",
    "heads_only",
    "split_half_ceiling",
    "ceiling_fraction",
    "stopping_rule",
]


# --------------------------------------------------------------------------
# Faithfulness
# --------------------------------------------------------------------------


def faithfulness(
    m_circuit: float, m_empty: float, m_full: float, *, tiny: float = 1e-8
) -> tuple[float, bool]:
    """f = (m_circuit - m_empty) / (m_full - m_empty); means already taken.

    Returns `(f, degenerate)`. When `|m_full - m_empty| < tiny`, the ratio is
    numerically meaningless (the model barely separates full from empty
    ablation), so this returns `(nan, True)` instead of dividing silently.
    """
    denom = m_full - m_empty
    if abs(denom) < tiny:
        return float("nan"), True
    return (m_circuit - m_empty) / denom, False


def log_size_mean_f(sizes: Sequence[float], f_values: Sequence[float]) -> float:
    """Mean of f over circuit sizes, weighted by trapezoid area on a log10 axis.

    Equal `f_values` return that common value regardless of size spacing. A
    single size has no range to integrate over, so it is returned as-is.
    """
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    f_arr = np.asarray(f_values, dtype=np.float64)
    if sizes_arr.shape != f_arr.shape or sizes_arr.ndim != 1 or sizes_arr.size == 0:
        raise ValueError("sizes and f_values must be equal-length, nonempty 1-D sequences")
    if np.any(sizes_arr <= 0):
        raise ValueError("sizes must be strictly positive (they are log10'd)")
    order = np.argsort(sizes_arr)
    log_sizes = np.log10(sizes_arr[order])
    f_sorted = f_arr[order]
    if log_sizes[0] == log_sizes[-1]:
        if not np.all(f_sorted == f_sorted[0]):
            raise ValueError("duplicate sizes require equal f_values (the area is undefined otherwise)")
        return float(f_sorted[0])
    area = np.trapezoid(f_sorted, log_sizes)
    return float(area / (log_sizes[-1] - log_sizes[0]))


# --------------------------------------------------------------------------
# Equivalence (TOST) and the two random-baseline tests
# --------------------------------------------------------------------------


def tost_equivalence(
    ld_circuit: Sequence[float], ld_full: Sequence[float], eps: float, alpha: float = 0.05
) -> dict[str, Any]:
    """Paired two one-sided tests (TOST) on d_i = ld_circuit_i - ld_full_i.

    Equivalence bounds are +/- eps. Returns the two one-sided p-values, their
    max (`p`), `mean_diff`, and `pass` (True iff `p < alpha`).
    """
    circuit = np.asarray(ld_circuit, dtype=np.float64)
    full = np.asarray(ld_full, dtype=np.float64)
    if circuit.shape != full.shape or circuit.ndim != 1 or circuit.size < 2:
        raise ValueError("ld_circuit and ld_full must be equal-length 1-D sequences, n >= 2")
    if eps <= 0:
        raise ValueError("eps must be positive")
    diff = circuit - full
    n = diff.size
    df = n - 1
    mean_diff = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        p_lower = 0.0 if mean_diff > -eps else 1.0
        p_upper = 0.0 if mean_diff < eps else 1.0
    else:
        t_lower = (mean_diff - (-eps)) / se
        t_upper = (mean_diff - eps) / se
        p_lower = float(student_t.sf(t_lower, df))
        p_upper = float(student_t.cdf(t_upper, df))
    p = max(p_lower, p_upper)
    return {
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p": p,
        "mean_diff": mean_diff,
        "pass": bool(p < alpha),
        "n": n,
        "eps": eps,
        "alpha": alpha,
    }


def random_baseline_test(
    stat_circuit: float,
    stat_random: Sequence[float],
    higher_is_better: bool = True,
    win_rate: float = 0.90,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Does the circuit beat >= win_rate of random same-size draws?

    "Beats" means strictly better in the declared direction; ties count as
    losses. `pass` requires both the observed win fraction >= `win_rate` and
    a one-sided binomial test of wins against p=0.5 at `alpha`. Also returns
    an empirical one-sided p-value and the random band (5th/50th/95th pct).

    `higher_is_better=True` is for sufficiency (f should be high);
    `higher_is_better=False` is for partial necessity (ablating the circuit
    should give a LOWER f than ablating random same-size sets).
    """
    random_arr = np.asarray(stat_random, dtype=np.float64)
    if random_arr.ndim != 1 or random_arr.size == 0:
        raise ValueError("stat_random must be a nonempty 1-D sequence")
    n = random_arr.size
    if higher_is_better:
        wins = int(np.sum(stat_circuit > random_arr))
        at_least_as_good = int(np.sum(random_arr >= stat_circuit))
    else:
        wins = int(np.sum(stat_circuit < random_arr))
        at_least_as_good = int(np.sum(random_arr <= stat_circuit))
    win_fraction = wins / n
    binom_p = float(binomtest(wins, n, 0.5, alternative="greater").pvalue)
    empirical_p = (1 + at_least_as_good) / (n + 1)
    p5, p50, p95 = np.percentile(random_arr, [5, 50, 95])
    return {
        "wins": wins,
        "n": n,
        "win_fraction": win_fraction,
        "binom_pvalue": binom_p,
        "empirical_pvalue": float(empirical_p),
        "random_p5": float(p5),
        "random_p50": float(p50),
        "random_p95": float(p95),
        "pass": bool(win_fraction >= win_rate and binom_p < alpha),
    }


# --------------------------------------------------------------------------
# Top-k edge selection
# --------------------------------------------------------------------------


def topk_edges(scores: Sequence[float], k: int) -> np.ndarray:
    """Indices of the top-k entries by signed score, descending.

    Ties are broken deterministically by ascending index (the lower-index
    edge among equal scores is kept first).
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    if scores_arr.ndim != 1 or scores_arr.size == 0:
        raise ValueError("scores must be a nonempty 1-D sequence")
    n = scores_arr.size
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= n:
        raise ValueError("k must be an integer with 1 <= k <= len(scores)")
    order = sorted(range(n), key=lambda i: (-scores_arr[i], i))
    return np.asarray(order[:k], dtype=np.int64)


def size_to_k(frac: float, n_edges: int) -> int:
    """Round frac * n_edges to an edge count, at least 1."""
    if not 0 < frac <= 1:
        raise ValueError("frac must lie in (0, 1]")
    if not isinstance(n_edges, (int, np.integer)) or n_edges <= 0:
        raise ValueError("n_edges must be a positive integer")
    return max(1, int(round(frac * n_edges)))


# --------------------------------------------------------------------------
# Consistency
# --------------------------------------------------------------------------


def consistency(
    per_example_scores: np.ndarray, k: int, share: float = 0.5
) -> dict[str, Any]:
    """Per-example top-k circuits, the edges shared across >= `share` of them.

    `per_example_scores` is (N, E): one row of edge scores per example.
    `coverage_i = |shared ∩ C_i| / |C_i|` for each example's own top-k set
    `C_i`. An empty shared set yields coverage 0 for every example (a fail
    under `consistency_pass`), not an error.

    Memory stays O(N*k + E): per-row top-k via `argpartition`, edge counts
    via `bincount`, so this is safe at N=512, E=195,865.
    """
    scores = np.asarray(per_example_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("per_example_scores must be a nonempty 2-D (N, E) array")
    n, e = scores.shape
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= e:
        raise ValueError("k must be an integer with 1 <= k <= number of edges")
    if not 0 < share <= 1:
        raise ValueError("share must lie in (0, 1]")
    # Top-k indices per row (unordered within the row; ties at the boundary
    # may be resolved arbitrarily by argpartition, which is fine here since
    # only set membership and aggregate coverage are used downstream).
    top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    counts = np.bincount(top_idx.ravel(), minlength=e)
    frac = counts / n
    shared_mask = frac >= share - 1e-12
    shared_idx = np.flatnonzero(shared_mask)
    membership = shared_mask[top_idx]
    coverage = membership.mean(axis=1)
    return {
        "shared": shared_idx,
        "mean_coverage": float(coverage.mean()),
        "coverage": coverage,
    }


def consistency_pass(
    mean_coverage: float, ablation_test_p: float, cov_min: float = 0.70, alpha: float = 0.05
) -> bool:
    """Consistency passes iff mean coverage is high AND ablating the shared
    set beats random same-size sets.

    `ablation_test_p` is the one-sided p-value from the caller's
    `random_baseline_test` on ablating the shared edges vs. random same-size
    sets (e.g. its `empirical_pvalue` or `binom_pvalue`).
    """
    return bool(mean_coverage >= cov_min and ablation_test_p < alpha)


# --------------------------------------------------------------------------
# Specificity
# --------------------------------------------------------------------------


def specificity(
    add_full: float,
    add_ablated: float,
    copy_full: float,
    copy_ablated: float,
    add_drop_random: Sequence[float],
    ratio: float = 3.0,
) -> dict[str, Any]:
    """Does ablating the circuit hurt addition much more than copying?

    `rel_add`/`rel_copy` are relative LD drops. If `copy_full <= 0`, the copy
    task isn't measurable for this model, so status is "not_measurable" (an
    owner decision: neither pass nor fail). A `rel_copy <= 0` (copy LD
    improved after ablation) is treated as satisfying the ratio bound, since
    `ratio * max(rel_copy, 0)` is then 0 and `rel_add` need only be
    non-negative.

    Pass requires `rel_add >= ratio * max(rel_copy, 0)` AND the absolute
    addition drop exceeds the 95th percentile of `add_drop_random`.
    """
    if add_full <= 0:
        raise ValueError(
            "add_full must be positive; gate on m(full) > 0 before calling specificity"
        )
    random_arr = np.asarray(add_drop_random, dtype=np.float64)
    if random_arr.ndim != 1 or random_arr.size == 0:
        raise ValueError("add_drop_random must be a nonempty 1-D sequence")
    rel_add = (add_full - add_ablated) / add_full
    add_drop = add_full - add_ablated
    p95 = float(np.percentile(random_arr, 95))
    if copy_full <= 0:
        return {
            "status": "not_measurable",
            "pass": None,
            "rel_add": rel_add,
            "rel_copy": None,
            "add_drop": add_drop,
            "add_drop_p95_random": p95,
        }
    rel_copy = (copy_full - copy_ablated) / copy_full
    ratio_ok = rel_add >= ratio * max(rel_copy, 0.0)
    drop_ok = add_drop > p95
    passed = bool(ratio_ok and drop_ok)
    return {
        "status": "pass" if passed else "fail",
        "pass": passed,
        "rel_add": rel_add,
        "rel_copy": rel_copy,
        "add_drop": add_drop,
        "add_drop_p95_random": p95,
        "ratio_ok": bool(ratio_ok),
        "drop_ok": bool(drop_ok),
    }


# --------------------------------------------------------------------------
# Overlap measures on edge index sets
# --------------------------------------------------------------------------


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    """Set overlap of two edge-index collections; two empty sets give 1."""
    left, right = set(a), set(b)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def chance_jaccard(k1: int, k2: int, n: int) -> float:
    """Exact E[Jaccard] for two independent uniform random subsets of {0..n-1}.

    |intersection| ~ Hypergeometric(n, k1, k2): P(X=x) = C(k1,x)C(n-k1,k2-x)/C(n,k2).
    Jaccard = X / (k1 + k2 - X), averaged over that distribution (not the
    incorrect ratio-of-expectations shortcut).
    """
    if not all(isinstance(v, (int, np.integer)) for v in (k1, k2, n)):
        raise ValueError("k1, k2, n must be integers")
    if not (0 <= k1 <= n and 0 <= k2 <= n):
        raise ValueError("require 0 <= k1, k2 <= n")
    if k1 == 0 and k2 == 0:
        return 1.0
    lo = max(0, k1 + k2 - n)
    hi = min(k1, k2)
    xs = np.arange(lo, hi + 1)
    pmf = hypergeom.pmf(xs, n, k1, k2)
    union = k1 + k2 - xs  # > 0 for every x in range since k1 + k2 > 0 here
    return float(np.sum(pmf * xs / union))


def change_rate(
    parent: Iterable[int], child: Iterable[int]
) -> tuple[int, int, float]:
    """(added, removed, (added + removed) / |parent|) edges from parent to child."""
    parent_set, child_set = set(parent), set(child)
    if not parent_set:
        raise ValueError("parent must be nonempty to compute a change rate")
    added = len(child_set - parent_set)
    removed = len(parent_set - child_set)
    return added, removed, (added + removed) / len(parent_set)


def topk_overlap(scores_a: Sequence[float], scores_b: Sequence[float], k: int) -> int:
    """|top_k(a) ∩ top_k(b)| by signed score."""
    return len(set(topk_edges(scores_a, k).tolist()) & set(topk_edges(scores_b, k).tolist()))


def layer_histogram(
    edge_idx: Iterable[int], receiving_layer_of_edge: Sequence[int], n_layers: int
) -> np.ndarray:
    """Count of `edge_idx` entries per receiving layer, length `n_layers`.

    `receiving_layer_of_edge` maps every edge in the full graph to a
    receiving-layer index in `[0, n_layers)`; callers wanting added/removed
    histograms call this once per edge set. Receivers outside the per-layer
    axis (e.g. logits) must be excluded from `edge_idx` by the caller.
    """
    layers_full = np.asarray(receiving_layer_of_edge)
    idx = np.fromiter(edge_idx, dtype=np.int64)
    if not isinstance(n_layers, (int, np.integer)) or n_layers <= 0:
        raise ValueError("n_layers must be a positive integer")
    if idx.size == 0:
        return np.zeros(n_layers, dtype=np.int64)
    layers = layers_full[idx]
    if np.any((layers < 0) | (layers >= n_layers)):
        raise ValueError("receiving layer indices for edge_idx must lie in [0, n_layers)")
    return np.bincount(layers, minlength=n_layers).astype(np.int64)


# --------------------------------------------------------------------------
# Node-level
# --------------------------------------------------------------------------

_HEAD_PREFIX = ("a",)  # a{l}.h{i} and a{l}.kv{g}; everything else is not a head


def nodes_touched(
    edge_idx: Iterable[int],
    upstream_node_of_edge: Sequence[str],
    receiver_node_of_edge: Sequence[str],
) -> set[str]:
    """Nodes touched by a set of edges: each edge's upstream node and receiver node."""
    upstream = np.asarray(upstream_node_of_edge, dtype=object)
    receiver = np.asarray(receiver_node_of_edge, dtype=object)
    idx = np.fromiter(edge_idx, dtype=np.int64)
    touched: set[str] = set()
    touched.update(upstream[idx].tolist())
    touched.update(receiver[idx].tolist())
    return touched


def heads_only(nodes: Iterable[str]) -> set[str]:
    """Filter node names to attention heads: `a{l}.h{i}` and `a{l}.kv{g}`.

    `embed`, `m{l}` (MLPs), and `logits` are excluded.
    """
    result = set()
    for name in nodes:
        if not isinstance(name, str):
            raise ValueError("node names must be strings")
        if name.startswith(_HEAD_PREFIX) and "." in name:
            _, _, rest = name.partition(".")
            if rest.startswith("h") or rest.startswith("kv"):
                result.add(name)
    return result


# --------------------------------------------------------------------------
# Split-half ceiling
# --------------------------------------------------------------------------


def split_half_ceiling(scores_a: Sequence[float], scores_b: Sequence[float], k: int) -> float:
    """Jaccard of the two halves' top-k edge sets: the overlap ceiling at size k."""
    return jaccard(topk_edges(scores_a, k).tolist(), topk_edges(scores_b, k).tolist())


def ceiling_fraction(j: float, ceiling: float) -> float:
    """j as a fraction of the split-half ceiling; nan if the ceiling is zero."""
    if ceiling == 0:
        return float("nan")
    return j / ceiling


# --------------------------------------------------------------------------
# Stopping rule
# --------------------------------------------------------------------------


def stopping_rule(results_by_size: Mapping[float, Mapping[str, bool | None]]) -> float | None:
    """First size (ascending) where every test passed; None blocks, True passes.

    A `None` test result ("not measurable", e.g. specificity when the copy
    task is undefined) does not block that size from qualifying. Returns
    None if no size has every test passing or not-measurable.
    """
    for size in sorted(results_by_size):
        tests = results_by_size[size]
        if all(value is not False for value in tests.values()):
            return size
    return None
