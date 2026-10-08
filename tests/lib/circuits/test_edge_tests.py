"""Offline mathematical checks for the EAP-IG edge-circuit validation tests.

Every test exercises pure numpy/scipy math against hand-derived or
brute-force-enumerated ground truth; nothing here touches a model.
"""

import itertools

import numpy as np
import pytest
from scipy.stats import binomtest, t as student_t

from geode.circuits.edge_tests import (
    ceiling_fraction,
    chance_jaccard,
    change_rate,
    consistency,
    consistency_pass,
    faithfulness,
    heads_only,
    jaccard,
    kl_equivalence,
    kl_faithfulness,
    kl_specificity,
    layer_histogram,
    log_size_mean_f,
    nodes_touched,
    random_baseline_test,
    size_to_k,
    specificity,
    split_half_ceiling,
    stopping_rule,
    topk_edges,
    topk_overlap,
    tost_equivalence,
)


# --------------------------------------------------------------------------
# faithfulness
# --------------------------------------------------------------------------


def test_faithfulness_known_values():
    assert faithfulness(5, 0, 10) == (pytest.approx(0.5), False)
    assert faithfulness(10, 0, 10) == (pytest.approx(1.0), False)
    assert faithfulness(0, 0, 10) == (pytest.approx(0.0), False)
    assert faithfulness(-2, -2, 8) == (pytest.approx(0.0), False)


def test_faithfulness_can_exceed_unit_interval():
    # Circuit can beat the full model (f > 1) or underperform empty (f < 0).
    assert faithfulness(12, 0, 10) == (pytest.approx(1.2), False)
    assert faithfulness(-1, 0, 10) == (pytest.approx(-0.1), False)


def test_faithfulness_tiny_denominator_returns_nan_and_flag():
    f, degenerate = faithfulness(1.0, 1e-10, -1e-10)
    assert degenerate is True
    assert np.isnan(f)


def test_faithfulness_tiny_denominator_respects_custom_threshold():
    # denom = 1e-5 is NOT tiny under a looser threshold.
    f, degenerate = faithfulness(1.0, 0.0, 1e-5, tiny=1e-6)
    assert degenerate is False
    assert f == pytest.approx(1e5)
    # ... but IS tiny under a stricter one.
    f2, degenerate2 = faithfulness(1.0, 0.0, 1e-5, tiny=1e-4)
    assert degenerate2 is True
    assert np.isnan(f2)


def test_faithfulness_exact_zero_denominator_flagged():
    f, degenerate = faithfulness(3.0, 2.0, 2.0)
    assert degenerate is True
    assert np.isnan(f)


# --------------------------------------------------------------------------
# log_size_mean_f
# --------------------------------------------------------------------------


def test_log_size_mean_f_constant_values_return_that_value():
    assert log_size_mean_f([10, 100, 1000], [0.7, 0.7, 0.7]) == pytest.approx(0.7)
    assert log_size_mean_f([5, 5000], [-0.3, -0.3]) == pytest.approx(-0.3)


def test_log_size_mean_f_linear_in_log_size_is_exact_trapezoid():
    # f linear in log10(size): endpoints alone give the exact mean (1/2)(f0+f1).
    sizes = [1, 10]
    f_values = [0.0, 1.0]
    assert log_size_mean_f(sizes, f_values) == pytest.approx(0.5)
    # Adding a midpoint consistent with the same line changes nothing.
    sizes2 = [1, 10, 100]
    f_values2 = [0.0, 1.0, 2.0]
    assert log_size_mean_f(sizes2, f_values2) == pytest.approx(1.0)


def test_log_size_mean_f_is_order_invariant():
    sizes = [1000, 1, 100, 10]
    f_values = [2.0, 0.0, 1.5, 1.0]
    forward = log_size_mean_f(sizes, f_values)
    sizes_sorted = [1, 10, 100, 1000]
    f_sorted = [0.0, 1.0, 1.5, 2.0]
    assert forward == pytest.approx(log_size_mean_f(sizes_sorted, f_sorted))


def test_log_size_mean_f_single_size_returns_that_f():
    assert log_size_mean_f([50], [0.42]) == pytest.approx(0.42)


def test_log_size_mean_f_duplicate_sizes_require_equal_f():
    assert log_size_mean_f([10, 10], [0.3, 0.3]) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        log_size_mean_f([10, 10], [0.3, 0.9])


def test_log_size_mean_f_rejects_bad_input():
    with pytest.raises(ValueError):
        log_size_mean_f([], [])
    with pytest.raises(ValueError):
        log_size_mean_f([1, 2], [0.1])
    with pytest.raises(ValueError):
        log_size_mean_f([0, 10], [0.1, 0.2])
    with pytest.raises(ValueError):
        log_size_mean_f([-5, 10], [0.1, 0.2])


# --------------------------------------------------------------------------
# tost_equivalence
# --------------------------------------------------------------------------


def test_tost_equivalence_zero_variance_inside_bounds_passes_trivially():
    result = tost_equivalence([1.0] * 5, [1.0] * 5, eps=0.1)
    assert result["mean_diff"] == pytest.approx(0.0)
    assert result["p"] == pytest.approx(0.0)
    assert result["pass"] is True


def test_tost_equivalence_zero_variance_outside_bounds_fails():
    # Constant difference of 1.0, way outside +/- 0.1.
    result = tost_equivalence([2.0] * 4, [1.0] * 4, eps=0.1)
    assert result["mean_diff"] == pytest.approx(1.0)
    assert result["pass"] is False
    assert result["p"] == pytest.approx(1.0)


def test_tost_equivalence_symmetric_at_zero_mean_difference():
    diff_circuit = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5]
    diff_full = [0.0] * 6
    result = tost_equivalence(diff_circuit, diff_full, eps=5.0)
    assert result["mean_diff"] == pytest.approx(0.0, abs=1e-9)
    assert result["p_lower"] == pytest.approx(result["p_upper"], abs=1e-9)


def test_tost_equivalence_large_bounds_pass_tight_bounds_fail():
    ld_circuit = [1.0, 0.9, 1.1, 0.95, 1.05]
    ld_full = [1.0, 1.0, 1.0, 1.0, 1.0]
    loose = tost_equivalence(ld_circuit, ld_full, eps=5.0)
    tight = tost_equivalence(ld_circuit, ld_full, eps=1e-6)
    assert loose["pass"] is True
    assert tight["pass"] is False
    assert loose["p"] < tight["p"]


def test_tost_equivalence_matches_hand_rolled_t_statistics():
    diff_circuit = [1.1, 0.9, 1.3, 0.7, 1.0]
    diff_full = [1.0] * 5
    eps = 0.5
    result = tost_equivalence(diff_circuit, diff_full, eps=eps)
    d = np.array(diff_circuit) - np.array(diff_full)
    n = d.size
    mean_diff = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    t_lower = (mean_diff + eps) / se
    t_upper = (mean_diff - eps) / se
    expected_p_lower = float(student_t.sf(t_lower, n - 1))
    expected_p_upper = float(student_t.cdf(t_upper, n - 1))
    assert result["p_lower"] == pytest.approx(expected_p_lower)
    assert result["p_upper"] == pytest.approx(expected_p_upper)
    assert result["p"] == pytest.approx(max(expected_p_lower, expected_p_upper))


def test_tost_equivalence_rejects_bad_input():
    with pytest.raises(ValueError):
        tost_equivalence([1.0], [1.0], eps=0.1)  # n < 2
    with pytest.raises(ValueError):
        tost_equivalence([1.0, 2.0], [1.0], eps=0.1)  # mismatched length
    with pytest.raises(ValueError):
        tost_equivalence([1.0, 2.0], [1.0, 2.0], eps=0.0)  # eps must be > 0
    with pytest.raises(ValueError):
        tost_equivalence([1.0, 2.0], [1.0, 2.0], eps=-1.0)


# --------------------------------------------------------------------------
# random_baseline_test
# --------------------------------------------------------------------------


def test_random_baseline_test_sufficiency_all_wins_passes():
    # n=5 is the smallest all-wins sample where binom_p = 0.5**5 < 0.05.
    result = random_baseline_test(10.0, [1.0, 2.0, 3.0, 4.0, 5.0], higher_is_better=True)
    assert result["wins"] == 5
    assert result["win_fraction"] == pytest.approx(1.0)
    assert result["pass"] is True
    assert result["empirical_pvalue"] == pytest.approx(1 / 6)


def test_random_baseline_test_too_few_draws_cannot_reach_significance():
    # n=4 all-wins has binom_p = 0.5**4 = 0.0625 > 0.05: win_fraction alone isn't enough.
    result = random_baseline_test(10.0, [1.0, 2.0, 3.0, 4.0], higher_is_better=True)
    assert result["wins"] == 4
    assert result["win_fraction"] == pytest.approx(1.0)
    assert result["binom_pvalue"] == pytest.approx(0.0625)
    assert result["pass"] is False


def test_random_baseline_test_ties_count_as_losses():
    # circuit == 5 ties two of the five random draws; those are losses, not wins.
    result = random_baseline_test(5.0, [5.0, 5.0, 3.0, 3.0, 3.0], higher_is_better=True)
    assert result["wins"] == 3
    assert result["win_fraction"] == pytest.approx(0.6)
    assert result["pass"] is False  # below win_rate=0.90 default
    # "at least as good" counts the two ties plus zero draws strictly better.
    assert result["empirical_pvalue"] == pytest.approx((1 + 2) / 6)


def test_random_baseline_test_lower_is_better_for_partial_necessity():
    # Ablating the circuit should leave LOWER f than ablating random sets.
    result = random_baseline_test(0.1, [0.5, 0.6, 0.7, 0.8, 0.9], higher_is_better=False)
    assert result["wins"] == 5
    assert result["win_fraction"] == pytest.approx(1.0)
    assert result["pass"] is True


def test_random_baseline_test_lower_is_better_ties_count_as_losses():
    # Lower is better: circuit "beats" a draw by being strictly SMALLER.
    result = random_baseline_test(0.2, [0.2, 0.4, 0.3], higher_is_better=False)
    assert result["wins"] == 2  # beats 0.4 and 0.3; ties the 0.2 draw, which is a loss
    assert result["empirical_pvalue"] == pytest.approx((1 + 1) / 4)  # the tie is "at least as good"


def test_random_baseline_test_win_rate_boundary_is_inclusive():
    # 9/10 == win_rate exactly; should not fail purely on the >= boundary.
    random_vals = [0.0] * 9 + [2.0]
    result = random_baseline_test(1.0, random_vals, win_rate=0.9)
    assert result["win_fraction"] == pytest.approx(0.9)
    assert result["wins"] == 9


def test_random_baseline_test_binom_pvalue_matches_scipy():
    stat_random = [1.0] * 10
    result = random_baseline_test(2.0, stat_random)  # beats all 10
    expected = float(binomtest(10, 10, 0.5, alternative="greater").pvalue)
    assert result["binom_pvalue"] == pytest.approx(expected)


def test_random_baseline_test_percentiles_known_array():
    stat_random = list(range(1, 101))  # 1..100
    result = random_baseline_test(1000.0, stat_random)
    assert result["random_p50"] == pytest.approx(np.percentile(stat_random, 50))
    assert result["random_p5"] == pytest.approx(np.percentile(stat_random, 5))
    assert result["random_p95"] == pytest.approx(np.percentile(stat_random, 95))


def test_random_baseline_test_rejects_empty_random():
    with pytest.raises(ValueError):
        random_baseline_test(1.0, [])


# --------------------------------------------------------------------------
# topk_edges / size_to_k
# --------------------------------------------------------------------------


def test_topk_edges_basic_descending_order():
    scores = [5.0, 1.0, 9.0, 3.0]
    assert topk_edges(scores, 2).tolist() == [2, 0]


def test_topk_edges_tie_break_is_ascending_index():
    scores = [5.0, 5.0, 5.0, 1.0]
    assert topk_edges(scores, 2).tolist() == [0, 1]


def test_topk_edges_handles_negative_signed_scores():
    scores = [-5.0, -1.0, -9.0, 3.0]
    # Top-1 is the single positive value, not the smallest magnitude.
    assert topk_edges(scores, 1).tolist() == [3]
    assert topk_edges(scores, 4).tolist() == [3, 1, 0, 2]


def test_topk_edges_k_equals_full_length():
    scores = [2.0, 1.0, 3.0]
    assert sorted(topk_edges(scores, 3).tolist()) == [0, 1, 2]


def test_topk_edges_single_element():
    assert topk_edges([7.0], 1).tolist() == [0]


def test_topk_edges_rejects_bad_k():
    with pytest.raises(ValueError):
        topk_edges([1.0, 2.0], 0)
    with pytest.raises(ValueError):
        topk_edges([1.0, 2.0], 3)
    with pytest.raises(ValueError):
        topk_edges([1.0, 2.0], -1)
    with pytest.raises(ValueError):
        topk_edges([], 1)


def test_size_to_k_rounds_and_floors_at_one():
    n_edges = 195_865
    assert size_to_k(0.001, n_edges) == round(0.001 * n_edges)
    assert size_to_k(0.002, n_edges) == round(0.002 * n_edges)
    assert size_to_k(0.005, n_edges) == round(0.005 * n_edges)
    assert size_to_k(0.01, n_edges) == round(0.01 * n_edges)


def test_size_to_k_never_returns_zero():
    assert size_to_k(0.0001, 10) == 1  # rounds to 0, clamped up to 1


def test_size_to_k_half_rounding_matches_python_banker_rounding():
    # 0.25 * 10 = 2.5 -> Python's round-half-to-even gives 2, not 3.
    assert size_to_k(0.25, 10) == round(2.5) == 2


def test_size_to_k_rejects_bad_input():
    with pytest.raises(ValueError):
        size_to_k(0.0, 100)
    with pytest.raises(ValueError):
        size_to_k(1.5, 100)
    with pytest.raises(ValueError):
        size_to_k(0.1, 0)
    with pytest.raises(ValueError):
        size_to_k(0.1, -5)


# --------------------------------------------------------------------------
# consistency / consistency_pass
# --------------------------------------------------------------------------


def test_consistency_known_shared_set_and_coverage():
    # 4 rows, 5 edges, k=2. Rows 0-2 favor edges {0,1}; row 3 favors {2,3}.
    scores = np.array(
        [
            [9, 8, 1, 1, 0],
            [8, 9, 1, 1, 0],
            [7, 6, 2, 2, 0],
            [1, 1, 9, 8, 0],
        ],
        dtype=float,
    )
    result = consistency(scores, k=2, share=0.5)
    assert sorted(result["shared"].tolist()) == [0, 1]
    assert result["coverage"].shape == (4,)
    assert result["coverage"][:3] == pytest.approx([1.0, 1.0, 1.0])
    assert result["coverage"][3] == pytest.approx(0.0)
    assert result["mean_coverage"] == pytest.approx(0.75)


def test_consistency_empty_shared_set_gives_zero_coverage():
    # 4 rows, 4 edges, k=1, every row picks a different edge -> nothing shared.
    scores = np.eye(4) * 10
    result = consistency(scores, k=1, share=0.5)
    assert result["shared"].size == 0
    assert result["mean_coverage"] == pytest.approx(0.0)
    assert np.all(result["coverage"] == 0.0)


def test_consistency_share_one_requires_unanimous_edges():
    scores = np.array(
        [
            [5, 4, 0],
            [5, 4, 0],
            [5, 1, 3],
        ],
        dtype=float,
    )
    result = consistency(scores, k=2, share=1.0)
    # Edge 0 is in every row's top-2; edge 1 misses row 2's top-2 (2/3 < 1.0).
    assert result["shared"].tolist() == [0]


def test_consistency_at_stated_scale_does_not_crash_or_blow_memory():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(512, 195_865))
    result = consistency(scores, k=size_to_k(0.001, 195_865), share=0.5)
    assert result["coverage"].shape == (512,)
    assert 0.0 <= result["mean_coverage"] <= 1.0


def test_consistency_rejects_bad_input():
    with pytest.raises(ValueError):
        consistency(np.zeros((3, 4)), k=0)
    with pytest.raises(ValueError):
        consistency(np.zeros((3, 4)), k=5)
    with pytest.raises(ValueError):
        consistency(np.zeros((3, 4)), k=1, share=0.0)
    with pytest.raises(ValueError):
        consistency(np.zeros((3, 4)), k=1, share=1.5)
    with pytest.raises(ValueError):
        consistency(np.zeros((0, 4)), k=1)
    with pytest.raises(ValueError):
        consistency(np.zeros(4), k=1)


def test_consistency_pass_requires_both_conditions():
    assert consistency_pass(0.8, 0.01) is True
    assert consistency_pass(0.5, 0.01) is False  # coverage too low
    assert consistency_pass(0.8, 0.5) is False  # p not significant


def test_consistency_pass_boundaries():
    assert consistency_pass(0.70, 0.049, cov_min=0.70, alpha=0.05) is True
    assert consistency_pass(0.70, 0.05, cov_min=0.70, alpha=0.05) is False  # p == alpha fails


# --------------------------------------------------------------------------
# specificity
# --------------------------------------------------------------------------


def test_specificity_not_measurable_when_copy_full_nonpositive():
    result = specificity(
        add_full=10.0, add_ablated=2.0, copy_full=0.0, copy_ablated=0.0, add_drop_random=[1.0, 2.0]
    )
    assert result["status"] == "not_measurable"
    assert result["pass"] is None
    assert result["rel_copy"] is None


def test_specificity_passes_when_ratio_and_drop_both_satisfied():
    result = specificity(
        add_full=10.0,
        add_ablated=1.0,  # rel_add = 0.9
        copy_full=10.0,
        copy_ablated=9.7,  # rel_copy = 0.03
        add_drop_random=[1.0, 2.0, 3.0, 4.0, 5.0],  # p95 well below 9.0
        ratio=3.0,
    )
    assert result["rel_add"] == pytest.approx(0.9)
    assert result["rel_copy"] == pytest.approx(0.03)
    assert result["ratio_ok"] is True
    assert result["drop_ok"] is True
    assert result["status"] == "pass"
    assert result["pass"] is True


def test_specificity_fails_ratio_not_met():
    result = specificity(
        add_full=10.0,
        add_ablated=8.0,  # rel_add = 0.2
        copy_full=10.0,
        copy_ablated=8.0,  # rel_copy = 0.2, ratio*rel_copy = 0.6 > rel_add
        add_drop_random=[0.1, 0.1, 0.1],
    )
    assert result["ratio_ok"] is False
    assert result["status"] == "fail"
    assert result["pass"] is False


def test_specificity_fails_drop_not_above_random_band_even_if_ratio_ok():
    result = specificity(
        add_full=10.0,
        add_ablated=9.0,  # rel_add = 0.1, add_drop = 1.0
        copy_full=10.0,
        copy_ablated=10.0,  # rel_copy = 0.0
        add_drop_random=[5.0, 6.0, 7.0, 8.0, 9.0],  # p95 way above 1.0
    )
    assert result["ratio_ok"] is True  # rel_add=0.1 >= 3*0 = 0
    assert result["drop_ok"] is False
    assert result["status"] == "fail"


def test_specificity_negative_rel_copy_satisfies_ratio_via_max_zero():
    # Copy LD improves after ablation (rel_copy < 0); ratio bound becomes 0.
    result = specificity(
        add_full=10.0,
        add_ablated=9.5,  # rel_add = 0.05 >= 3*0
        copy_full=10.0,
        copy_ablated=10.5,  # rel_copy = -0.05
        add_drop_random=[0.01, 0.02, 0.03],  # p95 well below 0.5
    )
    assert result["rel_copy"] == pytest.approx(-0.05)
    assert result["ratio_ok"] is True
    assert result["status"] == "pass"


def test_specificity_rejects_nonpositive_add_full():
    with pytest.raises(ValueError):
        specificity(add_full=0.0, add_ablated=0.0, copy_full=1.0, copy_ablated=1.0, add_drop_random=[0.1])
    with pytest.raises(ValueError):
        specificity(add_full=-1.0, add_ablated=0.0, copy_full=1.0, copy_ablated=1.0, add_drop_random=[0.1])


def test_specificity_rejects_empty_random_band():
    with pytest.raises(ValueError):
        specificity(add_full=10.0, add_ablated=5.0, copy_full=10.0, copy_ablated=9.0, add_drop_random=[])


# --------------------------------------------------------------------------
# jaccard / chance_jaccard
# --------------------------------------------------------------------------


def test_jaccard_edge_sets_known_overlap():
    assert jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(2 / 4)
    assert jaccard([1, 2], [1, 2]) == pytest.approx(1.0)
    assert jaccard([1, 2], [3, 4]) == pytest.approx(0.0)


def test_jaccard_edge_sets_empty_conventions():
    assert jaccard([], []) == 1.0
    assert jaccard([1], []) == 0.0
    assert jaccard([], [1]) == 0.0


def test_jaccard_edge_sets_ignores_duplicates():
    assert jaccard([1, 1, 2], [2, 2, 2]) == pytest.approx(1 / 2)


def test_chance_jaccard_both_empty_is_one():
    assert chance_jaccard(0, 0, 100) == 1.0


def test_chance_jaccard_one_empty_is_zero():
    assert chance_jaccard(0, 5, 100) == 0.0
    assert chance_jaccard(5, 0, 100) == 0.0


def test_chance_jaccard_full_universe_is_one():
    assert chance_jaccard(10, 10, 10) == pytest.approx(1.0)


def test_chance_jaccard_is_symmetric():
    assert chance_jaccard(3, 5, 20) == pytest.approx(chance_jaccard(5, 3, 20))


def test_chance_jaccard_matches_brute_force_enumeration():
    n, k1, k2 = 6, 2, 3
    sets1 = list(itertools.combinations(range(n), k1))
    sets2 = list(itertools.combinations(range(n), k2))
    exact_mean = np.mean([jaccard(a, b) for a in sets1 for b in sets2])
    assert chance_jaccard(k1, k2, n) == pytest.approx(exact_mean, abs=1e-9)


def test_chance_jaccard_matches_monte_carlo_simulation():
    rng = np.random.default_rng(42)
    n, k1, k2 = 50, 10, 16
    n_draws = 200_000
    universe = np.arange(n)
    jaccards = np.empty(n_draws)
    for i in range(n_draws):
        a = rng.choice(universe, size=k1, replace=False)
        b = rng.choice(universe, size=k2, replace=False)
        jaccards[i] = jaccard(a.tolist(), b.tolist())
    simulated_mean = jaccards.mean()
    assert chance_jaccard(k1, k2, n) == pytest.approx(simulated_mean, abs=0.01)


def test_chance_jaccard_rejects_bad_input():
    with pytest.raises(ValueError):
        chance_jaccard(-1, 5, 10)
    with pytest.raises(ValueError):
        chance_jaccard(5, 11, 10)
    with pytest.raises(ValueError):
        chance_jaccard(1.5, 2, 10)


# --------------------------------------------------------------------------
# change_rate / topk_overlap / layer_histogram
# --------------------------------------------------------------------------


def test_change_rate_known_values():
    parent = [1, 2, 3, 4, 5]
    child = [1, 2, 6, 7, 5]
    added, removed, rate = change_rate(parent, child)
    assert (added, removed) == (2, 2)
    assert rate == pytest.approx(4 / 5)


def test_change_rate_identical_sets_is_zero():
    assert change_rate([1, 2, 3], [1, 2, 3]) == (0, 0, 0.0)


def test_change_rate_fully_disjoint():
    parent, child = [1, 2], [3, 4, 5]
    added, removed, rate = change_rate(parent, child)
    assert added == 3
    assert removed == 2
    assert rate == pytest.approx(5 / 2)


def test_change_rate_child_empty_means_everything_removed():
    added, removed, rate = change_rate([1, 2, 3], [])
    assert (added, removed) == (0, 3)
    assert rate == pytest.approx(1.0)


def test_change_rate_rejects_empty_parent():
    with pytest.raises(ValueError):
        change_rate([], [1, 2])
    with pytest.raises(ValueError):
        change_rate([], [])


def test_topk_overlap_known_cases():
    a = [9.0, 1.0, 8.0, 2.0]
    b = [1.0, 9.0, 2.0, 8.0]
    # top-2 of a: {0, 2}; top-2 of b: {1, 3}. Disjoint.
    assert topk_overlap(a, b, 2) == 0
    assert topk_overlap(a, a, 2) == 2
    assert topk_overlap(a, a, 4) == 4


def test_layer_histogram_known_counts():
    receiving_layer = [0, 0, 1, 1, 2, 2, 2]
    hist = layer_histogram([0, 2, 4, 5, 6], receiving_layer, n_layers=3)
    assert hist.tolist() == [1, 1, 3]


def test_layer_histogram_empty_edges_is_all_zero():
    hist = layer_histogram([], [0, 1, 2], n_layers=3)
    assert hist.tolist() == [0, 0, 0]


def test_layer_histogram_rejects_out_of_range_layers():
    with pytest.raises(ValueError):
        layer_histogram([0], [5], n_layers=3)


def test_layer_histogram_rejects_bad_n_layers():
    with pytest.raises(ValueError):
        layer_histogram([0], [0], n_layers=0)


# --------------------------------------------------------------------------
# nodes_touched / heads_only
# --------------------------------------------------------------------------


def test_nodes_touched_unions_upstream_and_receiver():
    upstream = ["embed", "a0.h1", "m0"]
    receiver = ["a0.h0", "m0", "logits"]
    touched = nodes_touched([0, 1], upstream, receiver)
    assert touched == {"embed", "a0.h0", "a0.h1", "m0"}


def test_nodes_touched_empty_edge_set():
    assert nodes_touched([], ["embed"], ["logits"]) == set()


def test_nodes_touched_all_edges():
    upstream = ["embed", "a0.h1"]
    receiver = ["a0.h0", "logits"]
    assert nodes_touched([0, 1], upstream, receiver) == {"embed", "a0.h1", "a0.h0", "logits"}


def test_heads_only_filters_correctly():
    nodes = {"embed", "a0.h1", "a2.kv3", "m5", "logits", "a1.h31"}
    assert heads_only(nodes) == {"a0.h1", "a2.kv3", "a1.h31"}


def test_heads_only_empty_input():
    assert heads_only([]) == set()


def test_heads_only_rejects_non_strings():
    with pytest.raises(ValueError):
        heads_only([1, 2, 3])


def test_heads_only_excludes_lookalike_names():
    # Starts with "a" but doesn't match "a{l}.h{i}" / "a{l}.kv{g}".
    assert heads_only(["array", "am5", "a0.mlp", "a0.h0"]) == {"a0.h0"}


# --------------------------------------------------------------------------
# split_half_ceiling / ceiling_fraction
# --------------------------------------------------------------------------


def test_split_half_ceiling_matches_manual_jaccard():
    a = [9.0, 8.0, 1.0, 2.0]
    b = [8.0, 9.0, 2.0, 1.0]
    k = 2
    expected = jaccard(topk_edges(a, k).tolist(), topk_edges(b, k).tolist())
    assert split_half_ceiling(a, b, k) == pytest.approx(expected)


def test_split_half_ceiling_identical_halves_is_one():
    a = [3.0, 1.0, 2.0, 0.0]
    assert split_half_ceiling(a, a, 2) == pytest.approx(1.0)


def test_split_half_ceiling_disjoint_top_k_is_zero():
    a = [9.0, 8.0, 1.0, 2.0]
    b = [1.0, 2.0, 9.0, 8.0]
    assert split_half_ceiling(a, b, 2) == pytest.approx(0.0)


def test_ceiling_fraction_normal_and_zero_ceiling():
    assert ceiling_fraction(0.5, 1.0) == pytest.approx(0.5)
    assert ceiling_fraction(0.3, 0.6) == pytest.approx(0.5)
    assert np.isnan(ceiling_fraction(0.1, 0.0))


def test_ceiling_fraction_zero_overlap_with_positive_ceiling():
    assert ceiling_fraction(0.0, 0.5) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# stopping_rule
# --------------------------------------------------------------------------


def test_stopping_rule_first_fully_passing_size():
    results = {
        0.001: {"equivalence": False, "sufficiency": True},
        0.002: {"equivalence": True, "sufficiency": True, "specificity": None},
        0.005: {"equivalence": True, "sufficiency": True},
    }
    assert stopping_rule(results) == 0.002


def test_stopping_rule_none_does_not_block():
    results = {0.001: {"equivalence": True, "specificity": None, "consistency": None}}
    assert stopping_rule(results) == 0.001


def test_stopping_rule_no_size_qualifies_returns_none():
    results = {0.001: {"a": False}, 0.01: {"a": False, "b": True}}
    assert stopping_rule(results) is None


def test_stopping_rule_is_ascending_not_insertion_order():
    results = {
        0.01: {"a": True},
        0.001: {"a": False},
        0.005: {"a": True},
    }
    assert stopping_rule(results) == 0.005


def test_stopping_rule_empty_input_returns_none():
    assert stopping_rule({}) is None


def test_stopping_rule_all_none_passes_at_first_size():
    results = {0.001: {"a": None, "b": None}}
    assert stopping_rule(results) == 0.001


# --------------------------------------------------------------------------
# KL versions
# --------------------------------------------------------------------------


def test_kl_faithfulness_identity_endpoints():
    assert kl_faithfulness(0.0, 2.5) == (1.0, False)  # circuit = full
    assert kl_faithfulness(2.5, 2.5) == (0.0, False)  # circuit = empty


def test_kl_faithfulness_known_value():
    f, degenerate = kl_faithfulness(0.5, 2.0)
    assert f == pytest.approx(0.75) and degenerate is False


def test_kl_faithfulness_can_exceed_one_or_go_negative():
    assert kl_faithfulness(-0.1, 1.0)[0] > 1.0  # KL below 0 only via float noise, but formula allows it
    assert kl_faithfulness(3.0, 1.0)[0] == pytest.approx(-2.0)


def test_kl_faithfulness_degenerate_denominator_returns_nan_and_flag():
    f, degenerate = kl_faithfulness(0.0, 1e-12)
    assert np.isnan(f) and degenerate is True
    f, degenerate = kl_faithfulness(0.0, 0.0)
    assert np.isnan(f) and degenerate is True


def test_kl_faithfulness_custom_tiny():
    assert kl_faithfulness(0.0, 1e-3, tiny=1e-2)[1] is True
    assert kl_faithfulness(0.0, 1e-3, tiny=1e-4) == (1.0, False)


def test_kl_equivalence_full_vs_full_passes():
    r = kl_equivalence([0.0, 0.0, 0.0, 0.0], kl_empty_mean=1.0)
    assert r["pass"] is True and r["p"] == 0.0 and r["mean_kl"] == 0.0 and r["n"] == 4


def test_kl_equivalence_circuit_equal_to_empty_fails():
    per_example_empty = [0.8, 1.0, 1.2, 0.9, 1.1]
    r = kl_equivalence(per_example_empty, kl_empty_mean=float(np.mean(per_example_empty)))
    assert r["pass"] is False and r["p"] > 0.5


def test_kl_equivalence_matches_hand_rolled_t_test():
    kl = [0.02, 0.05, 0.08, 0.03, 0.06, 0.04]
    r = kl_equivalence(kl, kl_empty_mean=1.0, frac=0.10, alpha=0.05)
    arr = np.array(kl)
    t = (arr.mean() - 0.10) / (arr.std(ddof=1) / np.sqrt(6))
    assert r["p"] == pytest.approx(student_t.cdf(t, 5))
    assert r["bound"] == pytest.approx(0.10) and r["mean_kl"] == pytest.approx(arr.mean())
    assert r["pass"] is True and r["frac"] == 0.10 and r["alpha"] == 0.05


def test_kl_equivalence_boundary_mean_equal_to_bound_is_not_a_pass():
    kl = [0.05, 0.15, 0.05, 0.15]  # mean exactly 0.10
    r = kl_equivalence(kl, kl_empty_mean=1.0)
    assert r["mean_kl"] == pytest.approx(0.10)
    assert r["p"] == pytest.approx(0.5)
    assert r["pass"] is False


def test_kl_equivalence_zero_variance_decided_by_mean():
    # dyadic values so the sample std is exactly 0
    below = kl_equivalence([0.0625] * 3, 1.0)
    assert below["p"] == 0.0 and below["pass"] is True
    at_bound = kl_equivalence([0.125] * 3, 0.25, frac=0.5)  # bound = 0.125
    assert at_bound["p"] == 1.0 and at_bound["pass"] is False
    assert kl_equivalence([0.5, 0.5], 1.0)["p"] == 1.0


def test_kl_equivalence_p_monotone_in_mean():
    base = np.array([-1.0, 1.0, -0.5, 0.5, 0.0])  # fixed spread, zero mean
    ps = [kl_equivalence(base * 0.02 + m, kl_empty_mean=1.0)["p"] for m in (0.0, 0.05, 0.10, 0.15, 0.3)]
    assert ps == sorted(ps) and ps[0] < ps[-1]


def test_kl_equivalence_frac_and_alpha_respected():
    kl = [0.08, 0.09, 0.07, 0.085]
    assert kl_equivalence(kl, 1.0, frac=0.10)["pass"] is True
    assert kl_equivalence(kl, 1.0, frac=0.05)["pass"] is False
    p = kl_equivalence(kl, 1.0)["p"]
    assert kl_equivalence(kl, 1.0, alpha=p / 2)["pass"] is False


def test_kl_equivalence_rejects_bad_input():
    with pytest.raises(ValueError):
        kl_equivalence([0.1], 1.0)
    with pytest.raises(ValueError):
        kl_equivalence([], 1.0)
    with pytest.raises(ValueError):
        kl_equivalence([[0.1, 0.2], [0.1, 0.2]], 1.0)
    with pytest.raises(ValueError):
        kl_equivalence([0.1, 0.2], 0.0)
    with pytest.raises(ValueError):
        kl_equivalence([0.1, 0.2], -1.0)


def test_kl_specificity_not_measurable_when_copy_none():
    r = kl_specificity(0.8, None, [0.1, 0.2])
    assert r["status"] == "not_measurable" and r["pass"] is None
    assert r["rel_copy"] is None and r["rel_add"] == 0.8


def test_kl_specificity_passes_when_ratio_and_p95_ok():
    r = kl_specificity(0.9, 0.2, [0.1, 0.2, 0.3])
    assert r["status"] == "pass" and r["pass"] is True
    assert r["ratio_ok"] is True and r["drop_ok"] is True
    assert r["rel_add"] == 0.9 and r["rel_copy"] == 0.2 and r["add_drop"] == 0.9
    assert r["add_drop_p95_random"] == pytest.approx(np.percentile([0.1, 0.2, 0.3], 95))


def test_kl_specificity_fails_ratio():
    r = kl_specificity(0.5, 0.2, [0.1, 0.2, 0.3])  # 0.5 < 3 * 0.2
    assert r["pass"] is False and r["status"] == "fail"
    assert r["ratio_ok"] is False and r["drop_ok"] is True


def test_kl_specificity_fails_p95_even_if_ratio_ok():
    r = kl_specificity(0.5, 0.01, [0.1, 0.2, 0.9])
    assert r["pass"] is False
    assert r["ratio_ok"] is True and r["drop_ok"] is False


def test_kl_specificity_ratio_boundary_is_inclusive_p95_boundary_is_strict():
    assert kl_specificity(0.75, 0.25, [0.1])["ratio_ok"] is True  # exactly 3x
    r = kl_specificity(0.5, 0.0, [0.5, 0.5, 0.5])
    assert r["drop_ok"] is False and r["pass"] is False


def test_kl_specificity_negative_copy_damage_satisfies_ratio():
    r = kl_specificity(0.4, -0.3, [0.1, 0.2])
    assert r["ratio_ok"] is True and r["pass"] is True and r["rel_copy"] == -0.3


def test_kl_specificity_ratio_parameter():
    assert kl_specificity(0.5, 0.2, [0.1], ratio=2.0)["ratio_ok"] is True
    assert kl_specificity(0.5, 0.2, [0.1], ratio=3.0)["ratio_ok"] is False


def test_kl_specificity_rejects_empty_or_non_1d_random():
    with pytest.raises(ValueError):
        kl_specificity(0.5, 0.1, [])
    with pytest.raises(ValueError):
        kl_specificity(0.5, None, [])
    with pytest.raises(ValueError):
        kl_specificity(0.5, 0.1, [[0.1], [0.2]])
