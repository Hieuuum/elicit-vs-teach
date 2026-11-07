"""train_sft.py ``stratified_val_picks``: check prompts for the exact_match stop.

Silent-failure risk: if the picks reached outside the held-out split, the
stop would score the model on rows it trains on (stopping too early); if the
stratum mix drifted from the training file's, "90% accuracy" would mean a
different problem-size mix than training and test (owner, 2026-11-06). Either
would change when the ts1b children stop with nothing crashing.
"""

from __future__ import annotations

from collections import Counter

import pytest

from tests._scriptloader import load

ts = load("train_sft")


def _strata(n_small: int, n_big: int) -> list:
    """Row i's stratum: n_small rows of (1,4,'+') then n_big rows of (4,4,'-')."""
    return [(1, 4, "+")] * n_small + [(4, 4, "-")] * n_big


def test_picks_are_positions_into_val_idx():
    strata = _strata(100, 900)
    val_idx = list(range(0, 1000, 5))  # 200 held-out rows
    picks = ts.stratified_val_picks(strata, val_idx, 50, seed=0)
    assert all(0 <= p < len(val_idx) for p in picks)
    assert len(set(picks)) == len(picks) == 50


def test_picks_never_leave_the_held_out_rows():
    strata = _strata(100, 900)
    val_idx = list(range(0, 1000, 5))
    held = set(val_idx)
    picks = ts.stratified_val_picks(strata, val_idx, 50, seed=1)
    assert {val_idx[p] for p in picks} <= held


def test_mix_follows_whole_file_not_val_pool():
    # whole file 10% small; val pool deliberately 50% small
    strata = _strata(100, 900)
    val_idx = list(range(0, 100, 2)) + list(range(100, 200, 2))  # 50 small + 50 big
    picks = ts.stratified_val_picks(strata, val_idx, 40, seed=2)
    got = Counter(strata[val_idx[p]] for p in picks)
    assert got == Counter({(1, 4, "+"): 4, (4, 4, "-"): 36})


def test_deterministic_and_seed_dependent():
    strata = _strata(300, 700)
    val_idx = list(range(1000))
    a = ts.stratified_val_picks(strata, val_idx, 100, seed=316)
    b = ts.stratified_val_picks(strata, val_idx, 100, seed=316)
    c = ts.stratified_val_picks(strata, val_idx, 100, seed=317)
    assert a == b and a != c


def test_short_stratum_raises():
    strata = _strata(100, 900)
    val_idx = [0] + list(range(100, 300))  # only 1 small row held out
    with pytest.raises(ValueError, match="raise data.val_fraction"):
        ts.stratified_val_picks(strata, val_idx, 100, seed=0)


def test_picks_are_shuffled_across_strata():
    strata = _strata(500, 500)
    val_idx = list(range(1000))
    picks = ts.stratified_val_picks(strata, val_idx, 100, seed=4)
    first = Counter(strata[val_idx[p]] for p in picks[:50])
    assert 10 < first[(1, 4, "+")] < 40


def test_exact_n_with_rounding():
    strata = [(1, 4, "+"), (2, 3, "-"), (4, 4, "+")]
    strata = strata * 100
    val_idx = list(range(300))
    assert len(ts.stratified_val_picks(strata, val_idx, 100, seed=0)) == 100


def test_behavioral_metrics_registered():
    assert ts.BEHAVIORAL_METRICS == ("format_validity", "exact_match")
