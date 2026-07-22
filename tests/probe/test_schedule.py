"""PROBE-1 schedule tests — specs/02 §7 ``snapshot_steps`` (V5.55-V5.61).

Derived from the spec words only (specs/02 §6 "Runs 5-6" + §7): exactly
min(n, total_steps) strictly increasing steps in [1, total_steps]; dense
unit-stride prefix 1..dense_until; first and final step included; log-then-
uniform tail ("every step through ~30, then stretching, uniform later").
Spacing assertions target the words, not the exact formula: gaps may jitter
by 1 from rounding, but never more, and the last tenth must be ~uniform —
a pure log-to-the-end tail would stretch it by e**(ln(total/dense)/10) >= 2
on the production-shaped cases below, so the 1.6x bound discriminates.
"""

from __future__ import annotations

import random

import pytest

from geode.probe import snapshot_steps

# Production-shaped and degenerate (total_steps, n, dense_until) corners.
# 116_595 is the run-2 canonical 15-epoch ceiling shape.
GRID = [
    (116_595, 1024, 30),
    (100_000, 1024, 30),
    (30_000, 1024, 30),
    (50_000, 512, 30),
    (3_000, 256, 30),
    (2_000, 1024, 30),
    (1_024, 1024, 30),  # n == total_steps
    (500, 1024, 30),  # n > total_steps
    (100, 10, 30),  # n <= dense_until
    (100, 31, 30),  # n == dense_until + 1
    (20, 1024, 30),  # total_steps <= dense_until
    (10, 3, 2),
    (2, 2, 30),
    (1, 1, 30),
    (1, 1024, 30),
    (50, 1, 30),  # n == 1
    (7, 5, 0),  # dense_until == 0
    (100_000, 64, 0),
    (10**7, 1024, 30),  # far over any real ceiling
]

# Cases with room for a real log-then-uniform tail (n < total_steps, big gap
# budget); the spacing-shape assertions only make sense here.
SPACING_GRID = [case for case in GRID if case[1] < case[0] and case[1] >= 64]


def test_v5_55_deterministic_no_seed():
    # V5.55: purely deterministic — no seed argument, no RNG dependence.
    # Perturbing global random state between calls must change nothing.
    random.seed(0)
    first = snapshot_steps(116_595, n=1024, dense_until=30)
    random.seed(12345)
    random.random()
    second = snapshot_steps(116_595, n=1024, dense_until=30)
    assert first == second
    assert all(type(s) is int for s in first)  # plain ints, manifest-ready


@pytest.mark.parametrize(("total_steps", "n", "dense_until"), GRID)
def test_v5_56_strictly_increasing_within_bounds(total_steps, n, dense_until):
    # V5.56: strictly increasing (hence unique) ints, all in [1, total_steps].
    steps = snapshot_steps(total_steps, n=n, dense_until=dense_until)
    assert all(a < b for a, b in zip(steps, steps[1:]))
    assert steps[0] >= 1
    assert steps[-1] <= total_steps


@pytest.mark.parametrize(("total_steps", "n", "dense_until"), GRID)
def test_v5_57_exact_count(total_steps, n, dense_until):
    # V5.57: exactly min(n, total_steps) entries — n == total_steps and
    # n > total_steps degrade to every step.
    steps = snapshot_steps(total_steps, n=n, dense_until=dense_until)
    assert len(steps) == min(n, total_steps)


def test_v5_58_dense_prefix_and_clamps():
    # V5.58: every step 1..dense_until present when the budget allows.
    for total_steps, n, dense_until in [(116_595, 1024, 30), (50_000, 512, 30)]:
        steps = snapshot_steps(total_steps, n=n, dense_until=dense_until)
        assert set(range(1, dense_until + 1)) <= set(steps)
    # Clamp: total_steps <= dense_until => the schedule is every step.
    assert snapshot_steps(20, n=1024, dense_until=30) == list(range(1, 21))
    # Clamp: n <= dense_until => prefix truncated to n-1, final step kept
    # (count and the final snapshot outrank the prefix).
    assert snapshot_steps(100, n=10, dense_until=30) == [*range(1, 10), 100]
    assert snapshot_steps(100, n=31, dense_until=30) == [*range(1, 31), 100]
    # dense_until == 0: no forced prefix, schedule still starts at step 1.
    assert snapshot_steps(1_000, n=16, dense_until=0)[0] == 1


@pytest.mark.parametrize(("total_steps", "n", "dense_until"), GRID)
def test_v5_59_first_and_final_step_present(total_steps, n, dense_until):
    # V5.59: final step always present; first step present whenever there are
    # >= 2 slots. At n == 1 the final snapshot outranks step 1.
    steps = snapshot_steps(total_steps, n=n, dense_until=dense_until)
    assert steps[-1] == total_steps
    if len(steps) >= 2:
        assert steps[0] == 1
    assert snapshot_steps(50, n=1) == [50]


def test_v5_60_invalid_inputs_raise():
    # V5.60: loud validation — a silently empty/short schedule would launch a
    # run with missing checkpoints.
    with pytest.raises(ValueError):
        snapshot_steps(0)
    with pytest.raises(ValueError):
        snapshot_steps(-5)
    with pytest.raises(ValueError):
        snapshot_steps(100, n=0)
    with pytest.raises(ValueError):
        snapshot_steps(100, n=1024, dense_until=-1)


@pytest.mark.parametrize(("total_steps", "n", "dense_until"), SPACING_GRID)
def test_v5_61_spacing_log_then_uniform(total_steps, n, dense_until):
    # V5.61: the spec's shape — "every step through ~30, then stretching,
    # uniform later" — without pinning the exact formula.
    steps = snapshot_steps(total_steps, n=n, dense_until=dense_until)
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    # Stretching: gaps never shrink by more than the unit rounding jitter.
    assert all(b >= a - 1 for a, b in zip(gaps, gaps[1:]))
    # Log-spaced early: unit stride continues just past the dense prefix
    # (the stretch is gradual, not a cliff at dense_until).
    assert steps[1] - steps[0] == 1
    if dense_until >= 1:
        assert {dense_until + 1, dense_until + 2} <= set(steps)
    # Uniform later: the last tenth of the gaps is near-constant. A pure
    # log tail would stretch by >= 2x over this window on these cases.
    tail = gaps[-max(3, len(gaps) // 10) :]
    assert max(tail) <= 1.6 * min(tail)
    # The largest gap lives in that late tail (up to downward ceil drift of 1).
    assert max(gaps) <= max(tail) + 1
