"""TRAIN-1 stopping tests — specs/02 §6.1 ``StoppingRule`` / ``ConvergenceTracker``.

Module under test (does not exist yet):

    from geode.train.stopping import StoppingRule, ConvergenceTracker

Derived from the spec only (specs/02-training-run.md §6.1):

    An eval improves iff ``(best_so_far - val_loss_nats) > eps_nats`` (STRICT;
    equality does NOT improve). Improvement updates ``best_nats`` and resets
    the stale counter; the k-th consecutive non-improving eval returns True
    (and keeps returning True). NaN input raises ValueError. ``best_nats`` is
    +inf before the first update.

All of V5.20 is exercised here. Numbers are chosen so the relevant
subtractions are exactly representable in float (1.0, 0.75, 0.5, 0.25) and so
no two quantities that could be confused are coincidentally equal.
"""

from __future__ import annotations

import pytest

from geode.train.stopping import (
    BehavioralStoppingRule,
    BehaviorTracker,
    ConvergenceTracker,
    StoppingRule,
)


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_stopping_plateau_stops_after_exactly_k(k):
    # V5.20: a plateau (improvements <= eps) stops after exactly k consecutive
    # non-improving evals -- not before, and exactly on the k-th.
    t = ConvergenceTracker(StoppingRule(eps_nats=0.1, k=k))
    assert t.best_nats == float("inf")  # +inf before first update
    # The first eval always improves (inf - finite > eps): sets best, no stop.
    assert t.update(1.0) is False
    assert t.best_nats == 1.0
    # A constant loss makes every later improvement 0 (<= eps): non-improving.
    for _ in range(1, k):  # k-1 non-improving: no stop
        assert t.update(1.0) is False
    assert t.update(1.0) is True  # k-th non-improving: stop


def test_stopping_improvement_resets_counter():
    # V5.20: any improvement > eps resets the stale counter (and best).
    t = ConvergenceTracker(StoppingRule(eps_nats=0.1, k=3))
    assert t.update(1.0) is False  # best <- 1.0, stale 0
    assert t.update(1.0) is False  # stale 1
    assert t.update(1.0) is False  # stale 2 (one short of k=3)
    assert t.stale_evals == 2
    # 1.0 - 0.5 == 0.5 > 0.1 -> improvement: best <- 0.5, stale resets to 0.
    assert t.update(0.5) is False
    assert t.stale_evals == 0
    assert t.best_nats == 0.5
    # Without the reset the 3rd non-improving eval above would have stopped;
    # instead it now takes a fresh run of k non-improving evals to stop.
    assert t.update(0.5) is False  # stale 1
    assert t.update(0.5) is False  # stale 2
    assert t.update(0.5) is True  # stale 3 -> stop


def test_stopping_eps_boundary_is_strict():
    # V5.20: an improvement of exactly eps does NOT count (strict >).
    t = ConvergenceTracker(StoppingRule(eps_nats=0.25, k=1))
    assert t.update(1.0) is False  # best <- 1.0
    # 1.0 - 0.75 == 0.25 exactly, and 0.25 > 0.25 is False -> non-improving,
    # so with k=1 this single non-improving eval triggers the stop.
    assert t.update(0.75) is True
    assert t.best_nats == 1.0  # 0.75 did NOT update best
    # Contrast: an improvement strictly greater than eps DOES count.
    t2 = ConvergenceTracker(StoppingRule(eps_nats=0.25, k=1))
    assert t2.update(1.0) is False  # best <- 1.0
    # 1.0 - 0.7 == 0.3 > 0.25 -> improvement -> no stop, best updates.
    assert t2.update(0.7) is False
    assert t2.best_nats == 0.7


def test_stopping_nan_raises():
    # V5.20: NaN input raises ValueError (before any update and after one).
    t = ConvergenceTracker(StoppingRule(eps_nats=0.1, k=2))
    with pytest.raises(ValueError):
        t.update(float("nan"))
    t2 = ConvergenceTracker(StoppingRule(eps_nats=0.1, k=2))
    assert t2.update(1.0) is False
    with pytest.raises(ValueError):
        t2.update(float("nan"))


def test_v5_41_min_nats_true_minimum():
    # V5.41: min_nats is the exact minimum of every value passed to update,
    # independent of the eps gate that freezes best_nats.
    t = ConvergenceTracker(StoppingRule(eps_nats=0.5, k=3))
    assert t.min_nats == float("inf")  # +inf before first update
    assert t.update(1.0) is False  # best <- 1.0, min <- 1.0
    # The gate compares against the FROZEN best, so the cumulative drop must
    # stay <= eps (0.125, 0.25, 0.375 vs best 1.0): best freezes, min tracks.
    assert t.update(0.875) is False  # stale 1
    assert t.update(0.75) is False  # stale 2
    assert t.update(0.625) is True  # stale 3 == k -> stop
    assert t.best_nats == 1.0  # eps-gated: never updated after the first eval
    assert t.min_nats == 0.625  # true minimum
    # Post-latch updates still record the min (the latch only pins the
    # return value); a higher value never raises the min.
    assert t.update(0.25) is True
    assert t.min_nats == 0.25
    assert t.update(0.375) is True
    assert t.min_nats == 0.25


def test_v5_41_nan_does_not_corrupt_min():
    # V5.41: NaN raises before any state change, so min_nats is untouched.
    t = ConvergenceTracker(StoppingRule(eps_nats=0.1, k=2))
    assert t.update(1.0) is False
    with pytest.raises(ValueError):
        t.update(float("nan"))
    assert t.min_nats == 1.0


def test_v5_42_min_steps_grace_never_counts():
    # V5.42: evals at step < min_steps can never stop training, consume
    # patience, or set best_nats — only min_nats tracks them.
    t = ConvergenceTracker(StoppingRule(eps_nats=0.1, k=2, min_steps=5000))
    # A k-long (and longer) plateau entirely inside the grace: no stop.
    for step in (1000, 2000, 3000, 4000):
        assert t.update(1.0, step=step) is False
    assert t.stale_evals == 0  # patience untouched
    assert t.best_nats == float("inf")  # best untouched
    assert t.min_nats == 1.0  # true min still tracks
    # First counted eval (step >= min_steps) sets best; the plateau then
    # needs a fresh k non-improving evals measured from here.
    assert t.update(1.0, step=5000) is False  # best <- 1.0
    assert t.best_nats == 1.0
    assert t.update(1.0, step=6000) is False  # stale 1
    assert t.update(1.0, step=7000) is True  # stale 2 == k -> stop


def test_v5_42_grace_does_not_plant_transient_best():
    # V5.42 motivation: a warm-start transient LOW value during grace must
    # not become the best that post-grace evals are measured against.
    t = ConvergenceTracker(StoppingRule(eps_nats=0.1, k=2, min_steps=3000))
    assert t.update(0.5, step=1000) is False  # transient, grace: ignored
    assert t.best_nats == float("inf")
    assert t.min_nats == 0.5  # but the true min keeps it
    # Post-grace evals at 0.9 would be "non-improving" vs a planted 0.5
    # best; against the untouched tracker they set best and train on.
    assert t.update(0.9, step=3000) is False
    assert t.best_nats == 0.9
    assert t.stale_evals == 0


def test_v5_42_min_steps_zero_reproduces_pre_grace_behavior():
    # V5.42: min_steps=0 (the default) with or without a step argument
    # behaves exactly like the pre-grace tracker.
    t_default = ConvergenceTracker(StoppingRule(eps_nats=0.1, k=2))
    t_explicit = ConvergenceTracker(StoppingRule(eps_nats=0.1, k=2, min_steps=0))
    for tracker in (t_default, t_explicit):
        assert tracker.update(1.0) is False  # no step passed: fine
        assert tracker.update(1.0, step=0) is False  # step 0 counts
        assert tracker.update(1.0, step=1) is True  # stale 2 == k


def test_v5_42_min_steps_without_step_raises():
    # V5.42: min_steps > 0 needs the step to apply the grace — omitting it
    # raises instead of silently counting a grace-period eval.
    t = ConvergenceTracker(StoppingRule(eps_nats=0.1, k=2, min_steps=1000))
    with pytest.raises(ValueError):
        t.update(1.0)
    assert t.update(1.0, step=1000) is False  # tracker still usable


def test_stopping_latches_true_after_stop():
    # V5.20 / §6.1: the k-th non-improving eval returns True "and keeps
    # returning True". Ruling: the latch is unconditional — True even when a
    # later value WOULD have counted as an improvement. (A reading limited to
    # non-improving evals would make the clause redundant: a continued
    # plateau already returns True via the consecutive-count rule itself.)
    t = ConvergenceTracker(StoppingRule(eps_nats=0.1, k=2))
    assert t.update(1.0) is False
    assert t.update(1.0) is False  # stale 1
    assert t.update(1.0) is True  # stale 2 == k -> stop
    assert t.update(1.0) is True  # stays stopped (non-improving)
    # 1.0 - 0.2 == 0.8 > eps: pre-stop this would improve; post-stop the
    # tracker must still report stopped.
    assert t.update(0.2) is True
    assert t.update(0.2) is True  # and keeps returning True after that


# --- V5.44: behavioral stopping rule (runs 3-4, specs/02 §6) ---------------


@pytest.mark.parametrize("k", [1, 2, 3])
def test_v5_44_behavior_stops_on_kth_consecutive_hit(k):
    # V5.44: the k-th consecutive eval >= threshold stops -- not before,
    # exactly on the k-th. The threshold is INCLUSIVE (">= 99%").
    t = BehaviorTracker(BehavioralStoppingRule(threshold=0.99, k=k))
    assert t.update(0.5) is False  # miss: not counted
    for _ in range(k - 1):
        assert t.update(0.99) is False  # inclusive hit, still short of k
    assert t.update(0.99) is True  # k-th consecutive hit -> stop


def test_v5_44_miss_resets_consecutive_count():
    # V5.44: a sub-threshold eval resets the run of hits; "consecutive"
    # means consecutive.
    t = BehaviorTracker(BehavioralStoppingRule(threshold=0.99, k=3))
    assert t.update(1.0) is False  # hit 1
    assert t.update(1.0) is False  # hit 2
    assert t.update(0.98) is False  # miss: reset (0.98 < 0.99, exclusive miss)
    assert t.hits == 0
    assert t.update(1.0) is False  # hit 1 again
    assert t.update(1.0) is False  # hit 2
    assert t.update(1.0) is True  # hit 3 -> stop


def test_v5_44_latches_and_tracks_best_rate():
    # V5.44: latched after stop (mirrors ConvergenceTracker); best_rate is
    # the exact running max over EVERY update, -inf before the first.
    t = BehaviorTracker(BehavioralStoppingRule(threshold=0.99, k=1))
    assert t.best_rate == float("-inf")
    assert t.update(0.25) is False
    assert t.best_rate == 0.25
    assert t.update(1.0) is True  # k=1: first hit stops
    assert t.update(0.0) is True  # latched despite a miss-value
    assert t.best_rate == 1.0  # max unaffected by the post-stop miss


def test_v5_44_nan_raises():
    t = BehaviorTracker(BehavioralStoppingRule(threshold=0.99, k=2))
    with pytest.raises(ValueError):
        t.update(float("nan"))
    assert t.update(1.0) is False  # tracker still usable
