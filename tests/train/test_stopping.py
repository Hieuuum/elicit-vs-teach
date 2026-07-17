"""TRAIN-1 stopping tests — specs/05 §6.1 ``StoppingRule`` / ``ConvergenceTracker``.

Module under test (does not exist yet):

    from geode.train.stopping import StoppingRule, ConvergenceTracker

Derived from the spec only (specs/05-elicit-vs-teach.md §6.1):

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

from geode.train.stopping import ConvergenceTracker, StoppingRule


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
