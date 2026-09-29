"""V1.12 — EDL/D scaling-signature call (``geode.edl.edl_signature``, specs/01 §4).

Pure-function properties on hand-built curves; no model, CPU, instant.
"""

from __future__ import annotations

import math

import pytest

from geode.edl import edl_signature

NS = [8, 16, 32, 64, 128, 256, 512]


def _elicit(scale: float) -> list[float]:
    return [scale * 40.0 / n for n in NS]  # a fixed cost amortised over n


def _teach(scale: float) -> list[float]:
    return [scale * y for y in (0.2, 0.4, 0.9, 1.5, 1.3, 1.0, 0.8)]  # hump, then decline


def test_v1_12_elicit_curves_read_decreasing():
    # V1.12: monotonically decreasing curves in every seed read "decreasing", all seeds agreeing.
    r = edl_signature(NS, [_elicit(s) for s in (0.9, 1.0, 1.1)])
    assert r["call"] == "decreasing"
    assert r["agree"] == r["n_seeds"] == 3
    assert r["max_rise"] < 0


def test_v1_12_teach_hump_reads_increasing():
    # V1.12: a rise past tol shared by every seed reads "increasing"; rise_at names the steepest pair.
    r = edl_signature(NS, [_teach(s) for s in (0.9, 1.0, 1.1)])
    assert r["call"] == "increasing"
    assert r["rise_at"] == (32, 64)
    assert r["seed_calls"] == ["increasing"] * 3


def test_v1_12_rise_not_shared_by_every_seed_is_mixed():
    # V1.12: two seeds rise, one falls at the same pair; the mean still rises past tol. That is
    # seed-dependent, so neither "increasing" nor "decreasing".
    ns = [1, 2, 3, 4]
    curves = [[3.0, 1.0, 2.5, 0.0], [3.0, 1.0, 2.5, 0.0], [3.0, 1.0, 0.8, 0.0]]
    r = edl_signature(ns, curves)
    assert r["call"] == "mixed"
    assert r["seed_calls"] == ["increasing", "increasing", "decreasing"]
    assert r["agree"] == 0


def test_v1_12_wiggles_below_tol_read_flat():
    # V1.12: changes no larger than abs_tol are not a shape.
    r = edl_signature([1, 2, 3, 4, 5], [[1.0, 1.005, 0.998, 1.004, 1.0]])
    assert r["call"] == "flat"
    assert r["tol"] == pytest.approx(0.01)


@pytest.mark.parametrize("curves", [[_elicit(s) for s in (0.9, 1.0, 1.1)], [_teach(s) for s in (0.9, 1.0, 1.1)]])
def test_v1_12_offset_and_unit_invariance(curves):
    # V1.12: a constant shift (another floor level) and nats -> bits leave the call unchanged;
    # max_rise shifts not at all and scales with the unit.
    base = edl_signature(NS, curves)
    shifted = edl_signature(NS, [[y - 0.37 for y in c] for c in curves])
    bits = edl_signature(NS, [[y / math.log(2) for y in c] for c in curves])
    assert shifted["call"] == bits["call"] == base["call"]
    assert shifted["max_rise"] == pytest.approx(base["max_rise"])
    assert bits["max_rise"] == pytest.approx(base["max_rise"] / math.log(2))


def test_v1_12_guards():
    # V1.12: unsorted sizes, a short curve, a NaN, and no curves raise; under three sizes is no call.
    with pytest.raises(ValueError):
        edl_signature([16, 8, 32], [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        edl_signature([8, 16, 32], [[1.0, 2.0]])
    with pytest.raises(ValueError):
        edl_signature([8, 16, 32], [[1.0, float("nan"), 3.0]])
    with pytest.raises(ValueError):
        edl_signature([8, 16, 32], [])
    assert edl_signature([8, 16], [[2.0, 1.0]])["call"] == "too few sizes"
