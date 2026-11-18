"""V1.12 — EDL/D scaling-signature call (``geode.edl.edl_signature``, specs/01 §4).

Pure-function properties on hand-built curves; no model, CPU, instant.
"""

from __future__ import annotations

import math

import pytest

from geode.edl import edl_signature, half_rise_n, knowledge_read

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


@pytest.mark.parametrize(
    "curves", [[_elicit(s) for s in (0.9, 1.0, 1.1)], [_teach(s) for s in (0.9, 1.0, 1.1)]]
)
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


# ------------------------------------------------------------------ V1.13 knowledge-specific read
KN = [8, 16, 32, 64, 128, 256, 512]


def test_v1_13_the_five_calibration_shapes_read_as_named():
    teach = [
        [0.1, 0.05, 0.4, 1.2, 1.6, 1.3, 0.9],
        [0.2, 0.1, 0.5, 1.1, 1.5, 1.2, 0.8],
    ]  # TinyStories arms
    r = knowledge_read(KN, teach)
    assert r["read"] == "teach" and r["max_n"] == 128 and r["block_ns"] == [8, 16, 32]
    gate = [
        [0.6, 0.3, 0.1, -0.1, -0.2, -0.3, -0.3],
        [0.5, 0.2, 0.05, -0.05, -0.2, -0.3, -0.3],
    ]  # LAT, Llama base
    r = knowledge_read(KN, gate)
    assert r["read"] == "gate" and r["crossing_n"] == 64
    elicit = [
        [-1.2, -1.0, -0.6, -0.3, -0.2, -0.1, -0.1],
        [-1.1, -0.9, -0.5, -0.3, -0.2, -0.1, -0.05],
    ]  # Zephyr
    assert knowledge_read(KN, elicit)["read"] == "elicit"
    assert (
        knowledge_read(KN, [[0.01, -0.02, 0.03, 0.0, -0.01, 0.02, 0.0]])["read"] == "none"
    )  # installed Llama
    # seeds that scatter by bits around a small mean: within two standard errors -> mixed, not a call
    noisy = [
        [1.9, -1.5, -1.2, -1.1, -0.7, -0.3, -0.2],
        [0.1, -1.6, -1.2, -1.1, -0.7, -0.3, -0.2],
        [-0.7, 2.0, 1.0, -1.1, -0.7, -0.3, -0.2],
    ]
    assert knowledge_read(KN, noisy)["read"] == "mixed"
    # the same scatter averaged over the early block, when it agrees, is a call
    assert (
        knowledge_read(
            KN,
            [[0.3, -1.6, -1.2, -1.1, -0.7, -0.3, -0.2], [-0.8, -1.5, -1.2, -1.1, -0.7, -0.3, -0.2]],
        )["read"]
        == "elicit"
    )


def test_v1_13_a_hump_one_seed_does_not_share_or_inside_the_early_block_is_not_teach():
    shared = [[0.1, 0.1, 0.5, 1.0, 0.8, 0.5, 0.3]]
    assert knowledge_read(KN, shared)["read"] == "teach"
    assert knowledge_read(KN, shared + [[0.1, 0.1, 0.1, 0.05, 0.0, 0.0, 0.0]])["read"] != "teach"
    early = [
        [0.2, 1.4, 2.0, 0.1, -0.1, -0.3, -0.3]
    ]  # a peak inside the first three sizes: an unlock, not a hump
    r = knowledge_read(KN, early)
    assert r["read"] == "gate" and r["crossing_n"] == 128
    small = [[0.43, 0.45, 0.52, 0.4, 0.22, 0.1, 0.06]]  # the mix arm
    r = knowledge_read(KN, small)
    assert (
        r["read"] == "gate"
        and r["crossing_n"] is None
        and r["rise_nats"] == pytest.approx(0.52 - 0.4667, abs=1e-3)
    )


def test_v1_13_a_positive_offset_turns_elicit_into_gate_and_nothing_else():
    elicit = [[-1.2, -1.0, -0.6, -0.3, -0.2, -0.1, -0.1]]  # rises by 1.1 toward zero: still elicit
    assert knowledge_read(KN, elicit)["read"] == "elicit"
    flat = [[-0.5, -0.45, -0.4, -0.38, -0.36, -0.35, -0.35]]
    r0, r1 = knowledge_read(KN, flat), knowledge_read(KN, [[y + 1.0 for y in flat[0]]])
    assert r0["read"] == "elicit" and r1["read"] == "gate" and r1["crossing_n"] is None
    assert r1["rise_nats"] == pytest.approx(r0["rise_nats"])


def test_v1_13_half_rise_size_and_guards():
    assert half_rise_n([0, 8, 16, 32, 64], [0.87, 1.0, 1.13, 1.51, 1.56]) == 32  # CB+LAT on B
    assert half_rise_n([0, 100, 1000, 10000, 100000], [0.1, 0.1, 0.1, 4.0, 8.0]) == 100000
    assert half_rise_n([0, 100, 1000, 10000, 100000], [0.1, 0.1, 4.1, 7.0, 8.0]) == 1000
    assert half_rise_n([8, 16, 32], [1.5, 1.52, 1.49]) is None
    with pytest.raises(ValueError):
        half_rise_n([8, 16], [1.0])
    with pytest.raises(ValueError):
        knowledge_read(KN, [[0.0] * 6])
    with pytest.raises(ValueError):
        knowledge_read(KN, [[math.nan] * 7])
