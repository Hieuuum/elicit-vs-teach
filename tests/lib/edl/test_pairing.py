"""V1.14: pairing accuracy, the pairing gap's score."""

from __future__ import annotations

import random

import pytest

from geode.edl.pairing import (
    bootstrap_ci,
    make_pairs,
    needed_cross,
    pair_accuracy,
    pairing_call,
    pairing_read,
    pairing_score,
)


def _cross(n, f):
    return {(q, a): f(q, a) for q in range(n) for a in range(n) if q != a}


def test_v1_14_planted_pairing_reads_one():
    n = 30
    r = pairing_read([1.0] * n, _cross(n, lambda q, a: 2.0), make_pairs(n, 4), seed=0)
    assert r["acc"] == 1.0 and r["s"] == 1.0 and r["call"] == "elicit"


def test_v1_14_prompt_and_response_effects_cancel_exactly():
    rng = random.Random(1)
    n = 40
    p, r = [rng.gauss(0, 10) for _ in range(n)], [rng.gauss(0, 10) for _ in range(n)]
    true = [p[i] + r[i] for i in range(n)]
    res = pair_accuracy(true, _cross(n, lambda q, a: p[q] + r[a]), make_pairs(n, 5))
    assert res["acc"] == 0.5
    planted = pair_accuracy(true, _cross(n, lambda q, a: p[q] + r[a] + 0.01), make_pairs(n, 5))
    assert planted["acc"] == 1.0


def test_v1_14_noise_reads_near_chance_and_is_teach():
    rng = random.Random(2)
    n = 200
    true = [rng.gauss(5, 1) for _ in range(n)]
    cross = _cross(n, lambda q, a: 0.0)
    for k in cross:
        cross[k] = rng.gauss(5, 1)
    r = pairing_read(true, cross, make_pairs(n, 4), seed=0)
    assert 0.45 < r["acc"] < 0.55 and r["call"] == "teach"


def test_v1_14_scaling_losses_leaves_accuracy_unchanged():
    rng = random.Random(3)
    n = 50
    true = [rng.random() for _ in range(n)]
    cross = {k: rng.random() for k in _cross(n, lambda q, a: 0)}
    pairs = make_pairs(n, 4)
    a = pair_accuracy(true, cross, pairs)["acc"]
    b = pair_accuracy([10 * x for x in true], {k: 10 * v for k, v in cross.items()}, pairs)["acc"]
    assert a == b


def test_v1_14_pairs_respect_kinds_blocks_and_are_unique():
    ps = make_pairs(10, 3)
    assert len(ps) == len(set(ps)) == 30 and all(i < j for i, j in ps)
    groups = [i % 3 for i in range(12)]
    assert all(groups[i] == groups[j] for i, j in make_pairs(12, 4, groups=groups))
    blocks = [i // 4 for i in range(16)]  # 4 prompts x 4 responses
    pb = make_pairs(16, 3, blocks=blocks)
    assert pb and all(blocks[i] != blocks[j] for i, j in pb)
    assert len(pb) == 16 * 3   # round-robin layout: the 3 nearest neighbours are other blocks
    assert set(needed_cross([(0, 1)])) == {(0, 1), (1, 0)}


def test_v1_14_score_clip_ceiling_and_call():
    assert pairing_score(0.75) == pytest.approx(0.5)
    assert pairing_score(0.4) == 0.0
    assert pairing_score(0.75, ceiling_acc=0.75) == pytest.approx(1.0)
    assert pairing_score(0.6, ceiling_acc=0.8) == pytest.approx(0.2 / 0.6)
    assert pairing_score(0.6, ceiling_acc=0.5) is None
    assert pairing_call((0.51, 0.6)) == "elicit" and pairing_call((0.49, 0.6)) == "teach"


def test_v1_14_bootstrap_brackets_mean_and_gap_per_token():
    lo, hi = bootstrap_ci([0.0, 1.0] * 50, seed=1, reps=500)
    assert lo < 0.5 < hi
    r = pairing_read([1.0, 1.0], {(0, 1): 2.0, (1, 0): 2.0}, [(0, 1)], seed=0, tokens=[2, 2])
    assert r["gap_nats_per_pair"] == pytest.approx(2.0) and r["gap_nats_per_token"] == pytest.approx(0.5)


# ------------------------------------------------------------------ V1.15: against a control set
from geode.edl.pairing import control_read, margins, pair_losses  # noqa: E402


def _set(n, f):
    return [f(i, i) for i in range(n)], _cross(n, f)


def test_v1_15_identical_data_and_control_read_exactly_half():
    rng = random.Random(4)
    n = 30
    tab = {(q, a): rng.gauss(0, 3) for q in range(n) for a in range(n)}
    d = _set(n, lambda q, a: tab[(q, a)])
    r = control_read(d, d, make_pairs(n, 4), seed=0)
    assert r["acc"] == 0.5 and r["s"] == 0.0 and r["call"] == "teach"


def test_v1_15_planted_excess_reads_one_and_shared_shortcut_cancels():
    rng = random.Random(5)
    n = 30
    short = {(q, a): rng.gauss(0, 3) for q in range(n) for a in range(n)}   # a link both sets share
    ctrl = _set(n, lambda q, a: short[(q, a)])
    data = _set(n, lambda q, a: short[(q, a)] + (0.0 if q == a else 0.5))  # plus the information
    r = control_read(data, ctrl, make_pairs(n, 4), seed=0, tokens=([1] * n, [1] * n))
    assert r["acc"] == 1.0 and r["call"] == "elicit"
    assert r["m21_excess_nats_per_token"] == pytest.approx(0.5)


def test_v1_15_prompt_and_response_shifts_of_either_set_cancel():
    rng = random.Random(6)
    n = 25
    base = {(q, a): rng.gauss(0, 1) for q in range(n) for a in range(n)}
    p1, r1 = [rng.gauss(0, 9) for _ in range(n)], [rng.gauss(0, 9) for _ in range(n)]
    p2, r2 = [rng.gauss(0, 9) for _ in range(n)], [rng.gauss(0, 9) for _ in range(n)]
    pairs = make_pairs(n, 4)
    plain = control_read(_set(n, lambda q, a: base[(q, a)] + (q == a) * -1.0), _set(n, lambda q, a: base[(q, a)]), pairs, seed=0)
    shifted = control_read(_set(n, lambda q, a: base[(q, a)] + (q == a) * -1.0 + p1[q] + r1[a]),
                           _set(n, lambda q, a: base[(q, a)] + p2[q] + r2[a]), pairs, seed=0)
    assert plain["acc"] == pytest.approx(shifted["acc"])


def test_v1_15_scaling_and_margins_and_losses_plumbing():
    rng = random.Random(7)
    n = 20
    d = _set(n, lambda q, a: rng.random())
    c = _set(n, lambda q, a: rng.random())
    pairs = make_pairs(n, 3)
    a = control_read(d, c, pairs, seed=0)["acc"]
    b = control_read(([10 * x for x in d[0]], {k: 10 * v for k, v in d[1].items()}),
                     ([10 * x for x in c[0]], {k: 10 * v for k, v in c[1].items()}), pairs, seed=0)["acc"]
    assert a == b
    assert margins([1.0, 1.0], {(0, 1): 2.0, (1, 0): 3.0}, [(0, 1)]) == [3.0]
    t, cnt, cross = pair_losses(lambda idx: ([float(q * 10 + a) for q, a in idx], [1] * len(idx)), 3, [(0, 1)])
    assert t == [0.0, 11.0, 22.0] and cnt == [1, 1, 1] and cross == {(0, 1): 1.0, (1, 0): 10.0}


# ------------------------------------------------------------------ V1.16: the trait context
from geode.edl.pairing import auc_read  # noqa: E402


def test_v1_16_identical_sets_read_half_and_shift_reads_above():
    rng = random.Random(8)
    xs = [rng.gauss(0, 1) for _ in range(200)]
    r = auc_read(xs, list(xs), seed=0, reps=300)
    assert r["acc"] == pytest.approx(0.5) and r["call"] == "teach"
    sh = auc_read([x + 3 for x in xs], xs, seed=0, reps=300)
    assert sh["acc"] > 0.95 and sh["call"] == "elicit"
    assert sh["gain_excess"] == pytest.approx(3.0)


def test_v1_16_monotone_rescaling_and_ties_and_empty():
    rng = random.Random(9)
    a, b = [rng.random() for _ in range(50)], [rng.random() for _ in range(60)]
    r1 = auc_read(a, b, seed=0, reps=200)["acc"]
    r2 = auc_read([10 * x**3 for x in a], [10 * x**3 for x in b], seed=0, reps=200)["acc"]
    assert r1 == pytest.approx(r2)
    assert auc_read([1.0, 1.0], [1.0], seed=0, reps=50)["acc"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        auc_read([], [1.0], seed=0)
