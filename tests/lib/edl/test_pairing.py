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
