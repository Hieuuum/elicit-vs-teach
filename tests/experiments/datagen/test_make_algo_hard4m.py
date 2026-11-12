"""make_algo_hard4m.py: the hard-weighted 4M set for continuing the ts1b children.

Silent-failure risk: a wrong allocation would quietly train on a different mix
than the owner chose; a leak of test, EAP-IG, reserve or already-trained
problems would confound the continued-training comparison with nothing
crashing. These tests check the pure helpers on small in-process inputs; the
real build re-checks every property on the written file.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from collections import Counter

import pytest

from tests._scriptloader import repo_root

REPO_ROOT = repo_root()
_SCRIPT = REPO_ROOT / "experiments" / "training-run" / "datagen" / "make_algo_hard4m.py"
_spec = importlib.util.spec_from_file_location("make_algo_hard4m", _SCRIPT)
mah = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mah
_spec.loader.exec_module(mah)


def _in_cell(a: int, b: int, cell) -> bool:
    (lx, hx), (ly, hy) = mah.DIGIT_BANDS[cell[0]], mah.DIGIT_BANDS[cell[1]]
    return lx <= a <= hx and ly <= b <= hy


# --- the owner's mix ----------------------------------------------------------


def test_mix_is_the_owner_choice():
    m = {f"{c[0]}x{c[1]}": s for c, s in mah.MIX.items()}
    assert m == {
        "4x4": 0.40,
        "3x3": 0.15,
        "3x4": 0.15,
        "4x3": 0.15,
        "2x4": 0.05,
        "4x2": 0.05,
        "1x4": 0.0125,
        "2x3": 0.0125,
        "3x2": 0.0125,
        "4x1": 0.0125,
    }
    assert sum(mah.MIX.values()) == pytest.approx(1.0)


def test_no_small_operand_cells_in_mix():
    small = {(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1)}
    assert not small & set(mah.MIX)


# --- allocation -----------------------------------------------------------------


@pytest.mark.parametrize("n", [4_000_000, 1_000, 997, 20, 1])
def test_allocation_sums_to_n(n):
    assert sum(mah.allocation(n, mah.MIX).values()) == n


def test_allocation_4m_matches_shares_exactly():
    alloc = mah.allocation(4_000_000, mah.MIX)
    per_cell = Counter()
    for (cell, _op), v in alloc.items():
        per_cell[cell] += v
    assert per_cell[(4, 4)] == 1_600_000
    assert per_cell[(3, 3)] == per_cell[(3, 4)] == per_cell[(4, 3)] == 600_000
    assert per_cell[(2, 4)] == per_cell[(4, 2)] == 200_000
    for c in mah.SMALL_CELLS:
        assert per_cell[c] == 50_000


def test_allocation_splits_ops_evenly():
    alloc = mah.allocation(4_000_000, mah.MIX)
    for cell in mah.MIX:
        assert alloc[(cell, "+")] == alloc[(cell, "-")]


def test_allocation_odd_n_ops_differ_by_at_most_one():
    alloc = mah.allocation(1_001, mah.MIX)
    for cell in mah.MIX:
        assert abs(alloc[(cell, "+")] - alloc[(cell, "-")]) <= 1


def test_allocation_deterministic():
    assert mah.allocation(12_345, mah.MIX) == mah.allocation(12_345, mah.MIX)


def test_allocation_rejects_bad_mix():
    with pytest.raises(ValueError):
        mah.allocation(100, {(4, 4): 0.5, (3, 3): 0.4})


def test_allocation_never_negative():
    assert all(v >= 0 for v in mah.allocation(7, mah.MIX).values())


# --- sample_new -------------------------------------------------------------------


def test_sample_new_distinct_in_cell_and_avoids_blocked():
    rng = random.Random(0)
    blocked = {(a, b) for a in range(10, 99) for b in range(100, 200)}
    pairs = mah.sample_new((2, 3), 500, blocked, rng)
    assert len(pairs) == len(set(pairs)) == 500
    assert all(_in_cell(a, b, (2, 3)) for a, b in pairs)
    assert not set(pairs) & blocked


def test_sample_new_ignores_blocked_pairs_outside_the_cell():
    rng = random.Random(0)
    # 1x4 has 9 * 9000 = 81,000 pairs; blocking 4x4 pairs must not reduce it
    blocked = {(1000 + i, 1000) for i in range(100)}
    pairs = mah.sample_new((1, 4), 100, blocked, rng)
    assert len(pairs) == 100


def test_sample_new_raises_when_asking_more_than_free():
    rng = random.Random(0)
    # every 1x2 pair except (9, 99): one free pair, so asking for 2 must fail
    blocked = {(a, b) for a in range(1, 10) for b in range(10, 100)} - {(9, 99)}
    with pytest.raises(ValueError):
        mah.sample_new((1, 2), 2, blocked, rng)


def test_sample_new_deterministic_per_seed():
    a = mah.sample_new((4, 4), 50, set(), random.Random(7))
    b = mah.sample_new((4, 4), 50, set(), random.Random(7))
    c = mah.sample_new((4, 4), 50, set(), random.Random(8))
    assert a == b and a != c


# --- sample_small -----------------------------------------------------------------


def _cell_pairs(cell):
    (lx, hx), (ly, hy) = mah.DIGIT_BANDS[cell[0]], mah.DIGIT_BANDS[cell[1]]
    return [(a, b) for a in range(lx, hx + 1) for b in range(ly, hy + 1)]


def test_sample_small_takes_only_unseen_when_enough():
    cell = (2, 3)
    allp = _cell_pairs(cell)
    seen = set(allp[: len(allp) // 2])
    pairs, rep = mah.sample_small(cell, 1_000, set(), seen, random.Random(0))
    assert rep == 0
    assert not set(pairs) & seen
    assert len(set(pairs)) == 1_000


def test_sample_small_takes_all_unseen_then_repeats():
    cell = (2, 3)
    allp = _cell_pairs(cell)
    hard = set(allp[:1_000])
    seen = set(allp[1_000:80_000])
    n_unseen = len(allp) - 80_000
    k = n_unseen + 500
    pairs, rep = mah.sample_small(cell, k, hard, seen, random.Random(0))
    assert rep == 500
    assert len(pairs) == len(set(pairs)) == k
    unseen = set(allp) - hard - seen
    assert unseen <= set(pairs)  # every unseen pair taken
    assert len(set(pairs) & seen) == 500
    assert not set(pairs) & hard


def test_sample_small_never_returns_hard_blocked():
    cell = (3, 2)
    allp = _cell_pairs(cell)
    hard = set(allp[::3])
    seen = set(allp[1::3])
    pairs, _ = mah.sample_small(cell, len(allp) - len(hard), hard, seen, random.Random(1))
    assert not set(pairs) & hard


def test_sample_small_raises_past_allowed():
    cell = (4, 1)
    allp = _cell_pairs(cell)
    hard = set(allp[10:])
    with pytest.raises(ValueError):
        mah.sample_small(cell, 11, hard, set(), random.Random(0))


# --- reserve_pairs ------------------------------------------------------------------


def test_reserve_pairs_distinct_in_cell_and_avoid_both_orders():
    rng = random.Random(0)
    blocked = {(100 + i, "+", 1000 + i) for i in range(300)} | {
        (2000 + i, "+", 200 + i) for i in range(300)
    }
    res = mah.reserve_pairs(2_000, (3, 4), blocked, rng)
    assert len(res) == len(set(res)) == 2_000
    assert all(_in_cell(a, b, (3, 4)) for a, b in res)
    for a, b in res:
        assert (a, "+", b) not in blocked and (b, "+", a) not in blocked


def test_reserve_pairs_respects_reversed_block():
    # the only free pair in a tiny region is excluded when its twin is blocked
    rng = random.Random(0)
    res = mah.reserve_pairs(50, (3, 4), {(5000, "+", 500)}, rng)
    assert (500, 5000) not in res


# --- build_triples (small n, real cells) ------------------------------------------------


def _fresh_fixture():
    rng = random.Random(3)
    fresh = set()
    for cell in mah.MIX:
        for op in mah.OPS:
            for a, b in mah.sample_new(cell, 40, set(), rng):
                fresh.add((a, op, b))
    return fresh


def test_build_triples_exact_alloc_unique_and_disjoint():
    fresh = _fresh_fixture()
    hard = {(1234, "+", 5678), (5678, "+", 1234), (12, "-", 345)}
    triples, alloc, repeats = mah.build_triples(2_000, 0, hard, fresh)
    assert len(triples) == len(set(triples)) == 2_000
    got = Counter(((mah.digits(a), mah.digits(b)), op) for a, op, b in triples)
    assert dict(got) == {k: v for k, v in alloc.items() if v}
    assert not set(triples) & hard
    outside_small = {
        t for t in triples if (mah.digits(t[0]), mah.digits(t[2])) not in mah.SMALL_CELLS
    }
    assert not outside_small & fresh
    assert all(v == 0 for v in repeats.values())  # small cells have room at this n


def test_build_triples_labels_ops_and_cells_valid():
    triples, _, _ = mah.build_triples(500, 1, set(), set())
    for a, op, b in triples:
        assert op in mah.OPS
        assert (mah.digits(a), mah.digits(b)) in mah.MIX


def test_build_triples_deterministic_and_seed_sensitive():
    t1, _, _ = mah.build_triples(300, 5, set(), set())
    t2, _, _ = mah.build_triples(300, 5, set(), set())
    t3, _, _ = mah.build_triples(300, 6, set(), set())
    assert t1 == t2 and t1 != t3


def test_build_triples_shuffled_not_grouped_by_cell():
    triples, _, _ = mah.build_triples(2_000, 0, set(), set())
    first = [(mah.digits(a), mah.digits(b)) for a, _, b in triples[:200]]
    assert len(set(first)) >= 5  # sorted-by-cell output would give one or two cells
