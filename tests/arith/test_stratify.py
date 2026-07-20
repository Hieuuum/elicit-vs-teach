"""geode.arith.stratify — capacity-aware water-fill allocation (V5.3).

V5.3: the allocation keeps every unique question in cells too small to reach an
even share, then splits the remainder as evenly as the capacities allow. Total
is exact; no cell exceeds its capacity. The full-scale cases anchor the concrete
per-cell numbers the owner signed off on.
"""

from __future__ import annotations

import pytest

from geode.arith.stratify import allocate, capacity

CELLS = [(x, y) for x in range(1, 5) for y in range(1, 5)]


def _capacities(n_ops: int, n_probe: int) -> dict[tuple[int, int], int]:
    return {(x, y): capacity(x, y, n_ops, n_probe) for x, y in CELLS}


def test_v5_3_capacity_counts_triples_minus_probe():
    assert capacity(1, 1, 2) == 162  # 9*9*2 ordered add/sub triples
    assert capacity(1, 1, 2, 64) == 98  # minus the 64 probe triples in the cell
    assert capacity(1, 1, 1) == 81  # one op (mult)
    assert capacity(2, 3, 1) == 81_000  # 90*900


def test_v5_3_allocate_sums_to_total_and_respects_caps():
    caps = _capacities(n_ops=2, n_probe=64)
    alloc = allocate(1_000_000, caps)
    assert sum(alloc.values()) == 1_000_000
    assert all(alloc[c] <= caps[c] for c in CELLS)


def test_v5_3_small_cells_take_all_and_free_cells_are_even():
    # Synthetic map: three tiny cells below any even share, one huge cell.
    caps = {(1, 1): 10, (1, 2): 20, (2, 1): 30, (4, 4): 10_000}
    alloc = allocate(1000, caps)
    assert alloc[(1, 1)] == 10  # taken whole
    assert alloc[(1, 2)] == 20
    assert alloc[(2, 1)] == 30
    assert alloc[(4, 4)] == 1000 - 60  # absorbs the rest
    assert sum(alloc.values()) == 1000


def test_v5_3_remainder_spread_evenly_over_free_cells():
    # 100 over four cells that all have room -> 25 each, no capping.
    alloc = allocate(100, {(1, 1): 999, (1, 2): 999, (2, 1): 999, (2, 2): 999})
    assert sorted(alloc.values()) == [25, 25, 25, 25]
    # 102 over the same -> two cells get 26, two get 25 (even ±1).
    alloc = allocate(102, {(1, 1): 999, (1, 2): 999, (2, 1): 999, (2, 2): 999})
    assert sorted(alloc.values()) == [25, 25, 26, 26]


def test_v5_3_raises_on_insufficient_capacity():
    with pytest.raises(ValueError):
        allocate(1000, {(1, 1): 100, (1, 2): 100})


def test_v5_3_addsub_full_scale_matches_owner_plan():
    # add/sub datasets: two ops, probe removes 64 triples per cell.
    alloc = allocate(1_000_000, _capacities(n_ops=2, n_probe=64))
    # Six small cells taken whole.
    assert alloc[(1, 1)] == 98
    assert alloc[(1, 2)] == alloc[(2, 1)] == 1_556
    assert alloc[(2, 2)] == alloc[(1, 3)] == alloc[(3, 1)] == 16_136
    # Ten free cells split the 948,382 remainder evenly: eight @ 94,838, two @ 94,839.
    free = [alloc[c] for c in CELLS if c not in {(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)}]
    assert set(free) == {94_838, 94_839}
    assert free.count(94_839) == 2
    assert sum(alloc.values()) == 1_000_000


def test_v5_3_mult_full_scale_matches_owner_plan():
    # mult dataset: one op, no probe removal (probe is add/sub only).
    alloc = allocate(1_000_000, _capacities(n_ops=1, n_probe=0))
    assert alloc[(1, 1)] == 81
    assert alloc[(1, 2)] == alloc[(2, 1)] == 810
    assert alloc[(2, 2)] == alloc[(1, 3)] == alloc[(3, 1)] == 8_100
    # Four mid cells saturate at capacity (can't reach the even share).
    assert alloc[(1, 4)] == alloc[(4, 1)] == alloc[(2, 3)] == alloc[(3, 2)] == 81_000
    # Six big cells split the remainder: five @ 108,333, one @ 108,334.
    big = [alloc[c] for c in [(2, 4), (4, 2), (3, 3), (3, 4), (4, 3), (4, 4)]]
    assert set(big) == {108_333, 108_334}
    assert big.count(108_334) == 1
    assert sum(alloc.values()) == 1_000_000
