"""geode.arith.validate — artifact validators (V5.1, V5.2, V5.3).

These run over a materialised dataset in ``make_data.py`` and catch the
silent-failure risks that moved from generator to artifact when the project
froze datasets to files (EXPERIMENTS.md 2026-07-17).
"""

from __future__ import annotations

import pytest

from geode.arith.validate import cell_counts, order_hash, probe_leakage, uniqueness_by_cell


def test_v5_1_no_leakage_when_disjoint():
    train = [(1, "+", 2), (3, "-", 4), (5, "+", 6)]
    probe = [(7, "+", 8), (9, "-", 10)]
    assert probe_leakage(train, probe) == set()


def test_v5_1_detects_leaked_triple():
    # An exact (a, op, b) match leaks, regardless of which format rendered it
    # (the triple carries no format).
    train = [(1, "+", 2), (3, "-", 4)]
    probe = [(3, "-", 4)]
    assert probe_leakage(train, probe) == {(3, "-", 4)}


def test_v5_1_same_pair_different_op_not_flagged():
    # The operand pair (3, 4) under '+' does not collide with the probe's (3, 4)
    # under '-': leakage is question-level, so a different op is a different item.
    assert probe_leakage([(3, "+", 4)], [(3, "-", 4)]) == set()


def test_v5_1_commuted_triple_not_flagged():
    # (4, +, 3) is a different ordered question from the probe's (3, +, 4);
    # no commutativity collapse under the triple rule.
    assert probe_leakage([(4, "+", 3)], [(3, "+", 4)]) == set()


def test_v5_3_cell_counts_tally_can_be_uneven():
    # Counts need not be uniform under capacity-aware allocation; the validator
    # just tallies per cell.
    cells = [(1, 1)] * 98 + [(4, 4)] * 902
    counts = cell_counts(cells)
    assert counts[(1, 1)] == 98
    assert counts[(4, 4)] == 902
    assert sum(counts.values()) == 1000


def test_v5_2_uniqueness_holds_when_all_distinct():
    # One cell, all-distinct question triples -> n == n_distinct.
    cells = [(2, 2)] * 4
    keys = [(10, "+", 20), (11, "+", 20), (12, "-", 20), (13, "+", 20)]
    report = uniqueness_by_cell(cells, keys)
    assert report[(2, 2)] == (4, 4)


def test_v5_2_detects_planted_duplicate():
    # A repeated question is a bug under the frozen design -> n_distinct < n.
    cells = [(1, 1)] * 4
    keys = [(3, "+", 4), (3, "+", 4), (5, "-", 6), (7, "+", 8)]
    report = uniqueness_by_cell(cells, keys)
    assert report[(1, 1)] == (4, 3)


def _rows():
    return [
        {"a": 3, "b": 5, "op": "+", "shown_answer": 8, "format": "nl", "label_mode": "correct"},
        {"a": 47, "b": 12, "op": "-", "shown_answer": 35, "format": "nl", "label_mode": "correct"},
    ]


def test_v5_40_order_hash_deterministic_and_order_sensitive():
    rows = _rows()
    assert order_hash(rows) == order_hash(_rows())  # pure function of content
    assert order_hash(rows) != order_hash(list(reversed(rows)))  # order counts
    changed = _rows()
    changed[0]["shown_answer"] = 9
    assert order_hash(rows) != order_hash(changed)  # content counts


def test_v5_40_order_hash_numpy_scalars_hash_like_builtins():
    # Parquet round-trips yield numpy scalars; the hash must not change.
    np = pytest.importorskip("numpy")
    rows = _rows()
    coerced = [
        {
            **r,
            "a": np.int64(r["a"]),
            "b": np.int64(r["b"]),
            "shown_answer": np.int64(r["shown_answer"]),
        }
        for r in rows
    ]
    assert order_hash(coerced) == order_hash(rows)
