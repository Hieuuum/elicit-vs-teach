"""make_algo_fresh4m.py: the ts1b redesign's fresh fine-tune set.

Silent-failure risk: if the exclusion wiring were wrong (a block set built
with the wrong key order, the EAP-IG pairs added in one operand order only,
the small-cell release dropping the wrong cells), the children would quietly
train on the elicit parent's symbol problems or on evaluation problems, and
the elicit-vs-teach comparison would be confounded with nothing crashing.
These tests check the pure helpers on tiny in-process fixtures; the real 4M
build re-checks disjointness on the final file before writing it.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter

import pandas as pd
import pytest

from tests._scriptloader import repo_root

REPO_ROOT = repo_root()
_SCRIPT = REPO_ROOT / "experiments" / "training-run" / "datagen" / "make_algo_fresh4m.py"
_spec = importlib.util.spec_from_file_location("make_algo_fresh4m", _SCRIPT)
maf = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = maf
_spec.loader.exec_module(maf)


def _df(rows):
    out = []
    for i, (a, op, b) in enumerate(rows):
        ans = a + b if op == "+" else a - b
        out.append(
            {
                "idx": i,
                "dataset": "src",
                "a": a,
                "b": b,
                "op": op,
                "x_digits": len(str(a)),
                "y_digits": len(str(b)),
                "format": "bare_nl",
                "label_mode": "correct",
                "true_answer": ans,
                "shown_answer": ans,
            }
        )
    return pd.DataFrame(out)


# --- triples_of ---------------------------------------------------------------


def test_triples_of_order_is_a_op_b():
    df = _df([(12, "-", 345), (7, "+", 8)])
    assert maf.triples_of(df) == [(12, "-", 345), (7, "+", 8)]


def test_triples_are_ints_not_numpy():
    t = maf.triples_of(_df([(1, "+", 2)]))[0]
    assert type(t[0]) is int and type(t[2]) is int


# --- release_small_cells -------------------------------------------------------


@pytest.mark.parametrize(
    "t", [(1, "+", 2), (3, "-", 45), (6, "+", 789), (12, "-", 3), (12, "+", 34), (123, "-", 4)]
)
def test_small_cells_are_released(t):
    assert maf.release_small_cells({t}) == set()


@pytest.mark.parametrize(
    "t",
    [
        (1, "+", 2345),
        (12, "-", 345),
        (123, "+", 45),
        (1234, "-", 5),
        (123, "+", 456),
        (9999, "+", 9999),
    ],
)
def test_other_cells_stay_blocked(t):
    assert maf.release_small_cells({t}) == {t}


def test_small_cell_set_is_exactly_six_cells():
    assert maf.SMALL_CELLS == frozenset({(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1)})


def test_release_is_by_operand_length_not_op():
    assert maf.release_small_cells({(5, "+", 6), (5, "-", 6)}) == set()


def test_two_by_three_is_not_small():
    assert maf.release_small_cells({(10, "+", 100)}) == {(10, "+", 100)}


# --- addition_pair_triples -----------------------------------------------------


def test_eapig_pairs_blocked_in_both_orders():
    m = pd.DataFrame({"a": [1234], "b": [56]})
    assert maf.addition_pair_triples([m]) == {(1234, "+", 56), (56, "+", 1234)}


def test_eapig_counterfactuals_blocked_too():
    m = pd.DataFrame({"a": [10], "b": [20], "cf_a": [30], "cf_b": [40]})
    got = maf.addition_pair_triples([m])
    assert {(30, "+", 40), (40, "+", 30), (10, "+", 20), (20, "+", 10)} == got


def test_eapig_pairs_are_addition_only():
    m = pd.DataFrame({"a": [5], "b": [7]})
    assert all(op == "+" for _, op, _ in maf.addition_pair_triples([m]))


def test_eapig_tables_without_operands_ignored():
    m = pd.DataFrame({"n": [1, 2], "cf_n": [3, 4]})  # copy-task meta
    assert maf.addition_pair_triples([m]) == set()


def test_eapig_multiple_tables_union():
    m1 = pd.DataFrame({"a": [1], "b": [2]})
    m2 = pd.DataFrame({"a": [3], "b": [4]})
    assert len(maf.addition_pair_triples([m1, m2])) == 4


def test_eapig_symmetric_pair_counted_once():
    m = pd.DataFrame({"a": [7], "b": [7]})
    assert maf.addition_pair_triples([m]) == {(7, "+", 7)}


# --- keep_rows -----------------------------------------------------------------


def test_keep_rows_drops_exact_blocked_triples_only():
    df = _df([(1, "+", 2), (2, "+", 1), (1, "-", 2)])
    kept = maf.keep_rows(df, {(1, "+", 2)})
    assert maf.triples_of(kept) == [(2, "+", 1), (1, "-", 2)]


def test_keep_rows_preserves_order():
    rows = [(i, "+", i + 1) for i in range(20)]
    kept = maf.keep_rows(_df(rows), {(5, "+", 6), (11, "+", 12)})
    assert maf.triples_of(kept) == [r for r in rows if r not in {(5, "+", 6), (11, "+", 12)}]


def test_keep_rows_empty_block_keeps_all():
    df = _df([(1, "+", 2), (3, "-", 4)])
    assert len(maf.keep_rows(df, set())) == 2


def test_keep_rows_op_matters():
    df = _df([(4, "-", 3)])
    assert len(maf.keep_rows(df, {(4, "+", 3)})) == 1


# --- assemble ------------------------------------------------------------------


def test_assemble_contains_every_row_once():
    kept = _df([(i, "+", 1000 + i) for i in range(30)])
    new = _df([(i, "-", 2000 + i) for i in range(20)])
    full = maf.assemble(kept, new, seed=1, name="X")
    assert sorted(maf.triples_of(full)) == sorted(maf.triples_of(kept) + maf.triples_of(new))


def test_assemble_reindexes_and_renames():
    full = maf.assemble(_df([(1, "+", 2)]), _df([(3, "+", 4), (5, "-", 6)]), seed=0, name="Y")
    assert full["idx"].tolist() == [0, 1, 2]
    assert set(full["dataset"]) == {"Y"}


def test_assemble_shuffles_so_new_rows_are_not_all_at_the_end():
    kept = _df([(i, "+", 1000 + i) for i in range(200)])
    new = _df([(i, "-", 2000 + i) for i in range(200)])
    full = maf.assemble(kept, new, seed=3, name="Z")
    first_half_ops = Counter(full["op"].iloc[:200])
    assert 50 < first_half_ops["-"] < 150


def test_assemble_deterministic_and_seed_dependent():
    kept = _df([(i, "+", 1000 + i) for i in range(50)])
    new = _df([(i, "-", 2000 + i) for i in range(50)])
    a = maf.assemble(kept, new, seed=7, name="Z")
    b = maf.assemble(kept, new, seed=7, name="Z")
    c = maf.assemble(kept, new, seed=8, name="Z")
    assert a.equals(b)
    assert not a.equals(c)


def test_assemble_keeps_labels_attached_to_their_problem():
    kept = _df([(i, "+", 1000 + i) for i in range(40)])
    new = _df([(i, "-", 2000 + i) for i in range(40)])
    full = maf.assemble(kept, new, seed=2, name="Z")
    expected = [a + b if op == "+" else a - b for a, op, b in maf.triples_of(full)]
    assert full["shown_answer"].tolist() == expected
