"""make_test_trainmix.py: the ts1b redesign's test set reshaped to the training mix.

Silent-failure risk: a rounding bug or a stratum keyed the wrong way would
give a test set whose problem-size mix differs from training (the very thing
it exists to prevent), or an unseeded sample would make the reported accuracy
irreproducible, with nothing crashing. These tests check the pure helpers on
tiny in-process fixtures; the real build re-checks counts and disjointness.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter

import pandas as pd
import pytest

from tests._scriptloader import repo_root

_SCRIPT = repo_root() / "experiments" / "training-run" / "datagen" / "make_test_trainmix.py"
_spec = importlib.util.spec_from_file_location("make_test_trainmix", _SCRIPT)
mtt = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mtt
_spec.loader.exec_module(mtt)


def _df(n_per: dict) -> pd.DataFrame:
    """n_per maps (x_digits, y_digits, op) -> row count; operands are unique."""
    rows, i = [], 0
    for (x, y, op), n in sorted(n_per.items()):
        for j in range(n):
            a = 10 ** (x - 1) + j
            b = 10 ** (y - 1) + (j % 7)
            rows.append({"idx": i, "a": a, "b": b, "op": op, "x_digits": x, "y_digits": y})
            i += 1
    return pd.DataFrame(rows)


# --- allocate --------------------------------------------------------------------


def test_allocate_sums_to_n():
    w = {"a": 1, "b": 1, "c": 1}
    assert sum(mtt.allocate(w, 100).values()) == 100


def test_allocate_exact_proportions_when_divisible():
    assert mtt.allocate({"a": 3, "b": 1}, 400) == {"a": 300, "b": 100}


def test_allocate_largest_remainder():
    # exact 33.33 / 33.33 / 33.33 of 100 -> one key gets the extra 1
    got = mtt.allocate({"a": 1, "b": 1, "c": 1}, 100)
    assert sorted(got.values()) == [33, 33, 34]


def test_allocate_zero_weight_gets_zero():
    assert mtt.allocate({"a": 0, "b": 5}, 10) == {"a": 0, "b": 10}


def test_allocate_each_within_one_of_exact():
    w = {k: v for k, v in zip("abcdefghij", [57, 57, 628, 57, 628, 628, 57, 628, 628, 628])}
    got = mtt.allocate(w, 20000)
    tot = sum(w.values())
    assert all(abs(got[k] - 20000 * w[k] / tot) < 1 for k in w)


def test_allocate_deterministic():
    w = {("x", 1): 7, ("y", 2): 11, ("z", 3): 13}
    assert mtt.allocate(w, 99) == mtt.allocate(w, 99)


def test_allocate_rejects_empty_weight():
    with pytest.raises(ValueError):
        mtt.allocate({"a": 0}, 5)


def test_allocate_tuple_keys():
    w = {(1, 4, "+"): 1, (2, 4, "-"): 3}
    assert mtt.allocate(w, 8) == {(1, 4, "+"): 2, (2, 4, "-"): 6}


# --- strata_counts -----------------------------------------------------------------


def test_strata_counts_keys_are_cell_and_op():
    df = _df({(1, 4, "+"): 2, (1, 4, "-"): 3, (2, 3, "+"): 1})
    assert mtt.strata_counts(df) == Counter({(1, 4, "+"): 2, (1, 4, "-"): 3, (2, 3, "+"): 1})


def test_strata_counts_ints_not_numpy():
    k = next(iter(mtt.strata_counts(_df({(3, 3, "+"): 1}))))
    assert type(k[0]) is int and type(k[1]) is int


# --- sample_strata -----------------------------------------------------------------


def test_sample_strata_exact_counts():
    df = _df({(1, 4, "+"): 50, (4, 4, "-"): 80})
    alloc = {(1, 4, "+"): 5, (4, 4, "-"): 20}
    out = mtt.sample_strata(df, alloc, seed=1)
    assert mtt.strata_counts(out) == Counter(alloc)


def test_sample_strata_rows_unchanged_and_unique():
    df = _df({(2, 4, "+"): 40, (3, 2, "-"): 40})
    out = mtt.sample_strata(df, {(2, 4, "+"): 10, (3, 2, "-"): 7}, seed=2)
    assert out["idx"].is_unique
    merged = out.merge(df, on="idx", suffixes=("", "_src"))
    assert (merged["a"] == merged["a_src"]).all() and (merged["op"] == merged["op_src"]).all()


def test_sample_strata_deterministic_and_seed_dependent():
    df = _df({(4, 3, "+"): 200})
    a = mtt.sample_strata(df, {(4, 3, "+"): 30}, seed=5)
    b = mtt.sample_strata(df, {(4, 3, "+"): 30}, seed=5)
    c = mtt.sample_strata(df, {(4, 3, "+"): 30}, seed=6)
    assert a["idx"].tolist() == b["idx"].tolist()
    assert a["idx"].tolist() != c["idx"].tolist()


def test_sample_strata_short_stratum_raises():
    df = _df({(1, 4, "+"): 3})
    with pytest.raises(ValueError):
        mtt.sample_strata(df, {(1, 4, "+"): 4}, seed=0)


def test_sample_strata_missing_stratum_raises():
    df = _df({(1, 4, "+"): 3})
    with pytest.raises(ValueError):
        mtt.sample_strata(df, {(2, 3, "+"): 1}, seed=0)


def test_sample_strata_zero_allocation_skipped():
    df = _df({(1, 4, "+"): 3, (2, 2, "+"): 3})
    out = mtt.sample_strata(df, {(1, 4, "+"): 2, (2, 2, "+"): 0}, seed=0)
    assert mtt.strata_counts(out) == Counter({(1, 4, "+"): 2})


def test_sample_strata_shuffles_across_strata():
    df = _df({(1, 4, "+"): 100, (4, 4, "-"): 100})
    out = mtt.sample_strata(df, {(1, 4, "+"): 50, (4, 4, "-"): 50}, seed=3)
    first_half = Counter(out["op"].iloc[:50])
    assert 10 < first_half["+"] < 40


def test_mix_follows_training_end_to_end():
    train = _df({(1, 4, "+"): 10, (1, 4, "-"): 10, (4, 4, "+"): 90, (4, 4, "-"): 90})
    test = _df({(1, 4, "+"): 100, (1, 4, "-"): 100, (4, 4, "+"): 100, (4, 4, "-"): 100})
    alloc = mtt.allocate(dict(mtt.strata_counts(train)), 100)
    out = mtt.sample_strata(test, alloc, seed=0)
    got = mtt.strata_counts(out)
    assert got[(1, 4, "+")] + got[(1, 4, "-")] == 10
    assert got[(4, 4, "+")] + got[(4, 4, "-")] == 90
