"""The phase-3 translation bridge builder: both directions, no leaks, no sums.

The bridge (``make_data.py --phase3-bridge``) samples addition pairs and emits
two rewriting rows per pair (operator<->NL) with an answer-free answer slot. Its
silent-failure modes are the promotion-rule kind: a row that leaks the computed
sum, a probe pair that slips into training, a pair missing one direction, or a
cell count that does not match the water-fill. These are checked on tiny
in-process builds — never the frozen 200K file.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter

import pytest

from tests._scriptloader import repo_root

REPO_ROOT = repo_root()
_SCRIPT = REPO_ROOT / "experiments" / "training-run" / "datagen" / "make_data.py"
# A distinct module name (the @dataclass in make_data resolves its module through
# sys.modules at class-creation time), so this load never clobbers another test's.
_spec = importlib.util.spec_from_file_location("make_data_bridge", _SCRIPT)
md = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = md
_spec.loader.exec_module(md)

SEED = 20260727


def test_bridge_both_directions_unique_and_valid():
    records, alloc = md.build_bridge(md.BRIDGE_SPEC, 200, set(), SEED)
    rep = md.validate_bridge(records, set(), alloc)  # raises on any violation
    assert rep["pairs"] == 200
    assert rep["n"] == 400
    assert rep["both_directions"] and rep["all_unique"] and rep["no_minus_char"]
    # Exactly two rows per pair, exactly both directions, all rows distinct.
    per_pair = Counter((r["a"], r["b"]) for r in records)
    assert set(per_pair.values()) == {2}
    assert len({(r["a"], r["b"], r["direction"]) for r in records}) == 400
    assert {r["direction"] for r in records} == {"to_op", "to_nl"}
    assert {r["format"] for r in records} == {"translate_to_op", "translate_to_nl"}


def test_bridge_is_deterministic():
    r1, a1 = md.build_bridge(md.BRIDGE_SPEC, 150, set(), SEED)
    r2, a2 = md.build_bridge(md.BRIDGE_SPEC, 150, set(), SEED)
    assert a1 == a2
    assert md.order_hash(r1) == md.order_hash(r2)


def test_bridge_excludes_probe_pairs():
    probe_pairs = {(1, 1), (2, 3), (7, 40)}
    blocked = {(a, "+", b) for a, b in probe_pairs}
    records, alloc = md.build_bridge(md.BRIDGE_SPEC, 300, blocked, SEED)
    sampled = {(r["a"], r["b"]) for r in records}
    assert sampled.isdisjoint(probe_pairs)
    # The question-level leakage check reports clean.
    assert md.validate_bridge(records, blocked, alloc)["leakage"] == 0


def test_bridge_avoids_target_pairs_where_capacity_allows():
    # A few target pairs in cells with ample free space are avoided entirely.
    target_pairs = {(23, 45), (10, 99), (500, 600)}
    blocked = {(a, "+", b) for a, b in target_pairs}
    records, _ = md.build_bridge(md.BRIDGE_SPEC, 300, blocked, SEED)
    assert {(r["a"], r["b"]) for r in records}.isdisjoint(target_pairs)


def test_bridge_eval_carved_first_is_disjoint_from_train():
    eval_records, eval_alloc = md.build_bridge(
        md.BRIDGE_EVAL_SPEC, 128, set(), SEED, carve_divisor=md.P3_CARVE_DIVISOR
    )
    eval_pairs = {(r["a"], r["b"]) for r in eval_records}
    blocked_train = {(a, "+", b) for a, b in eval_pairs}
    train_records, train_alloc = md.build_bridge(md.BRIDGE_SPEC, 300, blocked_train, SEED)
    train_pairs = {(r["a"], r["b"]) for r in train_records}
    assert eval_pairs.isdisjoint(train_pairs)
    md.validate_bridge(eval_records, set(), eval_alloc)
    md.validate_bridge(train_records, set(), train_alloc)


def test_bridge_pre_exposure_direct_and_commuted_twin():
    bridge_pairs = {(1, 2), (3, 4), (5, 6)}
    other = {(1, 2), (4, 3)}  # (1,2) is a direct hit; (4,3) is the twin of (3,4)
    r = md.bridge_pre_exposure(bridge_pairs, other, twin=True, by_cell=True)
    assert r["bridge_pairs"] == 3
    assert r["direct"] == 1
    assert r["incl_commuted_twin"] == 2
    assert r["frac_direct"] == pytest.approx(1 / 3)
    assert r["frac_incl_twin"] == pytest.approx(2 / 3)
    assert set(r["by_cell"]) == {"1x1"}  # all pairs are single-digit x single-digit
    # Parent-style call: direct only, no twin/by_cell keys.
    rp = md.bridge_pre_exposure(bridge_pairs, other, twin=False, by_cell=False)
    assert rp["direct"] == 1 and "incl_commuted_twin" not in rp and "by_cell" not in rp


def test_validate_bridge_raises_on_missing_direction():
    records, alloc = md.build_bridge(md.BRIDGE_SPEC, 100, set(), SEED)
    victim = records[0]
    dropped = [
        r
        for r in records
        if (r["a"], r["b"], r["direction"]) != (victim["a"], victim["b"], victim["direction"])
    ]
    with pytest.raises(AssertionError, match="both directions"):
        md.validate_bridge(dropped, set(), alloc)


def test_validate_bridge_raises_on_probe_leak():
    records, alloc = md.build_bridge(md.BRIDGE_SPEC, 100, set(), SEED)
    leaked_triple = {(records[0]["a"], "+", records[0]["b"])}
    with pytest.raises(AssertionError, match="probe"):
        md.validate_bridge(records, leaked_triple, alloc)


def test_validate_bridge_raises_on_sum_leak():
    records, alloc = md.build_bridge(md.BRIDGE_SPEC, 40, set(), SEED)
    bad = dict(records[0])
    bad["full_text"] = f"{bad['full_text']} {bad['a'] + bad['b']}"  # inject the sum
    with pytest.raises(AssertionError, match="sum"):
        md.validate_bridge([bad, *records[1:]], set(), alloc)


def test_validate_bridge_raises_on_minus_char():
    records, alloc = md.build_bridge(md.BRIDGE_SPEC, 40, set(), SEED)
    bad = dict(records[0])
    bad["full_text"] = f"{bad['full_text']} -"
    with pytest.raises(AssertionError, match="'-'"):
        md.validate_bridge([bad, *records[1:]], set(), alloc)


def test_validate_bridge_raises_on_bad_answer_span():
    records, alloc = md.build_bridge(md.BRIDGE_SPEC, 40, set(), SEED)
    bad = dict(records[0])
    bad["answer_char_start"] += 1
    with pytest.raises(AssertionError, match="answer span"):
        md.validate_bridge([bad, *records[1:]], set(), alloc)


def test_bridge_report_pieces_have_expected_shape():
    records, alloc = md.build_bridge(md.BRIDGE_SPEC, 120, set(), SEED)
    rep = md.validate_bridge(records, set(), alloc)
    assert {"n", "pairs", "order_hash", "cell_counts", "no_minus_char"} <= rep.keys()
    assert sum(rep["cell_counts"].values()) == rep["n"]
    vs = md.bridge_pre_exposure(
        {(r["a"], r["b"]) for r in records}, {(1, 2)}, twin=True, by_cell=True
    )
    assert {"direct", "incl_commuted_twin", "by_cell", "frac_direct"} <= vs.keys()
