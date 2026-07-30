"""Phase-3 embedding warm-start data: correct NL addition, no direct/twin leak."""

from __future__ import annotations

import importlib.util
import sys

from tests._scriptloader import repo_root

REPO_ROOT = repo_root()
_SCRIPT = REPO_ROOT / "experiments" / "training-run" / "datagen" / "make_data.py"
_spec = importlib.util.spec_from_file_location("make_data_p3_warmstart", _SCRIPT)
md = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = md
_spec.loader.exec_module(md)


def test_v5_68_p3_warmstart_is_deterministic_correct_and_twin_disjoint() -> None:
    frozen = {(1, "+", 2), (10, "+", 99), (123, "+", 45)}
    blocked = md._with_commuted_twins(frozen)
    records, plan = md.build_dataset(
        md.P3_WARMSTART_SPEC,
        512,
        blocked,
        md.P3_WARMSTART_SEED,
        cells=md.P3_CELLS,
    )
    again, again_plan = md.build_dataset(
        md.P3_WARMSTART_SPEC,
        512,
        blocked,
        md.P3_WARMSTART_SEED,
        cells=md.P3_CELLS,
    )

    triples = {(r["a"], r["op"], r["b"]) for r in records}
    assert plan == again_plan
    assert md.order_hash(records) == md.order_hash(again)
    assert triples.isdisjoint(blocked)
    assert all((b, op, a) not in frozen for a, op, b in triples)
    assert {r["op"] for r in records} == {"+"}
    assert {r["format"] for r in records} == {"nl"}
    assert {r["label_mode"] for r in records} == {"correct"}
    assert all(r["shown_answer"] == r["true_answer"] for r in records)
    assert md.validate(records, blocked, plan, cells=md.P3_CELLS)["leakage"] == 0
