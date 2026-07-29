"""Phase-3 teach installer: permuted NL addition with no target/eval/probe leak."""

from __future__ import annotations

import importlib.util
import sys

from tests._scriptloader import repo_root

REPO_ROOT = repo_root()
_SCRIPT = REPO_ROOT / "experiments" / "training-run" / "datagen" / "make_data.py"
_spec = importlib.util.spec_from_file_location("make_data_p3_teach", _SCRIPT)
md = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = md
_spec.loader.exec_module(md)

SEED = 20260727


def test_p3_teach_installer_is_deterministic_and_role_matched() -> None:
    blocked = {(1, "+", 1), (2, "+", 3), (10, "+", 99)}
    records, plan = md.build_dataset(
        md.P3_TEACH_INST_SPEC,
        512,
        blocked,
        SEED,
        cells=md.P3_CELLS,
    )
    again, again_plan = md.build_dataset(
        md.P3_TEACH_INST_SPEC,
        512,
        blocked,
        SEED,
        cells=md.P3_CELLS,
    )

    assert plan == again_plan
    assert md.order_hash(records) == md.order_hash(again)
    assert {(r["a"], r["op"], r["b"]) for r in records}.isdisjoint(blocked)
    assert {r["op"] for r in records} == {"+"}
    assert {r["format"] for r in records} == {"nl"}
    assert {r["label_mode"] for r in records} == {"permuted"}
    assert all("-" not in r["full_text"] for r in records)
    assert sorted(r["shown_answer"] for r in records) == sorted(r["true_answer"] for r in records)
    assert md.validate(records, blocked, plan, cells=md.P3_CELLS)["leakage"] == 0
