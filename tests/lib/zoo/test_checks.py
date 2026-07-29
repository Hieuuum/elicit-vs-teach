"""``require_parent_ready`` — child-launch refusal on a bad parent (V0.6).

EXPERIMENTS.md §3.1: a child run refuses to start if its parent is missing,
incomplete, or has failing gates. Every refusal is a raise with the offender
named; the only silent path is a parent that is genuinely ready.
"""

from __future__ import annotations

import pytest

from geode.zoo import ConsistencyError, register_run, require_parent_ready
from tests.lib.zoo.test_manifest import make_manifest


def _register_parent(store, *, status="complete", gates=None, run_id="parent-1"):
    fields = make_manifest(run_id)
    fields["status"] = status
    fields["experiment"] = {"gates": gates if gates is not None else {}}
    return register_run(fields, store=store)


def test_v0_6_missing_parent_raises(tmp_path):
    with pytest.raises(ConsistencyError, match="no manifest"):
        require_parent_ready("never-registered", store=tmp_path)


def test_v0_6_incomplete_parent_raises(tmp_path):
    _register_parent(tmp_path, status="running")
    with pytest.raises(ConsistencyError, match="status 'running'"):
        require_parent_ready("parent-1", store=tmp_path)


def test_v0_6_failing_gate_raises(tmp_path):
    _register_parent(tmp_path, gates={"G0": {"pass": False, "coherent": 5}})
    with pytest.raises(ConsistencyError, match="gate 'G0'"):
        require_parent_ready("parent-1", store=tmp_path)


def test_v0_6_gate_without_verdict_raises(tmp_path):
    # A recorded gate with no pass field is malformed, not a pass.
    _register_parent(tmp_path, gates={"G0": {"accuracy": 0.99}})
    with pytest.raises(ConsistencyError, match="gate 'G0'"):
        require_parent_ready("parent-1", store=tmp_path)


def test_v0_6_required_gate_unrecorded_raises(tmp_path):
    _register_parent(tmp_path, gates={})
    with pytest.raises(ConsistencyError, match="required"):
        require_parent_ready("parent-1", required_gates=("G0",), store=tmp_path)


def test_v0_6_ready_parent_passes(tmp_path):
    _register_parent(tmp_path, gates={"G0": {"pass": True, "coherent": 18}})
    require_parent_ready("parent-1", required_gates=("G0",), store=tmp_path)  # no raise
