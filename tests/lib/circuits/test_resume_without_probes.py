"""Resumption must not silently reuse changed inputs or incomplete circuits."""

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from geode.circuits.data import Example
from tests.lib.circuits.test_sanity import make_run


@pytest.fixture
def resume_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "experiments/olmo2-circuit-overlap/resume_without_probes.py"
    )
    spec = importlib.util.spec_from_file_location("resume_without_probes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reuse_preserves_whole_tasks_and_rejects_changed_inputs(resume_module):
    args = argparse.Namespace(max_new_tokens=384, math_max_new_tokens=384)
    examples = [Example(str(i), "gsm_symbolic", "Question", "3", "g") for i in range(2)]
    rows = [{**asdict(ex), "max_new_tokens": 384, "generation": "3"} for ex in examples]
    assert resume_module.reusable_tasks(rows, examples, None, args) == {"gsm_symbolic"}
    with pytest.raises(ValueError, match="Partial"):
        resume_module.reusable_tasks(rows[:1], examples, None, args)
    with pytest.raises(ValueError, match="input changed"):
        resume_module.reusable_tasks(
            rows, [replace(examples[0], prompt="Changed"), examples[1]], None, args
        )
    rows[0]["max_new_tokens"] = 192
    with pytest.raises(ValueError, match="cap changed"):
        resume_module.reusable_tasks(rows, examples, None, args)


def test_saved_circuits_require_completion_and_finite_matching_arrays(resume_module, tmp_path):
    root = tmp_path / "results"
    make_run(root)
    (root / "a").rename(root / "init")
    log = root.parent / "full.log"
    log.write_text("")
    assert resume_module.saved_circuit_times(root / "b", root, ["toy"]) is None
    log.write_text(
        json.dumps({"event": "circuits_done", "stage": "b", "task": "toy", "seconds": 1}) + "\n"
    )
    assert resume_module.saved_circuit_times(root / "b", root, ["toy"])[0]["seconds"] == 1
    np.savez(root / "b/circuits/toy.npz", scores=np.full((1, 4), np.nan))
    with pytest.raises(ValueError, match="Invalid saved circuit"):
        resume_module.saved_circuit_times(root / "b", root, ["toy"])
