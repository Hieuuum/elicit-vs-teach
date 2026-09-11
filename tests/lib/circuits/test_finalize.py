"""Final reporting preserves native failures and refuses unaudited artifacts."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def finalizer():
    path = Path(__file__).resolve().parents[3] / "experiments/olmo2-circuit-overlap/finalize.py"
    spec = importlib.util.spec_from_file_location("finalize", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_audit_blocks_report_and_saves_failure(finalizer, tmp_path, monkeypatch):
    monkeypatch.setattr(
        finalizer,
        "audit_run",
        lambda root: {"status": "failed", "failures": ["mismatched examples"], "warnings": []},
    )
    monkeypatch.setattr(finalizer, "generate_report", lambda *a, **k: pytest.fail("report ran"))
    with pytest.raises(RuntimeError, match="mismatched examples"):
        finalizer.finalize(tmp_path)
    status = json.loads((tmp_path.parent / "finalization_status.json").read_text())
    assert status["status"] == "failed"


def test_finalization_preserves_failure_answers_and_original_scores(
    finalizer, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        finalizer,
        "audit_run",
        lambda root: {
            "status": "passed",
            "failures": [],
            "warnings": [],
            "expected_stages": ["sft"],
        },
    )
    calls = []
    monkeypatch.setattr(finalizer, "generate_report", lambda *a, **k: calls.append((a, k)))
    stage = tmp_path / "sft"
    stage.mkdir()
    rows = [
        {
            "id": "a",
            "task": "cruxeval_output",
            "prompt": "question",
            "generation": "prose",
            "answer": "5",
            "correct": False,
            "parse_failure": True,
        },
        {"id": "b", "task": "cruxeval_output", "generation": "5", "correct": True},
    ]
    source = stage / "behavior.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    original = source.read_bytes()
    finalizer.finalize(tmp_path)
    assert source.read_bytes() == original
    failures = [
        json.loads(s) for s in (tmp_path / "format_failure_examples.jsonl").read_text().splitlines()
    ]
    assert len(failures) == 1 and failures[0]["generation"] == "prose"
    summary = json.loads((tmp_path / "format_failure_summary.json").read_text())["rows"][0]
    assert summary["n_rows"] == 2 and summary["correct"] == summary["parse_failure"] == 1
    assert calls[0][1] == {"n_bootstrap": 2000, "seed": 0}
    assert (
        json.loads((tmp_path.parent / "finalization_status.json").read_text())["status"]
        == "complete"
    )
