"""C9: reproducible artifacts, immutable revisions, and overwrite/leak guards."""

import json
from pathlib import Path

import pytest

from geode.circuits.artifacts import (
    artifact_upload_files,
    canonical_json,
    fingerprint,
    read_jsonl,
    start_run,
    validate_matched_rows,
    write_json,
    write_jsonl,
)
from geode.circuits.checkpoints import CHECKPOINTS, Checkpoint, select_checkpoints


def test_all_endpoints_are_immutable_and_ordered():
    assert [c.stage for c in CHECKPOINTS] == [
        "init",
        "stage1",
        "stage2",
        "sft",
        "dpo",
        "rlvr1",
        "rlvr2",
    ]
    assert all(len(c.revision) == 40 for c in CHECKPOINTS)
    with pytest.raises(ValueError, match="immutable"):
        Checkpoint("stage2", "a/b", "main", "main")
    with pytest.raises(ValueError, match="unknown"):
        select_checkpoints(["latest"])
    with pytest.raises(ValueError, match="duplicate"):
        select_checkpoints(["init", "init"])


def test_hash_is_order_invariant_and_content_sensitive():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_does_not_replace_valid_artifact(tmp_path, value):
    path = tmp_path / "result.json"
    write_json(path, {"score": 1})
    with pytest.raises(ValueError):
        write_json(path, {"score": value})
    assert json.loads(path.read_text()) == {"score": 1}
    assert not list(tmp_path.glob(".tmp-*"))


def test_jsonl_roundtrip_and_atomic_validation(tmp_path):
    p = tmp_path / "rows.jsonl"
    rows = [{"id": "é", "score": 0.1}, {"id": "2", "score": None}]
    write_jsonl(p, rows)
    assert read_jsonl(p) == rows
    with pytest.raises(ValueError):
        write_jsonl(p, [{"bad": float("nan")}])
    assert read_jsonl(p) == rows


def test_existing_run_cannot_be_silently_replaced(tmp_path):
    dest = tmp_path / "run"
    meta = start_run(dest, {"seed": 0}, {"source": "abc"})
    assert meta["status"] == "running"
    with pytest.raises(FileExistsError):
        start_run(dest, {"seed": 1}, {})
    assert json.loads((dest / "run_metadata.json").read_text())["config"]["seed"] == 0


@pytest.mark.parametrize(
    "field", ["item_ids", "group_ids", "node_names", "protocol_hash", "tokenizer_hash"]
)
def test_all_matched_comparison_keys_are_enforced(field):
    a = {
        "item_ids": ["1", "2"],
        "group_ids": ["g1", "g2"],
        "node_names": ["n1"],
        "protocol_hash": "p",
        "tokenizer_hash": "t",
    }
    validate_matched_rows(a, a)
    b = {**a, field: "different"}
    with pytest.raises(ValueError, match=field):
        validate_matched_rows(a, b)
    del b[field]
    with pytest.raises(ValueError, match=field):
        validate_matched_rows(a, b)


def test_upload_allowlist_excludes_model_cache_secrets(tmp_path):
    for name in [
        "report.json",
        "scores.npz",
        "plot.png",
        "model/model.safetensors",
        ".env",
        "cache/token.json",
        "credentials/key.json",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")
    assert {p.name for p in artifact_upload_files(tmp_path)} == {
        "report.json",
        "scores.npz",
        "plot.png",
    }
    (tmp_path / "linked.json").symlink_to(tmp_path / "report.json")
    with pytest.raises(ValueError, match="symlink"):
        artifact_upload_files(tmp_path)


def test_json_rejects_unknown_objects():
    with pytest.raises(TypeError):
        canonical_json({"path": Path("/tmp")})
