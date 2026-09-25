"""Saved-artifact audits fail on silent corruption, leakage and changed inputs."""

import copy
import json

import numpy as np
import pytest

from geode.circuits.artifacts import write_json, write_jsonl
from geode.circuits.sanity import audit_run, validate_interventions, validate_probe_splits


def fixture_data():
    names = ["layer.0.mlp", "layer.1.mlp", "layer.0.attn.0", "layer.1.attn.0"]
    circuit = {
        "node_names": names,
        "top16": [names[0], names[2]],
        "clean_group_ids": ["a"],
        "corrupt_group_ids": ["b"],
        "item_ids": ["x|y"],
        "group_ids": ["a|b"],
        "protocol_hash": "p",
        "tokenizer_hash": "t",
    }
    patch = {"clean_metric": -1.0, "patched_metric": -2.0, "effect": -1.0}
    row = {
        "clean_group": "c",
        "corrupt_group": "d",
        "top": copy.deepcopy(patch),
        "random": copy.deepcopy(patch),
        "random_nodes": names[:2],
        "random_type_matched": copy.deepcopy(patch),
        "random_type_matched_nodes": [names[1], names[3]],
    }
    return circuit, row


def test_type_matched_audit_keeps_uniform_comparator_distinct():
    circuit, row = fixture_data()
    validate_interventions([row], circuit, require_typed=True)
    del row["random_type_matched"]
    del row["random_type_matched_nodes"]
    validate_interventions([row], circuit, require_typed=False)
    with pytest.raises(KeyError):
        validate_interventions([row], circuit, require_typed=True)


@pytest.mark.parametrize("fault", ["overlap", "types", "duplicate", "nonfinite", "sign", "clean"])
def test_intervention_audit_rejects_silent_faults(fault):
    circuit, row = fixture_data()
    if fault == "overlap":
        row["clean_group"] = "a"
    elif fault == "types":
        row["random_type_matched_nodes"] = circuit["node_names"][:2]
    elif fault == "duplicate":
        row["random_nodes"] = [circuit["node_names"][0]] * 2
    elif fault == "nonfinite":
        row["random"]["effect"] = float("nan")
    elif fault == "sign":
        row["top"]["effect"] = 1.0
    else:
        row["random"]["clean_metric"] = 0.0
        row["random"]["effect"] = -2.0
    with pytest.raises(ValueError):
        validate_interventions([row], circuit, require_typed=True)


@pytest.mark.parametrize("fault", ["group_leak", "row_overlap", "test_labels"])
def test_probe_audit_checks_source_partition_and_saved_labels(fault):
    labels = np.array([0, 1, 0])
    groups = np.array(["a", "b", "c"])
    probe = {
        "layers": [
            {"split_indices": {"train": [0], "validation": [1], "test": [2]}, "test_labels": [0]}
        ]
    }
    validate_probe_splits(probe, labels, groups)
    if fault == "group_leak":
        groups[2] = "a"
    elif fault == "row_overlap":
        probe["layers"][0]["split_indices"]["train"].append(2)
    else:
        probe["layers"][0]["test_labels"] = [1]
    with pytest.raises(ValueError):
        validate_probe_splits(probe, labels, groups)


def make_run(root):
    checkpoints = [{"stage": stage, "repo": "fixture", "revision": stage} for stage in ["a", "b"]]
    write_json(
        root / "run_metadata.json",
        {"status": "complete", "config": {"tasks": ["toy"], "checkpoints": checkpoints}},
    )
    circuit, row = fixture_data()
    for checkpoint in checkpoints:
        base = root / checkpoint["stage"]
        write_json(
            base / "checkpoint.json", {**checkpoint, "tokenizer_hash": "t", "parameter_count": 10}
        )
        write_json(base / "hardware.json", {"device": "cpu"})
        write_jsonl(
            base / "behavior.jsonl",
            [
                {
                    "id": "x",
                    "group": "g",
                    "prompt": "p",
                    "answer": "a",
                    "task": "toy",
                    "answer_log_prob_nats": -1,
                    "truncated": False,
                    "context_overflow": False,
                }
            ],
        )
        write_json(base / "circuits/toy.json", circuit)
        np.savez_compressed(base / "circuits/toy.npz", scores=np.ones((1, 4)))
        write_json(base / "circuits/toy_interventions.json", [row])
        probe = {
            "n_rows": 2,
            "item_ids": ["p0", "p1"],
            "status": "complete",
            "layers": [
                {"split_indices": {"train": [0], "validation": [], "test": [1]}, "test_labels": [1]}
            ],
        }
        write_json(base / "probes/toy.json", probe)
        np.savez_compressed(
            base / "probes/toy.npz",
            features=np.zeros((2, 2, 3)),
            labels=np.array([0, 1]),
            groups=np.array(["p0", "p1"]),
        )


def test_complete_saved_run_passes_without_capability_threshold(tmp_path):
    make_run(tmp_path)
    result = audit_run(tmp_path, expected_stages=["a", "b"])
    assert result["status"] == "passed"
    assert set(result["checkpoints"]) == {"a", "b"}


def test_skipped_probes_are_explicit_and_do_not_relax_circuit_audit(tmp_path):
    make_run(tmp_path)
    (tmp_path / "b/probes/toy.json").unlink()
    (tmp_path / "b/probes/toy.npz").unlink()
    assert audit_run(tmp_path, expected_stages=["a", "b"])["status"] == "failed"
    path = tmp_path / "run_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["probe_policy_by_stage"] = {"a": "complete", "b": "skipped"}
    write_json(path, metadata)
    result = audit_run(tmp_path, expected_stages=["a", "b"])
    assert result["status"] == "passed" and not result["warnings"]
    assert result["checkpoints"]["b"]["tasks"]["toy"]["probe_status"] == "skipped by user request"
    (tmp_path / "b/circuits/toy_interventions.json").unlink()
    assert audit_run(tmp_path, expected_stages=["a", "b"])["status"] == "failed"


@pytest.mark.parametrize("fault", ["stage", "revision", "behavior", "nan", "protocol"])
def test_end_to_end_audit_detects_artifact_mismatch(tmp_path, fault):
    make_run(tmp_path)
    base = tmp_path / "b"
    if fault == "stage":
        (base / "checkpoint.json").unlink()
    elif fault == "nan":
        np.savez_compressed(base / "circuits/toy.npz", scores=np.full((1, 4), np.nan))
    else:
        path = (
            base
            / {
                "revision": "checkpoint.json",
                "behavior": "behavior.jsonl",
                "protocol": "circuits/toy.json",
            }[fault]
        )
        value = json.loads(path.read_text())
        value[
            {"revision": "revision", "behavior": "prompt", "protocol": "protocol_hash"}[fault]
        ] = "changed"
        path.write_text(json.dumps(value) + "\n")
    result = audit_run(tmp_path, expected_stages=["a", "b"])
    assert result["status"] == "failed"
    assert result["failures"]
