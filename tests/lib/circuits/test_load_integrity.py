"""Actual tiny cached-model integrity checks, with network calls prohibited."""

import importlib.util
import json
from pathlib import Path
import socket

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Olmo2Config, Olmo2ForCausalLM

from geode.circuits.artifacts import sha256_file


@pytest.fixture
def audit_module():
    script = (
        Path(__file__).resolve().parents[3] / "experiments/olmo2-circuit-overlap/audit_loads.py"
    )
    spec = importlib.util.spec_from_file_location("circuit_load_audit_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cached_checkpoint(tmp_path, monkeypatch):
    def no_network(*_args, **_kwargs):
        raise AssertionError("The cached integrity audit must never access the network")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    cache = tmp_path / "hub"
    snapshot = cache / "models--owner--tiny" / "snapshots" / ("a" * 40)
    with torch.random.fork_rng():
        torch.manual_seed(5)
        config = Olmo2Config(
            vocab_size=19,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            eos_token_id=1,
            pad_token_id=0,
        )
        Olmo2ForCausalLM(config).save_pretrained(snapshot)
    weight = snapshot / "model.safetensors"
    record = {
        "stage": "fixture",
        "repo": "owner/tiny",
        "revision": "a" * 40,
        "assets": [{"filename": "config.json", "sha256": sha256_file(snapshot / "config.json")}],
        "weight_files": [
            {
                "filename": weight.name,
                "advertised_sha256": sha256_file(weight),
                "bytes": weight.stat().st_size,
            }
        ],
    }
    return cache, snapshot, record


def test_cached_bf16_cpu_reload_checks_all_loader_fields_and_file_bytes(
    audit_module, cached_checkpoint
):
    cache, snapshot, record = cached_checkpoint
    result = audit_module.audit_checkpoint(record, cache_dir=cache)
    assert result["status"] == "passed"
    assert result["execution_dtype"] == "bfloat16"
    assert result["execution_device"] == "cpu"
    assert result["loading_info"] == {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    assert result["files"][0]["sha256"] == sha256_file(snapshot / "model.safetensors")
    assert result["loaded_weight_files"] == ["model.safetensors"]


@pytest.mark.parametrize("kind", ["missing", "unexpected", "mismatched"])
def test_loader_detects_invalid_state_dict_even_when_file_hash_matches(
    audit_module, cached_checkpoint, kind
):
    cache, snapshot, record = cached_checkpoint
    path = snapshot / "model.safetensors"
    weights = load_file(path)
    key = "model.layers.0.mlp.down_proj.weight"
    if kind == "missing":
        weights.pop(key)
    elif kind == "unexpected":
        weights["unexpected_fixture.weight"] = torch.zeros(2, 2)
    else:
        weights[key] = weights[key][:1]
    save_file(weights, path, metadata={"format": "pt"})
    record["weight_files"][0].update(advertised_sha256=sha256_file(path), bytes=path.stat().st_size)
    result = audit_module.audit_checkpoint(record, cache_dir=cache)
    assert result["status"] == "failed"
    if kind == "mismatched":
        assert "RuntimeError" in result["error"] and "mismatched" in result["error"]
    else:
        assert result["loading_info"][f"{kind}_keys"]


def test_bad_advertised_hash_fails_before_loading(audit_module, cached_checkpoint, monkeypatch):
    from transformers import AutoModelForCausalLM

    cache, _, record = cached_checkpoint
    record["weight_files"][0]["advertised_sha256"] = "0" * 64
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *_a, **_kw: pytest.fail("Unverified weights must not be loaded"),
    )
    result = audit_module.audit_checkpoint(record, cache_dir=cache)
    assert result["status"] == "failed" and "SHA256/size mismatch" in result["error"]


def test_uncached_weights_are_reported_without_downloading(audit_module, cached_checkpoint):
    cache, snapshot, record = cached_checkpoint
    (snapshot / "model.safetensors").unlink()
    result = audit_module.audit_checkpoint(record, cache_dir=cache)
    assert result["status"] == "failed"
    assert result["uncached_advertised_weights"] == ["model.safetensors"]


def test_progressive_audit_artifact_records_every_requested_checkpoint(
    audit_module, cached_checkpoint, tmp_path
):
    cache, _, record = cached_checkpoint
    missing = {**record, "stage": "not_cached", "revision": "b" * 40}
    preflight, output = tmp_path / "preflight.json", tmp_path / "load_integrity.json"
    preflight.write_text(json.dumps({"checkpoints": [record, missing]}))
    result = audit_module.run_audit(preflight, output, cache_dir=cache)
    assert result["status"] == "failed"
    assert [r["status"] for r in result["checkpoints"]] == ["passed", "failed"]
    assert json.loads(output.read_text()) == result
    assert (
        result["cpu_only"] and not result["network_allowed"] and not result["inference_performed"]
    )


def test_incomplete_loading_info_cannot_be_mistaken_for_success(audit_module):
    with pytest.raises(ValueError, match="required integrity fields"):
        audit_module.normalized_loading_info({"missing_keys": []})
