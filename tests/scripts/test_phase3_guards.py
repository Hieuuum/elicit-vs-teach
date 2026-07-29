"""Negative tests for phase 3's four pre-spend refusal classes."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from geode.arith import order_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "experiments" / "training-run" / "scripts"
CONFIGS = REPO_ROOT / "experiments" / "training-run" / "configs"


def _load_guards():
    spec = importlib.util.spec_from_file_location("phase3_guards", SCRIPTS / "phase3_guards.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def _fixture(tmp_path: Path) -> tuple[object, Path, Path, Path]:
    guards = _load_guards()
    configs = tmp_path / "configs"
    configs.mkdir()
    for name in {
        "p3_elicit_parent.yaml",
        "p3_elicit_inst.yaml",
        "p3_elicit_target.yaml",
        "p3_teach_inst.yaml",
        "p3_embedding_warmstart.yaml",
        "eval_p3_data.yaml",
        "p3_bridge.yaml",
        "eval_p3_bridge_data.yaml",
        "lr_pin.yaml",
    }:
        shutil.copy(CONFIGS / name, configs / name)
    shutil.copytree(CONFIGS / "p3", configs / "p3")

    data = tmp_path / "artifact.parquet"
    rows = [
        {
            "a": 1,
            "b": 2,
            "op": "+",
            "shown_answer": 3,
            "format": "nl",
            "label_mode": "correct",
        }
    ]
    pd.DataFrame(rows).to_parquet(data, index=False)
    pin = order_hash(rows)
    for name, key in guards.ARTIFACT_PINS:
        cfg = yaml.safe_load((configs / name).read_text())
        cfg["data"][key] = str(data)
        hash_key = "eval_order_hash" if key.startswith("eval_") else "order_hash"
        cfg["data"][hash_key] = pin
        _write_yaml(configs / name, cfg)
    return guards, configs, tmp_path, data


def _set_lr(configs: Path, name: str, lr: float) -> None:
    cfg = yaml.safe_load((configs / name).read_text())
    cfg["train"]["lr"] = lr
    _write_yaml(configs / name, cfg)


def test_phase3_guards_passing_control(tmp_path: Path) -> None:
    guards, configs, repo_root, _data = _fixture(tmp_path)
    guards.check_phase3_guards(configs, repo_root)


def test_phase3_guards_refuse_missing_artifact(tmp_path: Path) -> None:
    guards, configs, repo_root, data = _fixture(tmp_path)
    data.unlink()
    with pytest.raises(SystemExit, match="does not exist"):
        guards.check_phase3_guards(configs, repo_root)


def test_phase3_teach_scope_ignores_unrelated_elicit_and_bridge_artifacts(
    tmp_path: Path,
) -> None:
    guards, configs, repo_root, data = _fixture(tmp_path)
    other = tmp_path / "unrelated.parquet"
    data.rename(other)
    for name, key in guards.TEACH_ARTIFACT_PINS:
        cfg = yaml.safe_load((configs / name).read_text())
        cfg["data"][key] = str(other)
        _write_yaml(configs / name, cfg)
    guards.check_phase3_guards(configs, repo_root, scope="teach")


def test_phase3_guards_refuse_corrupt_hash(tmp_path: Path) -> None:
    guards, configs, repo_root, _data = _fixture(tmp_path)
    cfg = yaml.safe_load((configs / "p3_bridge.yaml").read_text())
    cfg["data"]["order_hash"] = "0" * 64
    _write_yaml(configs / "p3_bridge.yaml", cfg)
    with pytest.raises(SystemExit, match="order_hash.*!= pinned"):
        guards.check_phase3_guards(configs, repo_root)


@pytest.mark.parametrize("name", ["p3_elicit_inst.yaml", "p3_teach_inst.yaml"])
def test_phase3_guards_refuse_installer_target_lr(tmp_path: Path, name: str) -> None:
    guards, configs, repo_root, _data = _fixture(tmp_path)
    _set_lr(configs, name, 1.0e-3)
    with pytest.raises(SystemExit, match="full-FT format stage pins the target lr"):
        guards.check_phase3_guards(configs, repo_root)


def test_phase3_guards_refuse_parent_target_lr(tmp_path: Path) -> None:
    guards, configs, repo_root, _data = _fixture(tmp_path)
    _set_lr(configs, "p3_elicit_parent.yaml", 1.0e-3)
    with pytest.raises(SystemExit, match="pre-intervention stage pins the TARGET lr"):
        guards.check_phase3_guards(configs, repo_root)


def test_phase3_warmstart_scope_ignores_unrelated_bridge_artifacts(tmp_path: Path) -> None:
    guards, configs, repo_root, data = _fixture(tmp_path)
    other = tmp_path / "warm.parquet"
    data.rename(other)
    for name, key in guards.WARMSTART_ARTIFACT_PINS:
        cfg = yaml.safe_load((configs / name).read_text())
        cfg["data"][key] = str(other)
        _write_yaml(configs / name, cfg)
    guards.check_phase3_guards(configs, repo_root, scope="warmstart")


def test_phase3_guards_refuse_warmstart_row_scope_drift(tmp_path: Path) -> None:
    guards, configs, repo_root, _data = _fixture(tmp_path)
    path = configs / "p3" / "warm_sum.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["warmstart"] = {"tokens": ["+"], "rows": [12]}
    _write_yaml(path, cfg)
    with pytest.raises(SystemExit, match="warm_sum.yaml must pin"):
        guards.check_phase3_guards(configs, repo_root, scope="warmstart")


def test_phase3_guards_refuse_warmstart_dose_drift(tmp_path: Path) -> None:
    guards, configs, repo_root, _data = _fixture(tmp_path)
    path = configs / "p3_embedding_warmstart.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["warmstart"]["steps"] = 201
    _write_yaml(path, cfg)
    with pytest.raises(SystemExit, match="steps=201"):
        guards.check_phase3_guards(configs, repo_root, scope="warmstart")


def test_phase3_guards_refuse_warmstart_target_prefix_drift(tmp_path: Path) -> None:
    guards, configs, repo_root, _data = _fixture(tmp_path)
    path = configs / "p3" / "target_warm_colon.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["data"]["n_examples"] = 500000
    _write_yaml(path, cfg)
    with pytest.raises(SystemExit, match="data.n_examples=100000"):
        guards.check_phase3_guards(configs, repo_root, scope="warmstart")


def test_phase3_guards_refuse_warmstart_target_g7_drift(tmp_path: Path) -> None:
    guards, configs, repo_root, _data = _fixture(tmp_path)
    path = configs / "p3" / "target_warm_sum_colon.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["experiment"]["match_data_order_with"] = "evt-p3-elicit-target"
    _write_yaml(path, cfg)
    with pytest.raises(SystemExit, match="match_data_order_with"):
        guards.check_phase3_guards(configs, repo_root, scope="warmstart")


def test_phase3_guards_refuse_warmstart_target_gate_requirement(tmp_path: Path) -> None:
    guards, configs, repo_root, _data = _fixture(tmp_path)
    path = configs / "p3" / "target_warm_sum.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["experiment"]["parent_required_gates"] = ["G2"]
    _write_yaml(path, cfg)
    with pytest.raises(SystemExit, match="must not require"):
        guards.check_phase3_guards(configs, repo_root, scope="warmstart")
