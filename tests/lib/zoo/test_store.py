"""``checkpoint_dir`` — flat/legacy final-checkpoint resolution (V0.8).

Silent failure mode guarded: loading the wrong checkpoint. The resolver must
find the run's single final checkpoint whether the run uses the flat layout
(``runs/{id}/model/``, specs/00 §1) or the legacy phase-dir layout
(``runs/{id}/{pretrain,sft}/model/``), and must refuse — naming the run dir —
whenever zero or several candidates exist. Stores are plain tmp trees; no
models are needed to test path resolution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geode.zoo import checkpoint_dir

RUN_ID = "evt-test-run"


def touch(store: Path, rel: str) -> Path:
    """Create an empty file at ``store/runs/RUN_ID/rel``, making parents."""
    path = store / "runs" / RUN_ID / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_v0_8_flat_layout_resolves(tmp_path):
    ckpt = touch(tmp_path, "model/model.safetensors")
    assert checkpoint_dir(RUN_ID, store=tmp_path) == ckpt.parent


@pytest.mark.parametrize("phase", ["pretrain", "sft"])
def test_v0_8_legacy_layout_resolves(tmp_path, phase):
    ckpt = touch(tmp_path, f"{phase}/model/model.safetensors")
    assert checkpoint_dir(RUN_ID, store=tmp_path) == ckpt.parent


def test_v0_8_snapshots_never_match(tmp_path):
    # Snapshots have no model/ path component (specs/00 §1) and must not
    # count as candidates alongside the flat checkpoint.
    ckpt = touch(tmp_path, "model/model.safetensors")
    touch(tmp_path, "snapshots/step_500/model.safetensors")
    touch(tmp_path, "snapshots/step_1000/model.safetensors")
    assert checkpoint_dir(RUN_ID, store=tmp_path) == ckpt.parent


def test_v0_8_flat_dir_without_weights_falls_through_to_legacy(tmp_path):
    # A flat model/ dir missing model.safetensors is not a candidate: only
    # the legacy checkpoint exists here.
    touch(tmp_path, "model/config.json")
    ckpt = touch(tmp_path, "sft/model/model.safetensors")
    assert checkpoint_dir(RUN_ID, store=tmp_path) == ckpt.parent


def test_v0_8_flat_plus_legacy_raises_naming_run_dir(tmp_path):
    touch(tmp_path, "model/model.safetensors")
    touch(tmp_path, "sft/model/model.safetensors")
    with pytest.raises(RuntimeError, match="ambiguous") as excinfo:
        checkpoint_dir(RUN_ID, store=tmp_path)
    assert str(tmp_path / "runs" / RUN_ID) in str(excinfo.value)


def test_v0_8_two_phase_dirs_raise_naming_run_dir(tmp_path):
    touch(tmp_path, "pretrain/model/model.safetensors")
    touch(tmp_path, "sft/model/model.safetensors")
    with pytest.raises(RuntimeError, match="ambiguous") as excinfo:
        checkpoint_dir(RUN_ID, store=tmp_path)
    assert str(tmp_path / "runs" / RUN_ID) in str(excinfo.value)


def test_v0_8_missing_raises_naming_run_dir(tmp_path):
    # Run dir exists (manifest only) but holds no checkpoint at all.
    touch(tmp_path, "manifest.json")
    with pytest.raises(FileNotFoundError) as excinfo:
        checkpoint_dir(RUN_ID, store=tmp_path)
    assert str(tmp_path / "runs" / RUN_ID) in str(excinfo.value)


def test_v0_8_absent_run_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        checkpoint_dir("never-created", store=tmp_path)
