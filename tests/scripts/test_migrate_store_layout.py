"""Core move logic of ``migrate_store_layout.py`` (legacy phase dir -> flat).

Silent failure mode guarded: a botched move corrupts the store the GPU
budget already paid for. Tested on tmp trees only — the ``--relay`` path is
loud-failure HF plumbing and is deliberately untested (CLAUDE.md testing
policy: no network). The script is not a package module, so it is loaded
from its file path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from geode.zoo import checkpoint_dir

from tests._scriptloader import repo_root

REPO_ROOT = repo_root()
_SCRIPT = REPO_ROOT / "experiments" / "training-run" / "scripts" / "migrate_store_layout.py"
_spec = importlib.util.spec_from_file_location("migrate_store_layout", _SCRIPT)
mig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mig)

RUN_ID = "evt-test-run"


def make_legacy_run(store: Path, phase: str) -> Path:
    """A run dir shaped like the real legacy output of train.py/train_sft.py."""
    run_dir = store / "runs" / RUN_ID
    (run_dir / phase / "model").mkdir(parents=True)
    (run_dir / phase / "model" / "model.safetensors").write_text("weights")
    (run_dir / phase / "model" / "config.json").write_text("{}")
    (run_dir / phase / "train_log.jsonl").write_text('{"step": 1}\n')
    (run_dir / phase / "eval_log.jsonl").write_text('{"step": 1, "val_loss_nats": 1.0}\n')
    (run_dir / phase / "training_meta.json").write_text('{"stop_reason": "converged"}')
    (run_dir / "manifest.json").write_text('{"run_id": "evt-test-run"}')
    return run_dir


@pytest.mark.parametrize("phase", ["pretrain", "sft"])
def test_legacy_to_flat(tmp_path, phase):
    run_dir = make_legacy_run(tmp_path, phase)
    moves = mig.migrate_run(run_dir, apply=True)
    assert len(moves) == 4  # model/ + the three phase-dir files
    assert not (run_dir / phase).exists()
    assert (run_dir / "model" / "config.json").exists()
    assert (run_dir / "eval_log.jsonl").read_text().startswith('{"step": 1')
    assert (run_dir / "train_log.jsonl").exists()
    assert (run_dir / "training_meta.json").exists()
    assert (run_dir / "manifest.json").read_text() == '{"run_id": "evt-test-run"}'
    # The zoo resolver must agree the run is now flat.
    assert checkpoint_dir(RUN_ID, store=tmp_path) == run_dir / "model"


def test_dry_run_moves_nothing(tmp_path):
    run_dir = make_legacy_run(tmp_path, "sft")
    moves = mig.migrate_run(run_dir, apply=False)
    assert len(moves) == 4
    assert (run_dir / "sft" / "model" / "model.safetensors").exists()
    assert not (run_dir / "model").exists()


def test_idempotent_second_apply_is_noop(tmp_path):
    run_dir = make_legacy_run(tmp_path, "sft")
    mig.migrate_run(run_dir, apply=True)
    before = sorted(p.relative_to(run_dir) for p in run_dir.rglob("*"))
    assert mig.migrate_run(run_dir, apply=True) == []
    assert sorted(p.relative_to(run_dir) for p in run_dir.rglob("*")) == before


def test_collision_refused_and_nothing_moved(tmp_path):
    run_dir = make_legacy_run(tmp_path, "sft")
    (run_dir / "eval_log.jsonl").write_text("pre-existing\n")
    with pytest.raises(mig.MigrationError, match="destination already exists"):
        mig.migrate_run(run_dir, apply=True)
    # Refusal happens before the first rename: the legacy tree is intact.
    assert (run_dir / "sft" / "model" / "model.safetensors").exists()
    assert (run_dir / "sft" / "eval_log.jsonl").exists()
    assert (run_dir / "eval_log.jsonl").read_text() == "pre-existing\n"


def test_two_phase_dirs_refused(tmp_path):
    run_dir = make_legacy_run(tmp_path, "sft")
    (run_dir / "pretrain" / "model").mkdir(parents=True)
    with pytest.raises(mig.MigrationError, match="both phase dirs"):
        mig.migrate_run(run_dir, apply=True)
    assert (run_dir / "sft" / "model" / "model.safetensors").exists()
