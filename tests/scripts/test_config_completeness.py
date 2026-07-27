"""Every committed run config satisfies the manifest builder it will be fed to.

Silent failure mode guarded: none — this one fails LOUDLY. It earns a test on
the other clause of the CLAUDE.md promotion rule, cost. ``manifest_fields``
raises a bare ``KeyError`` deep inside ``register_run``, which on a real launch
happens *after* the config is parsed, 500K rows are tokenized, and the
checkpoint is loaded onto the GPU — several minutes into a paid box, with the
budget confirmation already given. That is what happened to
``p3_elicit_parent.yaml`` on 2026-07-27: it lacked ``train.epochs_total_planned``
(a bare subscript in ``train_sft.py``, present in every pre-phase-3 full-FT
config), and ``p3_elicit_inst.yaml`` carried the same gap so the installer would
have hit it again later in the same chain.

The launcher guards hashes, LR pins, and checkpoint presence before spending;
this closes the remaining gap, that a config can be internally valid and still
be missing a key the trainer requires. It is a smoke test, not a property test:
it asserts the call does not raise, and says nothing about the values.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "experiments" / "training-run" / "scripts"
CONFIGS = REPO_ROOT / "experiments" / "training-run" / "configs"


def _load(name: str):
    """Import a script by path; they are not package modules and import siblings."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(SCRIPTS))


# Exactly the configs a launcher shell script passes to each trainer by name:
#   grep -o 'train_(sft|target).py --config \S+' scripts/*.sh
# Runs 1-4 predate these launchers and went through train.py, which has its own
# manifest builder and a config shape that train_sft.py cannot read (no `task`
# block in run1_pretrain.yaml) — they are not in scope here. train_target.py
# derives epochs_total itself rather than reading it from the config, which is
# why only the full-FT side needs epochs_total_planned.
FULL_FT = [
    "p2_armB_instperm.yaml",
    "run9_llama1b_inst.yaml",
    "p3_elicit_parent.yaml",
    "p3_elicit_inst.yaml",
    "p3_bridge.yaml",
]
LORA_TARGET = [
    ("run10_llama1b_target.yaml", None),
    ("p3_elicit_target.yaml", None),
    ("p3_elicit_target.yaml", "p3/target_after_bridge.yaml"),
]


@pytest.mark.parametrize("config", FULL_FT)
def test_full_ft_configs_build_a_manifest(config: str) -> None:
    train_sft = _load("train_sft")
    from train import load_config  # noqa: PLC0415 — SCRIPTS is only on sys.path via _load

    cfg = load_config(CONFIGS / config, None)
    train_sft.manifest_fields(
        cfg,
        n_params=1,
        n_rows=1,
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="fp32",
        lora_cfg=None,
        step0={},
        device="cpu",
    )


@pytest.mark.parametrize(("config", "override"), LORA_TARGET)
def test_lora_target_configs_build_a_manifest(config: str, override: str | None) -> None:
    train_target = _load("train_target")
    from train import load_config  # noqa: PLC0415

    override_path = CONFIGS / override if override is not None else None
    cfg = load_config(CONFIGS / config, override_path)
    train_target.manifest_fields(
        cfg,
        n_train=1,
        n_trainable=1,
        epochs_total=1,
        schedule=[],
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="fp32",
    )
