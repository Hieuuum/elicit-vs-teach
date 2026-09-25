"""V5.71 — geode.train.assert_lr_scope: per-role LR scope guard.

The LoRA target rate is scoped to LoRA target runs; a full-FT stage that pins it
is the 2026-07-25 scope leak that destroyed run 9's retention. The guard refuses
a mispinned rate (wrong pin) and a cross-scoped rate (full-FT stage at the target
rate).
"""

from __future__ import annotations

import pytest

from geode.train import assert_lr_scope

PIN = {"lr": 1.0e-3, "installer_lr": 3.0e-6}


def _cfg(lr: float) -> dict:
    return {"train": {"lr": lr}}


def test_v5_71_matching_scopes_pass() -> None:
    # target at the target pin; full-FT stages at a non-target rate (installer/
    # bridge at the installer pin, parent role-inherited) all pass silently.
    assert_lr_scope(_cfg(1.0e-3), PIN, stage="target")
    assert_lr_scope(_cfg(3.0e-6), PIN, stage="installer")
    assert_lr_scope(_cfg(3.0e-6), PIN, stage="bridge")
    assert_lr_scope(_cfg(3.0e-6), PIN, stage="parent")


def test_v5_71_target_pin_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="target"):
        assert_lr_scope(_cfg(3.0e-4), PIN, stage="target")


def test_v5_71_installer_pin_mismatch_raises() -> None:
    # 1e-5 is neither the target nor the installer pin: a plain pin mismatch.
    with pytest.raises(ValueError, match="installer lr"):
        assert_lr_scope(_cfg(1.0e-5), PIN, stage="installer")


@pytest.mark.parametrize("stage", ["installer", "bridge", "parent"])
def test_v5_71_full_ft_at_target_rate_is_scope_leak(stage: str) -> None:
    with pytest.raises(ValueError, match="scope leak"):
        assert_lr_scope(_cfg(1.0e-3), PIN, stage=stage)  # target rate on a full-FT stage


def test_unknown_stage_raises() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        assert_lr_scope(_cfg(3.0e-6), PIN, stage="bogus")
