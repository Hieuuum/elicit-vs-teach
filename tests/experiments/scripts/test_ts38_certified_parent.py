"""ts38 certified-checkpoint LoRA parent: overlays + selection rule.

decisions.md 2026-08-15 "ts38: owner fork taken (delegated -> owner-answered):
probe -> EARLIEST-CERTIFIED-CHECKPOINT parent pre-authorized" entry. Every
converged LoRA/full-FT parent HALTed on G8 (see the two entries above it in
decisions.md). The owner pre-authorized promoting the earliest probe-measured
step that passes both bars (with one step of G1 persistence) to a bit-exact,
max_steps-pinned replay of the 3e-4 LoRA config — `certified_step.py`'s
`select_certified_step` is the pure selection function over the probe table
(results/ts38_parent_lora_probe.json), and
`configs/sweeps/ts38/parent_lora_certified_s<S>.yaml` for S in
{10000, ..., 24000} step 1000 are the 15 pre-declared horizons it can pick
from — `launch_ts38_certified_parent.sh` never launches at an S outside this
set.

Two failure classes are guarded here, mirroring the existing ts38 LoRA-parent
tests (test_ts38_lora_*): (1) the CLAUDE.md promotion-rule *cost* clause — a
config that is internally valid YAML but missing a key `manifest_fields`
needs bites minutes into a paid box, after the GPU already spent (see that
module's docstring for the concrete 2026-07-27 incident); (2) the selection
rule itself is a silent-failure risk (CLAUDE.md's *silence* clause) — picking
the wrong step, or picking a step at all when persistence was not actually
met, would certify a parent the owner did not pre-authorize. It is pure and
argv/IO-free specifically so it can be checked without a probe run.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from tests._scriptloader import load, repo_root

CONFIGS = repo_root() / "experiments" / "training-run" / "configs"
STEPS_PER_EPOCH = 7773  # floor((1,000,000 - 5,000) / 128) — same pin as the base config

load_config = load("train").load_config
select_certified_step = load("certified_step").select_certified_step

CERTIFIED_STEPS = list(range(10000, 24001, 1000))


# =============================================================================
# Part 1 — the 15 overlays
# =============================================================================


@pytest.mark.parametrize("step", CERTIFIED_STEPS)
def test_certified_overlay_exists(step: int) -> None:
    path = CONFIGS / "sweeps" / "ts38" / f"parent_lora_certified_s{step}.yaml"
    assert path.is_file(), path


@pytest.mark.parametrize("step", CERTIFIED_STEPS)
def test_certified_overlay_merged_values(step: int) -> None:
    overlay = CONFIGS / "sweeps" / "ts38" / f"parent_lora_certified_s{step}.yaml"
    cfg = load_config(CONFIGS / "ts38_pretaught_parent_lora.yaml", overlay)
    assert cfg["run_id"] == "evt-ts38-pretaught-parent"
    assert isinstance(cfg["train"]["lr"], float)
    assert cfg["train"]["lr"] == pytest.approx(3.0e-4)
    assert cfg["train"]["max_steps"] == step
    assert cfg["train"]["epochs_total_planned"] == math.ceil(step / STEPS_PER_EPOCH)


@pytest.mark.parametrize("step", CERTIFIED_STEPS)
def test_certified_overlay_builds_a_lora_manifest(step: int) -> None:
    """Mirror of test_ts38_lora_parent_builds_a_lora_manifest: manifest_fields's
    LoRA branch must not raise for any of the 15 overlays before a GPU is ever
    spent, since only the probe-selected one gets exercised on a real run."""
    train_sft = load("train_sft")
    lora_path = CONFIGS / "ts38_pretaught_parent_lora.yaml"
    overlay = CONFIGS / "sweeps" / "ts38" / f"parent_lora_certified_s{step}.yaml"
    cfg = load_config(lora_path, overlay)
    lora_cfg = train_sft.own_lora_block(cfg, lora_path, overlay)
    assert lora_cfg is not None
    manifest = train_sft.manifest_fields(
        cfg,
        n_params=1,
        n_rows=1,
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="bf16",
        lora_cfg=lora_cfg,
        step0={},
        device="cpu",
    )
    assert manifest["training"]["method"] == "lora"


def test_certified_overlay_run_ids_share_the_parent_slot() -> None:
    """All 15 overlays keep the SAME run_id as the sibling LoRA-parent files —
    exactly one of them is ever launched (selected at runtime), reusing the
    ladder's one parent slot, never a distinct sweep-style id."""
    for step in CERTIFIED_STEPS:
        raw = yaml.safe_load(
            (CONFIGS / "sweeps" / "ts38" / f"parent_lora_certified_s{step}.yaml").read_text()
        )
        assert raw["run_id"] == "evt-ts38-pretaught-parent"


# =============================================================================
# Part 3 — select_certified_step (pure function, no probe run needed)
# =============================================================================


def _row(step: int, g1_pass: bool, g8_pass: bool) -> dict:
    return {
        "lr": "3e-4",
        "step": step,
        "g1_accuracy": 0.96 if g1_pass else 0.80,
        "g1_pass": g1_pass,
        "g8_val_loss_nats": 1.17 if g8_pass else 1.20,
        "g8_pass": g8_pass,
    }


def test_select_earliest_both_pass_with_persistence() -> None:
    rows = [
        _row(10000, g1_pass=False, g8_pass=True),
        _row(11000, g1_pass=True, g8_pass=True),  # both-pass, S+1000 g1_pass True -> qualifies
        _row(12000, g1_pass=True, g8_pass=True),
        _row(13000, g1_pass=True, g8_pass=False),
    ]
    result = select_certified_step(rows)
    assert result["step"] == 11000
    assert result["first_g1_pass_step"] == 11000
    assert result["last_g8_pass_step"] == 12000
    assert result["both_pass_steps"] == [11000, 12000]


def test_select_skips_a_both_pass_step_whose_next_row_fails_g1() -> None:
    rows = [
        _row(10000, g1_pass=True, g8_pass=True),  # both-pass, but 11000 fails G1 -> skipped
        _row(11000, g1_pass=False, g8_pass=True),
        _row(12000, g1_pass=True, g8_pass=True),  # both-pass, 13000 g1_pass True -> qualifies
        _row(13000, g1_pass=True, g8_pass=True),
    ]
    result = select_certified_step(rows)
    assert result["step"] == 12000
    assert result["both_pass_steps"] == [10000, 12000, 13000]
    assert result["first_g1_pass_step"] == 10000
    assert result["last_g8_pass_step"] == 13000


def test_select_lone_both_pass_at_final_row_is_none() -> None:
    rows = [
        _row(10000, g1_pass=False, g8_pass=True),
        _row(11000, g1_pass=True, g8_pass=False),
        _row(12000, g1_pass=True, g8_pass=True),  # final row, no S+1000 -> cannot persist
    ]
    result = select_certified_step(rows)
    assert result["step"] is None
    assert result["both_pass_steps"] == [12000]
    assert result["first_g1_pass_step"] == 11000
    assert result["last_g8_pass_step"] == 12000


def test_select_no_both_pass_at_all() -> None:
    rows = [
        _row(10000, g1_pass=False, g8_pass=True),
        _row(11000, g1_pass=True, g8_pass=False),
    ]
    result = select_certified_step(rows)
    assert result["step"] is None
    assert result["both_pass_steps"] == []
    assert result["first_g1_pass_step"] == 11000
    assert result["last_g8_pass_step"] == 10000


def test_select_handles_unsorted_rows() -> None:
    rows = [
        _row(13000, g1_pass=True, g8_pass=False),
        _row(11000, g1_pass=True, g8_pass=True),
        _row(10000, g1_pass=False, g8_pass=True),
        _row(12000, g1_pass=True, g8_pass=True),
    ]
    result = select_certified_step(rows)
    assert result["step"] == 11000
    assert result["both_pass_steps"] == [11000, 12000]
