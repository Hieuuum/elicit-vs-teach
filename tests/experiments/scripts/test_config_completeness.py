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

import math
import re
from pathlib import Path

import pytest
import yaml

from tests._scriptloader import load, repo_root

CONFIGS = repo_root() / "experiments" / "training-run" / "configs"

# ``train`` holds the shared config loader; loading it puts SCRIPTS on sys.path
# so the trainer modules' own ``from train import ...`` resolve.
load_config = load("train").load_config


# Exactly the configs a launcher shell script passes to each trainer by name:
#   grep -o 'train_(sft|target).py --config \S+' scripts/*.sh
# Runs 1-4 predate these launchers and went through train.py, which has its own
# manifest builder and a config shape that train_sft.py cannot read (no `task`
# block in run1_pretrain.yaml) — they are not in scope here. train_target.py
# derives epochs_total itself rather than reading it from the config, which is
# why only the full-FT side needs epochs_total_planned.
FULL_FT = [
    "archive/phase2/p2_armB_instperm.yaml",
    "archive/runs/run9_llama1b_inst.yaml",
    "archive/phase3/p3_elicit_parent.yaml",
    "archive/phase3/p3_elicit_inst.yaml",
    "archive/phase3/p3_teach_inst.yaml",
    "archive/phase3/p3_bridge.yaml",
    "archive/runs/llama_fig2_installer.yaml",
    "llama_fig2nl_installer.yaml",
    "llama_fig2nl2_installer.yaml",
    "llama_fig2nl3_installer.yaml",
]
EMBEDDING_WARMSTART = [
    "archive/phase3/p3/warm_sum.yaml",
    "archive/phase3/p3/warm_colon.yaml",
    "archive/phase3/p3/warm_sum_colon.yaml",
]
LORA_TARGET = [
    ("archive/runs/run10_llama1b_target.yaml", None),
    ("archive/runs/llama_fig2_noinst.yaml", None),
    ("archive/runs/llama_fig2_inst.yaml", None),
    ("llama_fig2nl_noinst.yaml", None),
    ("llama_fig2nl_inst.yaml", None),
    ("llama_fig2nl2_noinst.yaml", None),
    ("llama_fig2nl2_inst.yaml", None),
    ("llama_fig2nl3_noinst.yaml", None),
    ("llama_fig2nl3_inst.yaml", None),
    ("archive/phase3/p3_elicit_target.yaml", None),
    ("archive/phase3/p3_elicit_target.yaml", "archive/phase3/p3/target_after_bridge.yaml"),
    ("archive/phase3/p3_elicit_target.yaml", "archive/phase3/p3/target_after_recover.yaml"),
    ("archive/phase3/p3_elicit_target.yaml", "archive/phase3/p3/target_on_bridge.yaml"),
    ("archive/phase3/p3_elicit_target.yaml", "archive/phase3/p3/target_teach.yaml"),
    ("archive/phase3/p3_elicit_target.yaml", "archive/phase3/p3/target_warm_sum.yaml"),
    ("archive/phase3/p3_elicit_target.yaml", "archive/phase3/p3/target_warm_colon.yaml"),
    ("archive/phase3/p3_elicit_target.yaml", "archive/phase3/p3/target_warm_sum_colon.yaml"),
]


@pytest.mark.parametrize("override", EMBEDDING_WARMSTART)
def test_embedding_warmstart_configs_build_a_manifest(override: str) -> None:
    trainer = load("train_embedding_warmstart")

    cfg = load_config(CONFIGS / "archive/phase3/p3_embedding_warmstart.yaml", CONFIGS / override)
    cfg["run_id"] = trainer.candidate_run_id(cfg["run_id"], 0.01)
    manifest = trainer.manifest_fields(
        cfg,
        candidate={
            "lr": 0.01,
            "operator_accuracy": 0.0,
            "nl_accuracy": 0.5,
            "nl_loss_nats": 1.0,
        },
        n_trainable=1,
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
    )
    assert manifest["run_id"].endswith("-lr1e-2")
    assert manifest["experiment"]["gates"] == {}
    assert manifest["experiment"]["embedding_warmstart"]["final"]["operator_accuracy"] == 0.0


def test_llama_fig2nl_installer_opts_into_lora_via_own_lora_block() -> None:
    """Pins the CRITICAL gating footgun train_sft.py's ``own_lora_block``
    docstring warns about: common.yaml merges a default ``lora:`` block into
    every config (including full-FT ones), so gating on the merged
    ``cfg["lora"]`` — instead of the run's OWN ``lora:`` key — would silently
    switch every full-FT launch to LoRA (and, post-2026-07-31, write a bogus
    adapter sidecar for weights that were never LoRA-trained). The
    redesigned fig2nl installer must opt in via its own key; a genuine
    full-FT config must not."""
    train_sft = load("train_sft")

    lora_path = CONFIGS / "llama_fig2nl_installer.yaml"
    lora_cfg = train_sft.own_lora_block(load_config(lora_path, None), lora_path, None)
    assert lora_cfg is not None
    assert (lora_cfg["r"], lora_cfg["alpha"]) == (512, 32)

    full_ft_path = CONFIGS / "archive/phase2/p2_armB_instperm.yaml"
    assert train_sft.own_lora_block(load_config(full_ft_path, None), full_ft_path, None) is None


@pytest.mark.parametrize("config", FULL_FT)
def test_full_ft_configs_build_a_manifest(config: str) -> None:
    train_sft = load("train_sft")

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
    train_target = load("train_target")

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


@pytest.mark.parametrize(
    ("override", "match_with"),
    [
        ("archive/phase3/p3/target_warm_sum.yaml", None),
        ("archive/phase3/p3/target_warm_colon.yaml", "evt-p3-warm-sum-target"),
        ("archive/phase3/p3/target_warm_sum_colon.yaml", "evt-p3-warm-sum-target"),
    ],
)
def test_warmstart_targets_pin_one_100k_pass(override: str, match_with: str | None) -> None:
    cfg = load_config(CONFIGS / "archive/phase3/p3_elicit_target.yaml", CONFIGS / override)
    assert cfg["data"]["n_examples"] == 100000
    assert cfg["train"]["batch_size"] == 128
    assert cfg["train"]["max_steps"] == 782
    assert cfg["train"]["stopping"]["min_steps"] == 782
    assert cfg["experiment"]["parent_required_gates"] == []
    assert cfg["experiment"]["fixed_prefix_one_pass"] is True
    assert cfg["experiment"]["match_data_order_with"] == match_with


# =============================================================================
# ts38 mini (project-ts38-mini-plan-2026-08-14.md; EXPERIMENTS §6.14)
#
# Config-level checks for the ratified plan's pinned values, on top of the
# manifest-builder smoke coverage above. Distinct scope from
# tests/experiments/scripts/test_full_epoch1_guard.py, which exercises the
# require_full_epoch1 mechanics (train_target.py) in isolation with
# synthetic inputs; this section checks the COMMITTED YAML actually carries
# the values the plan pins.
# =============================================================================

# (relative path, n_examples, eval_every, max_steps, min_steps=ceil(n/128)).
# eval_every/max_steps are copied verbatim from the matching-size
# llama_fig2nl3 overlay (llama_fig2nl3_{inst,noinst}_n<size>.yaml);
# min_steps is the ts38-mini guard-1 value (require_full_epoch1).
# max_steps for n100000/n316228 raised 2026-08-15 (commit 6b735f1) to meet
# the >=20-epoch cost-ceiling floor (ceilings must never bind) —
# 10000->15625 and 30000->50000; this pin was stale until fixed here
# (found while adding the LoRA-parent tests below; re-read directly off the
# committed yaml, not the commit message).
TS38_OVERLAYS = [
    ("sweeps/ts38/ts38_base_n1000.yaml", 1000, 5, 1000, 8),
    ("sweeps/ts38/ts38_base_n4642.yaml", 4642, 5, 1000, 37),
    ("sweeps/ts38/ts38_base_n21544.yaml", 21544, 25, 5000, 169),
    ("sweeps/ts38/ts38_base_n100000.yaml", 100000, 125, 15625, 782),
    ("sweeps/ts38/ts38_base_n316228.yaml", 316228, 375, 50000, 2471),
    ("sweeps/ts38/ts38_pretaught_n1000.yaml", 1000, 5, 1000, 8),
    ("sweeps/ts38/ts38_pretaught_n4642.yaml", 4642, 5, 1000, 37),
    ("sweeps/ts38/ts38_pretaught_n21544.yaml", 21544, 25, 5000, 169),
    ("sweeps/ts38/ts38_pretaught_n100000.yaml", 100000, 125, 15625, 782),
    ("sweeps/ts38/ts38_pretaught_n316228.yaml", 316228, 375, 50000, 2471),
]
TS38_OVERLAY_PATHS = [row[0] for row in TS38_OVERLAYS]
TS38_BASE_FILES = ["ts38_pretaught_parent.yaml", "ts38_base.yaml", "ts38_pretaught.yaml"]
TS38_FALLBACK_OVERLAY = "sweeps/ts38/parent_lr_1e-4.yaml"
TS38_FILES = [*TS38_BASE_FILES, *TS38_OVERLAY_PATHS, TS38_FALLBACK_OVERLAY]
# run_id-bearing files: every ts38 file except the LR-fallback overlay, which
# (like installer_lr_1e-4.yaml elsewhere in the repo) carries only a train:
# override and no run_id of its own.
TS38_RUN_ID_FILES = [*TS38_BASE_FILES, *TS38_OVERLAY_PATHS]
TS38_PRETAUGHT_OVERLAY_ANCHORS = [
    ("sweeps/ts38/ts38_pretaught_n1000.yaml", "evt-ts38-base-n1000"),
    ("sweeps/ts38/ts38_pretaught_n4642.yaml", "evt-ts38-base-n4642"),
    ("sweeps/ts38/ts38_pretaught_n21544.yaml", "evt-ts38-base-n21544"),
    ("sweeps/ts38/ts38_pretaught_n100000.yaml", "evt-ts38-base-n100000"),
    ("sweeps/ts38/ts38_pretaught_n316228.yaml", "evt-ts38-base-n316228"),
]

# ARM_REGIME (train_target.py) hardcodes A=elicit/B=teach across every prior
# run family in this repo (run1-10, phase2, phase3, without exception); the
# plan's own prose table uses "Arm A"/"Arm B" with the OPPOSITE polarity for
# ts38 (see ts38_base.yaml's "ARM-LETTER NOTE"). `experiment.arm` therefore
# MUST differ between the two base yamls to get a correct manifest `regime`
# — it joins run_id/parent_run_id/parent_required_gates/match_data_order_with
# in the allowed-diff set below.
_TS38_ALLOWED_DIFF_PATHS = {
    "run_id",
    "experiment.parent_run_id",
    "experiment.parent_required_gates",
    "experiment.match_data_order_with",
    "experiment.arm",
}

_TS38_RUN_ID_PATTERN = re.compile(r"^evt-ts38-(pretaught-parent|base(-n\d+)?|pretaught(-n\d+)?)$")


def _leaf_diffs(a: dict, b: dict, prefix: str = "") -> set[str]:
    """Dotted paths where two nested dicts disagree (missing key counts as a diff)."""
    diffs: set[str] = set()
    for key in set(a) | set(b):
        path = f"{prefix}.{key}" if prefix else key
        av, bv = a.get(key, "<missing>"), b.get(key, "<missing>")
        if isinstance(av, dict) and isinstance(bv, dict):
            diffs |= _leaf_diffs(av, bv, path)
        elif av != bv:
            diffs.add(path)
    return diffs


def test_ts38_family_files_exist() -> None:
    assert len(TS38_FILES) == 14
    for rel in TS38_FILES:
        assert (CONFIGS / rel).is_file(), rel


def test_ts38_eval_bare_pin_exists() -> None:
    # 15th file, deliberately outside the 14-file family count (plan: create
    # only because the TS-chain has no bare-eval pin equivalent).
    assert (CONFIGS / "eval_bare_target_data_ts38.yaml").is_file()


@pytest.mark.parametrize("path", TS38_RUN_ID_FILES)
def test_ts38_run_ids_match_pattern(path: str) -> None:
    raw = yaml.safe_load((CONFIGS / path).read_text())
    assert _TS38_RUN_ID_PATTERN.match(raw["run_id"]), raw["run_id"]


@pytest.mark.parametrize(
    ("path", "n_examples", "eval_every", "max_steps", "min_steps"), TS38_OVERLAYS
)
def test_ts38_overlay_pinned_values(
    path: str, n_examples: int, eval_every: int, max_steps: int, min_steps: int
) -> None:
    raw = yaml.safe_load((CONFIGS / path).read_text())
    assert raw["data"]["n_examples"] == n_examples
    assert raw["train"]["eval_every"] == eval_every
    assert raw["train"]["max_steps"] == max_steps
    assert raw["train"]["stopping"]["min_steps"] == min_steps
    # Orchestrator addendum, sharpened: require_full_epoch1 checks ONLY
    # min_steps == ceil(n/128); it does not validate max_steps. A bare
    # max_steps >= min_steps is not the real bar — the eps/k rule cannot
    # fire until min_steps PLUS k more eval_every-spaced evaluations, so the
    # earliest possible convergence stop is min_steps + k*eval_every. A
    # config with max_steps below THAT still hits the ceiling mid-epoch-1
    # (the eps/k rule literally never gets a chance to fire) and only fails
    # the POST-run truncation guard, after GPU spend.
    base = "ts38_pretaught.yaml" if "pretaught" in path else "ts38_base.yaml"
    cfg = load_config(CONFIGS / base, CONFIGS / path)
    k = cfg["train"]["stopping"]["k"]
    assert cfg["train"]["max_steps"] >= min_steps + k * eval_every


@pytest.mark.parametrize("path", TS38_BASE_FILES[1:])  # the two target bases only
def test_ts38_base_placeholder_max_steps_covers_min_steps(path: str) -> None:
    cfg = load_config(CONFIGS / path, None)
    t = cfg["train"]
    assert t["max_steps"] >= t["stopping"]["min_steps"] + t["stopping"]["k"] * t["eval_every"]


def test_ts38_pretaught_parent_max_steps_covers_min_steps() -> None:
    # Same earliest-possible-stop arithmetic as the target overlays, for the
    # full-FT parent's own eps/k rule. min_steps 5000 + k*eval_every 5000 =
    # 10000, just over one epoch (floor((1e6-5e3)/128) = 7773) of the 40000
    # ceiling — the parent cannot converge before roughly one full pass.
    cfg = load_config(CONFIGS / "ts38_pretaught_parent.yaml", None)
    t = cfg["train"]
    assert t["max_steps"] >= t["stopping"]["min_steps"] + t["stopping"]["k"] * t["eval_every"]


@pytest.mark.parametrize(("path", "anchor"), TS38_PRETAUGHT_OVERLAY_ANCHORS)
def test_ts38_pretaught_overlay_match_data_order_with(path: str, anchor: str) -> None:
    raw = yaml.safe_load((CONFIGS / path).read_text())
    assert raw["experiment"]["match_data_order_with"] == anchor


@pytest.mark.parametrize("path", ["ts38_base.yaml", "ts38_pretaught.yaml"])
def test_ts38_require_full_epoch1_true(path: str) -> None:
    cfg = load_config(CONFIGS / path, None)
    assert cfg["experiment"]["require_full_epoch1"] is True


@pytest.mark.parametrize("path", ["ts38_base.yaml", "ts38_pretaught.yaml"])
def test_ts38_lora_r128_alpha32(path: str) -> None:
    cfg = load_config(CONFIGS / path, None)
    assert cfg["lora"]["r"] == 128
    assert cfg["lora"]["alpha"] == 32


def test_ts38_target_lr_identical_both_arms() -> None:
    base = load_config(CONFIGS / "ts38_base.yaml", None)
    pretaught = load_config(CONFIGS / "ts38_pretaught.yaml", None)
    assert base["train"]["lr"] == pretaught["train"]["lr"] == 1.0e-3


@pytest.mark.parametrize("path", ["ts38_base.yaml", "ts38_pretaught.yaml"])
def test_ts38_snapshots_off(path: str) -> None:
    cfg = load_config(CONFIGS / path, None)
    assert cfg["train"]["snapshots"]["n"] == 0


def test_ts38_parent_has_no_own_lora_block() -> None:
    """The full-FT footgun this family is most exposed to: common.yaml's
    shared default lora block must NOT cause the parent to train LoRA (see
    test_llama_fig2nl_installer_opts_into_lora_via_own_lora_block above)."""
    train_sft = load("train_sft")
    path = CONFIGS / "ts38_pretaught_parent.yaml"
    assert train_sft.own_lora_block(load_config(path, None), path, None) is None


def test_ts38_data_hashes_pinned() -> None:
    parent = load_config(CONFIGS / "ts38_pretaught_parent.yaml", None)
    assert (
        parent["data"]["order_hash"]
        == "69e3b09e2dd599e4ad8948fe2a5a19e67989be51bc006e8d4e220818ce16d0f7"
    )
    eval_target = yaml.safe_load((CONFIGS / "eval_target_data.yaml").read_text())
    assert (
        eval_target["data"]["order_hash"]
        == "588da81e0c016b8f0b3575a3aa7ad9b8e4c4421352a46fb4a30cd9ac713f6cb8"
    )
    for path in ("ts38_base.yaml", "ts38_pretaught.yaml"):
        cfg = load_config(CONFIGS / path, None)
        assert (
            cfg["data"]["order_hash"]
            == "946b5d02a8f9260fec00ce68a4db42a12f16966f6b49f685269382ae7b4b6ace"
        )
        assert (
            cfg["data"]["eval_order_hash"]
            == "e419baa213bbe07dfeb50f46fe17b464056cd18c2a7302238a66682d7c594631"
        )
    eval_bare = yaml.safe_load((CONFIGS / "eval_bare_target_data_ts38.yaml").read_text())
    assert (
        eval_bare["data"]["order_hash"]
        == "e419baa213bbe07dfeb50f46fe17b464056cd18c2a7302238a66682d7c594631"
    )


def test_ts38_arms_differ_only_in_theta0() -> None:
    base = yaml.safe_load((CONFIGS / "ts38_base.yaml").read_text())
    pretaught = yaml.safe_load((CONFIGS / "ts38_pretaught.yaml").read_text())
    diffs = _leaf_diffs(base, pretaught)
    unexplained = diffs - _TS38_ALLOWED_DIFF_PATHS
    assert not unexplained, f"ts38 arms differ in unexpected fields: {unexplained}"
    # And the fields that MUST differ (theta0 + its consequences) actually do.
    assert base["run_id"] != pretaught["run_id"]
    assert base["experiment"]["parent_run_id"] != pretaught["experiment"]["parent_run_id"]
    assert base["experiment"]["arm"] != pretaught["experiment"]["arm"]


def test_ts38_pretaught_parent_builds_a_full_ft_manifest() -> None:
    train_sft = load("train_sft")
    cfg = load_config(CONFIGS / "ts38_pretaught_parent.yaml", None)
    manifest = train_sft.manifest_fields(
        cfg,
        n_params=1,
        n_rows=1,
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="bf16",
        lora_cfg=None,
        step0={},
        device="cpu",
    )
    assert manifest["training"]["method"] == "full_ft"


def test_ts38_pretaught_parent_fallback_lr_overlay_builds_a_manifest() -> None:
    train_sft = load("train_sft")
    cfg = load_config(CONFIGS / "ts38_pretaught_parent.yaml", CONFIGS / TS38_FALLBACK_OVERLAY)
    assert cfg["train"]["lr"] == 1.0e-4
    train_sft.manifest_fields(
        cfg,
        n_params=1,
        n_rows=1,
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="bf16",
        lora_cfg=None,
        step0={},
        device="cpu",
    )


@pytest.mark.parametrize(
    ("path", "expected_regime"),
    [("ts38_base.yaml", "teach"), ("ts38_pretaught.yaml", "elicit")],
)
def test_ts38_target_base_builds_a_manifest_with_correct_regime(
    path: str, expected_regime: str
) -> None:
    """Encodes the ARM-LETTER NOTE as a regression-proof assertion: whichever
    `experiment.arm` value each file carries, the resulting manifest must
    tag the TEACH run "teach" and the ELICIT run "elicit" — never the
    reverse, and never silently "fixed" back to match the plan's prose
    letters without also updating train_target.py's ARM_REGIME."""
    train_target = load("train_target")
    cfg = load_config(CONFIGS / path, None)
    manifest = train_target.manifest_fields(
        cfg,
        n_train=1,
        n_trainable=1,
        epochs_total=1,
        schedule=[],
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="bf16",
    )
    assert manifest["regime"] == expected_regime


@pytest.mark.parametrize("path", TS38_OVERLAY_PATHS)
def test_ts38_overlay_builds_a_manifest(path: str) -> None:
    train_target = load("train_target")
    base = "ts38_pretaught.yaml" if "pretaught" in path else "ts38_base.yaml"
    cfg = load_config(CONFIGS / base, CONFIGS / path)
    train_target.manifest_fields(
        cfg,
        n_train=1,
        n_trainable=1,
        epochs_total=1,
        schedule=[],
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="bf16",
    )


# =============================================================================
# ts38 LoRA-installed parent (decisions.md 2026-08-15, "ts38 ladder CLOSED:
# 1e-5 converged ... full-FT parent is a DESIGN RESULT; LoRA-installed
# parent pre-declared"). Every rung of the full-FT ladder above
# (ts38_pretaught_parent.yaml + parent_lr_*.yaml) either passed G1 and
# failed G8, or (the bottom rung) failed G1 outright — full FT cannot
# certify a parent under both bars. ts38_pretaught_parent_lora.yaml
# supersedes it as the thing actually launched, reusing the SAME run_id
# (evt-ts38-pretaught-parent); the two base configs are mutually exclusive
# ways of producing that one run, never launched together. Kept as its own
# section (not folded into TS38_FILES/TS38_RUN_ID_FILES above) because
# those lists are indexed positionally elsewhere (TS38_BASE_FILES[1:]) and
# this family's run-id shape (sweep ids, not `-n<size>` overlays) does not
# fit _TS38_RUN_ID_PATTERN.
# =============================================================================

LORA_PARENT_CONFIG = "ts38_pretaught_parent_lora.yaml"
LORA_LR_TOKENS = {
    "1e-3": 1.0e-3,
    "3e-4": 3.0e-4,
    "1e-4": 1.0e-4,
    "3e-5": 3.0e-5,
}
LORA_SWEEP_OVERLAYS = [f"sweeps/ts38/parent_lora_sweep_lr_{tok}.yaml" for tok in LORA_LR_TOKENS]
LORA_FULL_OVERLAYS = [f"sweeps/ts38/parent_lora_lr_{tok}.yaml" for tok in LORA_LR_TOKENS]
LORA_FILES = [LORA_PARENT_CONFIG, *LORA_SWEEP_OVERLAYS, *LORA_FULL_OVERLAYS]

# edl_converged_val_floor.py's ts38 family regex — sweep/winner run ids must
# NOT match it (they are infrastructure, not `-n<size>` target overlays).
_TS38_FAMILY_REGEX = re.compile(r"^evt-ts38-(base|pretaught)-n(\d+)$")
# launch_ts38_mini.sh's ladder skip pattern for the retired full-FT ladder.
_TS38_LADDER_SKIP_PATTERN = re.compile(r"^evt-ts38-pretaught-parent-lr")


def test_ts38_lora_parent_family_files_exist() -> None:
    assert len(LORA_FILES) == 9
    for rel in LORA_FILES:
        assert (CONFIGS / rel).is_file(), rel


def test_ts38_lora_parent_config_merged_values() -> None:
    cfg = load_config(CONFIGS / LORA_PARENT_CONFIG, None)
    assert cfg["train"]["lr"] is None
    assert cfg["train"]["max_steps"] >= 20 * 7773
    assert cfg["run_id"] == "evt-ts38-pretaught-parent"
    parent = load_config(CONFIGS / "ts38_pretaught_parent.yaml", None)
    assert cfg["data"]["order_hash"] == parent["data"]["order_hash"]


def test_ts38_lora_parent_own_yaml_declares_lora() -> None:
    # The un-merged file, not the deep-merged cfg — this is what
    # own_lora_block actually reads to opt in (see the test below).
    raw = yaml.safe_load((CONFIGS / LORA_PARENT_CONFIG).read_text())
    assert raw["lora"]["r"] == 128
    assert raw["lora"]["alpha"] == 32


def test_ts38_lora_parent_merged_target_modules() -> None:
    cfg = load_config(CONFIGS / LORA_PARENT_CONFIG, None)
    assert cfg["lora"]["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


@pytest.mark.parametrize("token", sorted(LORA_LR_TOKENS))
def test_ts38_lora_sweep_overlay_values(token: str) -> None:
    overlay = CONFIGS / f"sweeps/ts38/parent_lora_sweep_lr_{token}.yaml"
    cfg = load_config(CONFIGS / LORA_PARENT_CONFIG, overlay)
    assert isinstance(cfg["train"]["lr"], float)
    assert cfg["train"]["lr"] == LORA_LR_TOKENS[token]
    assert cfg["train"]["max_steps"] == 8000


def test_ts38_lora_sweep_run_ids_distinct_and_out_of_family() -> None:
    run_ids = [
        yaml.safe_load((CONFIGS / f"sweeps/ts38/parent_lora_sweep_lr_{tok}.yaml").read_text())[
            "run_id"
        ]
        for tok in LORA_LR_TOKENS
    ]
    assert len(set(run_ids)) == len(run_ids)
    for rid in run_ids:
        assert not _TS38_FAMILY_REGEX.match(rid), rid
        assert rid != "evt-ts38-pretaught-parent"
        assert not _TS38_LADDER_SKIP_PATTERN.match(rid), rid


@pytest.mark.parametrize("token", sorted(LORA_LR_TOKENS))
def test_ts38_lora_full_overlay_shape(token: str) -> None:
    path = CONFIGS / f"sweeps/ts38/parent_lora_lr_{token}.yaml"
    raw = yaml.safe_load(path.read_text())
    assert set(raw.keys()) == {"train"}
    assert set(raw["train"].keys()) == {"lr"}
    assert isinstance(raw["train"]["lr"], float)
    assert raw["train"]["lr"] == LORA_LR_TOKENS[token]

    cfg = load_config(CONFIGS / LORA_PARENT_CONFIG, path)
    assert cfg["run_id"] == "evt-ts38-pretaught-parent"
    assert cfg["train"]["max_steps"] == 160000


def test_ts38_lora_parent_opts_into_lora_via_own_lora_block() -> None:
    """Mirror of test_llama_fig2nl_installer_opts_into_lora_via_own_lora_block
    and test_ts38_parent_has_no_own_lora_block, for the LoRA parent family:
    the config (with or without a sweep override) must opt in, and the
    retired full-FT parent must still opt out."""
    train_sft = load("train_sft")
    lora_path = CONFIGS / LORA_PARENT_CONFIG

    lora_cfg = train_sft.own_lora_block(load_config(lora_path, None), lora_path, None)
    assert lora_cfg is not None
    assert (lora_cfg["r"], lora_cfg["alpha"]) == (128, 32)
    assert lora_cfg["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    sweep_path = CONFIGS / "sweeps/ts38/parent_lora_sweep_lr_1e-3.yaml"
    lora_cfg_sweep = train_sft.own_lora_block(
        load_config(lora_path, sweep_path), lora_path, sweep_path
    )
    assert lora_cfg_sweep is not None
    assert (lora_cfg_sweep["r"], lora_cfg_sweep["alpha"]) == (128, 32)

    full_ft_path = CONFIGS / "ts38_pretaught_parent.yaml"
    assert train_sft.own_lora_block(load_config(full_ft_path, None), full_ft_path, None) is None


def test_ts38_lora_parent_builds_a_lora_manifest() -> None:
    """No committed config before this one has ever exercised
    manifest_fields's LoRA branch (lora_cfg["r"]/["alpha"]/["target_modules"]/
    ["dropout"]) — every existing LoRA-opted-in FULL_FT config
    (run9_llama1b_inst.yaml, llama_fig2nl_installer.yaml) is tested with
    lora_cfg=None. manifest_fields raises a bare KeyError deep inside
    register_run (module docstring) on a real launch, minutes into a paid
    box; this is the check that catches it before spend, mirroring
    test_ts38_pretaught_parent_builds_a_full_ft_manifest for the LoRA side."""
    train_sft = load("train_sft")
    lora_path = CONFIGS / LORA_PARENT_CONFIG
    winner_path = CONFIGS / "sweeps/ts38/parent_lora_lr_1e-3.yaml"
    cfg = load_config(lora_path, winner_path)
    lora_cfg = train_sft.own_lora_block(cfg, lora_path, winner_path)
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
    assert manifest["training"]["lora"]["rank"] == 128
    assert manifest["training"]["lora"]["alpha"] == 32


# =============================================================================
# ts38 LoRA-parent capability-vs-retention PROBE (decisions.md 2026-08-15,
# "LoRA parent HALT" entry): ONE deterministic replay to 24000 steps with
# snapshots every 1000 from 10000, superseding the four separate
# parent_lora_probe_lr3e-4_s{14000,16000,18000,20000}.yaml cut-and-replay
# overlays (deleted). Its own constant, NOT folded into LORA_FILES above —
# that list's length is asserted exactly (test_ts38_lora_parent_family_files_
# exist) and is indexed positionally elsewhere, so a probe overlay riding
# along would silently change what that count guards.
# =============================================================================

PROBE_OVERLAY = "sweeps/ts38/parent_lora_probe_lr3e-4.yaml"
PROBE_SNAPSHOT_STEPS = list(range(10000, 24001, 1000))
_RETIRED_PROBE_OVERLAYS = [
    f"sweeps/ts38/parent_lora_probe_lr3e-4_s{s}.yaml" for s in (14000, 16000, 18000, 20000)
]


def test_ts38_lora_probe_overlay_file_exists() -> None:
    assert (CONFIGS / PROBE_OVERLAY).is_file()


def test_ts38_lora_probe_retired_overlays_deleted() -> None:
    # The four single-horizon overlays this probe design replaces must not
    # still exist — a leftover file here would be relaunched by hand and
    # silently duplicate the single-replay design's GPU spend.
    for rel in _RETIRED_PROBE_OVERLAYS:
        assert not (CONFIGS / rel).is_file(), rel


def test_ts38_lora_probe_overlay_merged_values() -> None:
    cfg = load_config(CONFIGS / LORA_PARENT_CONFIG, CONFIGS / PROBE_OVERLAY)
    assert cfg["run_id"] == "evt-ts38-parent-loraprobe-lr3e-4"
    assert isinstance(cfg["train"]["lr"], float)
    assert cfg["train"]["lr"] == 3.0e-4
    assert cfg["train"]["max_steps"] == 24000
    assert cfg["train"]["snapshot_steps"] == PROBE_SNAPSHOT_STEPS


def test_ts38_lora_probe_overlay_builds_a_lora_manifest() -> None:
    # Mirror of test_ts38_lora_parent_builds_a_lora_manifest for the probe
    # overlay: manifest_fields must not choke on this run's shape before any
    # GPU spend (the KeyError-deep-in-register_run failure mode the module
    # docstring documents).
    train_sft = load("train_sft")
    lora_path = CONFIGS / LORA_PARENT_CONFIG
    overlay_path = CONFIGS / PROBE_OVERLAY
    cfg = load_config(lora_path, overlay_path)
    lora_cfg = train_sft.own_lora_block(cfg, lora_path, overlay_path)
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
    assert manifest["snapshot_steps"] == PROBE_SNAPSHOT_STEPS


def test_ts38_lora_probe_run_id_out_of_family_and_ladder() -> None:
    run_id = yaml.safe_load((CONFIGS / PROBE_OVERLAY).read_text())["run_id"]
    assert not _TS38_FAMILY_REGEX.match(run_id), run_id
    assert not _TS38_LADDER_SKIP_PATTERN.match(run_id), run_id


# =============================================================================
# ts38mw target-stage arm ("pretaught-mw"; docs/ts38mw-target-experiment-
# handoff.md, EXPERIMENTS.md §6.14/§6.15): the GO-B multiwrap-installed
# parent (evt-ts38mw-parent-probe-lr3e-4, step 28000) -> LoRA target on
# D_algo_bare, identical recipe to ts38_pretaught.yaml except theta0. The
# base arm is REUSED (evt-ts38-base-n<size>, tested above) — only this arm
# is new. Mirrors the ts38 section's style/helpers above
# (_leaf_diffs/load_config/CONFIGS, and the pinned-values/anchor/regime/
# manifest test shapes); kept as its own section because the parent is
# deliberately ungated (parent_required_gates: []), a protocol deviation
# the ts38 pretaught arm does not share.
# =============================================================================

# (relative path, n_examples, eval_every, max_steps, min_steps=ceil(n/128)) —
# identical numbers to TS38_OVERLAYS' pretaught rows above (same recipe,
# different theta0); also cross-checked against the sibling
# ts38_pretaught_n<size>.yaml overlay's own merged values below, not just
# these literals.
TS38MW_OVERLAYS = [
    ("sweeps/ts38/ts38mw_pretaught_n1000.yaml", 1000, 5, 1000, 8),
    ("sweeps/ts38/ts38mw_pretaught_n4642.yaml", 4642, 5, 1000, 37),
    ("sweeps/ts38/ts38mw_pretaught_n21544.yaml", 21544, 25, 5000, 169),
    ("sweeps/ts38/ts38mw_pretaught_n100000.yaml", 100000, 125, 15625, 782),
    ("sweeps/ts38/ts38mw_pretaught_n316228.yaml", 316228, 375, 50000, 2471),
]
TS38MW_OVERLAY_PATHS = [row[0] for row in TS38MW_OVERLAYS]
TS38MW_BASE_FILE = "ts38mw_pretaught.yaml"
TS38MW_FILES = [TS38MW_BASE_FILE, *TS38MW_OVERLAY_PATHS]
TS38MW_OVERLAY_ANCHORS = [
    ("sweeps/ts38/ts38mw_pretaught_n1000.yaml", "evt-ts38-base-n1000"),
    ("sweeps/ts38/ts38mw_pretaught_n4642.yaml", "evt-ts38-base-n4642"),
    ("sweeps/ts38/ts38mw_pretaught_n21544.yaml", "evt-ts38-base-n21544"),
    ("sweeps/ts38/ts38mw_pretaught_n100000.yaml", "evt-ts38-base-n100000"),
    ("sweeps/ts38/ts38mw_pretaught_n316228.yaml", "evt-ts38-base-n316228"),
]

# The sibling arm's recipe must be identical (same as-A allowed-diff set the
# ts38 base/pretaught pair already carries); the ts38mw-vs-pretaught pair is
# strictly narrower, since theta0 is the ONLY thing this arm changes relative
# to its sibling (arm and match_data_order_with placeholders are unchanged).
_TS38MW_VS_PRETAUGHT_ALLOWED_DIFF_PATHS = {
    "run_id",
    "experiment.parent_run_id",
    "experiment.parent_required_gates",
}

_TS38MW_RUN_ID_PATTERN = re.compile(r"^evt-ts38mw-pretaught(-n\d+)?$")


def _ts38mw_sibling_pretaught_path(mw_path: str) -> str:
    """The same-size ts38_pretaught_n<size>.yaml overlay for a ts38mw overlay path."""
    return mw_path.replace("ts38mw_pretaught", "ts38_pretaught", 1)


def test_ts38mw_family_files_exist() -> None:
    assert len(TS38MW_FILES) == 6
    for rel in TS38MW_FILES:
        assert (CONFIGS / rel).is_file(), rel


@pytest.mark.parametrize("path", TS38MW_FILES)
def test_ts38mw_run_ids_match_pattern_and_out_of_ts38_family(path: str) -> None:
    raw = yaml.safe_load((CONFIGS / path).read_text())
    run_id = raw["run_id"]
    assert _TS38MW_RUN_ID_PATTERN.match(run_id), run_id
    # Must NOT match the ts38 family regex the analysis tooling
    # (edl_converged_val_floor.py FAMILIES["ts38"]) uses to find base/pretaught
    # runs — a collision here would silently mix this arm into that curve.
    assert not _TS38_FAMILY_REGEX.match(run_id), run_id


@pytest.mark.parametrize(
    ("path", "n_examples", "eval_every", "max_steps", "min_steps"), TS38MW_OVERLAYS
)
def test_ts38mw_overlay_pinned_values(
    path: str, n_examples: int, eval_every: int, max_steps: int, min_steps: int
) -> None:
    raw = yaml.safe_load((CONFIGS / path).read_text())
    assert raw["data"]["n_examples"] == n_examples
    assert raw["train"]["eval_every"] == eval_every
    assert raw["train"]["max_steps"] == max_steps
    assert raw["train"]["stopping"]["min_steps"] == min_steps
    assert max_steps >= min_steps


@pytest.mark.parametrize("path", TS38MW_OVERLAY_PATHS)
def test_ts38mw_overlay_matches_sibling_pretaught_overlay(path: str) -> None:
    """Same n_examples/eval_every/max_steps/min_steps as the same-size
    ts38_pretaught_n<size>.yaml overlay — the two arms share one recipe, only
    theta0 differs. Compares MERGED configs (not just the overlay's own raw
    keys), since either side could in principle carry the value via its base
    file's placeholder instead of its own override."""
    mw_cfg = load_config(CONFIGS / TS38MW_BASE_FILE, CONFIGS / path)
    sibling_path = _ts38mw_sibling_pretaught_path(path)
    sibling_cfg = load_config(CONFIGS / "ts38_pretaught.yaml", CONFIGS / sibling_path)
    for field in ("n_examples",):
        assert mw_cfg["data"][field] == sibling_cfg["data"][field], field
    for field in ("eval_every", "max_steps"):
        assert mw_cfg["train"][field] == sibling_cfg["train"][field], field
    assert mw_cfg["train"]["stopping"]["min_steps"] == sibling_cfg["train"]["stopping"]["min_steps"]
    assert mw_cfg["train"]["max_steps"] >= mw_cfg["train"]["stopping"]["min_steps"]


@pytest.mark.parametrize(("path", "anchor"), TS38MW_OVERLAY_ANCHORS)
def test_ts38mw_overlay_match_data_order_with_is_base_arm(path: str, anchor: str) -> None:
    raw = yaml.safe_load((CONFIGS / path).read_text())
    assert raw["experiment"]["match_data_order_with"] == anchor
    # The anchor is the reused BASE arm, never the sibling pretaught arm —
    # a G7 anchor pointed at the wrong run would silently compare this arm's
    # data order against the wrong stream.
    assert not anchor.startswith("evt-ts38-pretaught-")


def test_ts38mw_vs_ts38_base_allowed_diffs() -> None:
    base = yaml.safe_load((CONFIGS / "ts38_base.yaml").read_text())
    mw = yaml.safe_load((CONFIGS / TS38MW_BASE_FILE).read_text())
    diffs = _leaf_diffs(base, mw)
    unexplained = diffs - _TS38_ALLOWED_DIFF_PATHS
    assert not unexplained, f"ts38mw vs ts38_base differ in unexpected fields: {unexplained}"


def test_ts38mw_vs_ts38_pretaught_differs_only_in_theta0_fields() -> None:
    """The important invariant: ts38mw_pretaught.yaml is the sibling arm's
    IDENTICAL recipe, changed only where theta0 (and its direct
    consequences) require it — a strictly narrower diff set than the
    ts38-family base/pretaught pair, since `arm` and
    `match_data_order_with`'s placeholder are unchanged here."""
    pretaught = yaml.safe_load((CONFIGS / "ts38_pretaught.yaml").read_text())
    mw = yaml.safe_load((CONFIGS / TS38MW_BASE_FILE).read_text())
    diffs = _leaf_diffs(pretaught, mw)
    assert diffs == _TS38MW_VS_PRETAUGHT_ALLOWED_DIFF_PATHS, diffs
    assert pretaught["run_id"] != mw["run_id"]
    assert pretaught["experiment"]["parent_run_id"] != mw["experiment"]["parent_run_id"]
    assert (
        pretaught["experiment"]["parent_required_gates"]
        != mw["experiment"]["parent_required_gates"]
    )
    # And the fields that must NOT differ (same recipe) actually don't.
    assert pretaught["experiment"]["arm"] == mw["experiment"]["arm"]
    assert pretaught["experiment"]["match_data_order_with"] is None
    assert mw["experiment"]["match_data_order_with"] is None


def test_ts38mw_merged_config_values() -> None:
    cfg = load_config(CONFIGS / TS38MW_BASE_FILE, None)
    base_cfg = load_config(CONFIGS / "ts38_base.yaml", None)
    assert cfg["experiment"]["parent_required_gates"] == []
    assert cfg["experiment"]["parent_run_id"] == "evt-ts38mw-parent-probe-lr3e-4"
    assert cfg["experiment"]["require_full_epoch1"] is True
    assert cfg["lora"]["r"] == 128
    assert cfg["lora"]["alpha"] == 32
    assert cfg["train"]["lr"] == 1.0e-3
    assert cfg["train"]["lr"] == base_cfg["train"]["lr"]
    assert cfg["train"]["snapshots"]["n"] == 0
    assert cfg["data"]["order_hash"] == base_cfg["data"]["order_hash"]
    assert cfg["data"]["eval_order_hash"] == base_cfg["data"]["eval_order_hash"]


def test_ts38mw_base_builds_a_manifest_with_correct_regime() -> None:
    """Mirror of test_ts38_target_base_builds_a_manifest_with_correct_regime:
    `arm: A` must map to `regime: "elicit"` (train_target.py's ARM_REGIME),
    same as ts38_pretaught.yaml."""
    train_target = load("train_target")
    cfg = load_config(CONFIGS / TS38MW_BASE_FILE, None)
    manifest = train_target.manifest_fields(
        cfg,
        n_train=1,
        n_trainable=1,
        epochs_total=1,
        schedule=[],
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="bf16",
    )
    assert manifest["regime"] == "elicit"


@pytest.mark.parametrize("path", TS38MW_OVERLAY_PATHS)
def test_ts38mw_overlay_builds_a_manifest(path: str) -> None:
    train_target = load("train_target")
    cfg = load_config(CONFIGS / TS38MW_BASE_FILE, CONFIGS / path)
    train_target.manifest_fields(
        cfg,
        n_train=1,
        n_trainable=1,
        epochs_total=1,
        schedule=[],
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="bf16",
    )


# =============================================================================
# ts38pf target-stage arm ("pre-teach-format"; EXPERIMENTS §6.16, decisions.md
# 2026-08-15 "ts38pf pre-registration"): a NEW format-only parent (built by
# THIS family, unlike ts38mw's already-built parent — operator-notation,
# randomly-permuted labels, paper App. E.1.2) -> LoRA target on D_algo_bare,
# identical recipe to ts38_pretaught.yaml/ts38mw_pretaught.yaml except theta0.
# The base arm is REUSED (evt-ts38-base-n<size>, tested above) — only the
# parent and the new target arm are new. Mirrors the ts38mw section's
# style/helpers directly above.
# =============================================================================

# (relative path, n_examples, eval_every, max_steps, min_steps=ceil(n/128)) —
# identical numbers to TS38MW_OVERLAYS (same grid, same recipe, different
# theta0); cross-checked against the sibling ts38_pretaught_n<size>.yaml
# overlay's own merged values below too, not just these literals.
TS38PF_OVERLAYS = [
    ("sweeps/ts38/ts38pf_preteachfmt_n1000.yaml", 1000, 5, 1000, 8),
    ("sweeps/ts38/ts38pf_preteachfmt_n4642.yaml", 4642, 5, 1000, 37),
    ("sweeps/ts38/ts38pf_preteachfmt_n21544.yaml", 21544, 25, 5000, 169),
    ("sweeps/ts38/ts38pf_preteachfmt_n100000.yaml", 100000, 125, 15625, 782),
    ("sweeps/ts38/ts38pf_preteachfmt_n316228.yaml", 316228, 375, 50000, 2471),
]
TS38PF_OVERLAY_PATHS = [row[0] for row in TS38PF_OVERLAYS]
TS38PF_BASE_FILE = "ts38pf_preteachfmt.yaml"
TS38PF_PARENT_FILE = "ts38_preteachfmt_parent.yaml"
TS38PF_FILES = [TS38PF_BASE_FILE, TS38PF_PARENT_FILE, *TS38PF_OVERLAY_PATHS]
TS38PF_OVERLAY_ANCHORS = [
    ("sweeps/ts38/ts38pf_preteachfmt_n1000.yaml", "evt-ts38-base-n1000"),
    ("sweeps/ts38/ts38pf_preteachfmt_n4642.yaml", "evt-ts38-base-n4642"),
    ("sweeps/ts38/ts38pf_preteachfmt_n21544.yaml", "evt-ts38-base-n21544"),
    ("sweeps/ts38/ts38pf_preteachfmt_n100000.yaml", "evt-ts38-base-n100000"),
    ("sweeps/ts38/ts38pf_preteachfmt_n316228.yaml", "evt-ts38-base-n316228"),
]

# Same shape as _TS38MW_VS_PRETAUGHT_ALLOWED_DIFF_PATHS: theta0 (and its
# direct consequences) is the ONLY thing this arm changes relative to
# ts38_pretaught.yaml.
_TS38PF_VS_PRETAUGHT_ALLOWED_DIFF_PATHS = {
    "run_id",
    "experiment.parent_run_id",
    "experiment.parent_required_gates",
}

_TS38PF_RUN_ID_PATTERN = re.compile(r"^evt-ts38pf-preteachfmt(-n\d+)?$")
# The parent's own run id doesn't fit the above ("-parent", not "-n<size>");
# checked separately below.
_TS38PF_PARENT_RUN_ID = "evt-ts38pf-preteachfmt-parent"


def _ts38pf_sibling_pretaught_path(pf_path: str) -> str:
    """The same-size ts38_pretaught_n<size>.yaml overlay for a ts38pf overlay path."""
    return pf_path.replace("ts38pf_preteachfmt", "ts38_pretaught", 1)


def test_ts38pf_family_files_exist() -> None:
    assert len(TS38PF_FILES) == 7
    for rel in TS38PF_FILES:
        assert (CONFIGS / rel).is_file(), rel


@pytest.mark.parametrize("path", [TS38PF_BASE_FILE, *TS38PF_OVERLAY_PATHS])
def test_ts38pf_run_ids_match_pattern_and_out_of_other_families(path: str) -> None:
    raw = yaml.safe_load((CONFIGS / path).read_text())
    run_id = raw["run_id"]
    assert _TS38PF_RUN_ID_PATTERN.match(run_id), run_id
    # Must NOT match the ts38 or ts38mw family regexes the analysis tooling
    # (edl_converged_val_floor.py's FAMILIES["ts38"]/["ts38mw"]) uses to find
    # base/pretaught runs — a collision here would silently mix this arm
    # into either curve. (The full cross-check against the analysis
    # script's actual regex objects lives in
    # test_edl_converged_val_floor_families.py's _MATRIX.)
    assert not _TS38_FAMILY_REGEX.match(run_id), run_id
    assert not _TS38MW_RUN_ID_PATTERN.match(run_id), run_id


def test_ts38pf_parent_run_id() -> None:
    raw = yaml.safe_load((CONFIGS / TS38PF_PARENT_FILE).read_text())
    assert raw["run_id"] == _TS38PF_PARENT_RUN_ID
    assert not _TS38_FAMILY_REGEX.match(raw["run_id"])
    assert not _TS38MW_RUN_ID_PATTERN.match(raw["run_id"])
    assert not _TS38PF_RUN_ID_PATTERN.match(raw["run_id"])  # "-parent", not "-n<size>"


@pytest.mark.parametrize(
    ("path", "n_examples", "eval_every", "max_steps", "min_steps"), TS38PF_OVERLAYS
)
def test_ts38pf_overlay_pinned_values(
    path: str, n_examples: int, eval_every: int, max_steps: int, min_steps: int
) -> None:
    raw = yaml.safe_load((CONFIGS / path).read_text())
    assert raw["data"]["n_examples"] == n_examples
    assert raw["train"]["eval_every"] == eval_every
    assert raw["train"]["max_steps"] == max_steps
    assert raw["train"]["stopping"]["min_steps"] == min_steps
    assert max_steps >= min_steps


@pytest.mark.parametrize("path", TS38PF_OVERLAY_PATHS)
def test_ts38pf_overlay_matches_sibling_pretaught_overlay(path: str) -> None:
    """Same n_examples/eval_every/max_steps/min_steps as the same-size
    ts38_pretaught_n<size>.yaml overlay — the target-stage arms all share one
    recipe, only theta0 differs."""
    pf_cfg = load_config(CONFIGS / TS38PF_BASE_FILE, CONFIGS / path)
    sibling_path = _ts38pf_sibling_pretaught_path(path)
    sibling_cfg = load_config(CONFIGS / "ts38_pretaught.yaml", CONFIGS / sibling_path)
    for field in ("n_examples",):
        assert pf_cfg["data"][field] == sibling_cfg["data"][field], field
    for field in ("eval_every", "max_steps"):
        assert pf_cfg["train"][field] == sibling_cfg["train"][field], field
    assert pf_cfg["train"]["stopping"]["min_steps"] == sibling_cfg["train"]["stopping"]["min_steps"]
    assert pf_cfg["train"]["max_steps"] >= pf_cfg["train"]["stopping"]["min_steps"]


@pytest.mark.parametrize(("path", "anchor"), TS38PF_OVERLAY_ANCHORS)
def test_ts38pf_overlay_match_data_order_with_is_base_arm(path: str, anchor: str) -> None:
    raw = yaml.safe_load((CONFIGS / path).read_text())
    assert raw["experiment"]["match_data_order_with"] == anchor
    assert not anchor.startswith("evt-ts38-pretaught-")


def test_ts38pf_vs_ts38_base_allowed_diffs() -> None:
    base = yaml.safe_load((CONFIGS / "ts38_base.yaml").read_text())
    pf = yaml.safe_load((CONFIGS / TS38PF_BASE_FILE).read_text())
    diffs = _leaf_diffs(base, pf)
    unexplained = diffs - _TS38_ALLOWED_DIFF_PATHS
    assert not unexplained, f"ts38pf vs ts38_base differ in unexpected fields: {unexplained}"


def test_ts38pf_vs_ts38_pretaught_differs_only_in_theta0_fields() -> None:
    pretaught = yaml.safe_load((CONFIGS / "ts38_pretaught.yaml").read_text())
    pf = yaml.safe_load((CONFIGS / TS38PF_BASE_FILE).read_text())
    diffs = _leaf_diffs(pretaught, pf)
    assert diffs == _TS38PF_VS_PRETAUGHT_ALLOWED_DIFF_PATHS, diffs
    assert pretaught["run_id"] != pf["run_id"]
    assert pretaught["experiment"]["parent_run_id"] != pf["experiment"]["parent_run_id"]
    assert (
        pretaught["experiment"]["parent_required_gates"]
        != pf["experiment"]["parent_required_gates"]
    )
    assert pretaught["experiment"]["arm"] == pf["experiment"]["arm"]
    assert pretaught["experiment"]["match_data_order_with"] is None
    assert pf["experiment"]["match_data_order_with"] is None


def test_ts38pf_merged_config_values() -> None:
    cfg = load_config(CONFIGS / TS38PF_BASE_FILE, None)
    base_cfg = load_config(CONFIGS / "ts38_base.yaml", None)
    assert cfg["experiment"]["parent_required_gates"] == []
    assert cfg["experiment"]["parent_run_id"] == _TS38PF_PARENT_RUN_ID
    assert cfg["experiment"]["require_full_epoch1"] is True
    assert cfg["lora"]["r"] == 128
    assert cfg["lora"]["alpha"] == 32
    assert cfg["train"]["lr"] == 1.0e-3
    assert cfg["train"]["lr"] == base_cfg["train"]["lr"]
    assert cfg["train"]["snapshots"]["n"] == 0
    # Target data is UNCHANGED D_algo_bare — only the parent differs from
    # base_cfg, not the target task.
    assert cfg["data"]["order_hash"] == base_cfg["data"]["order_hash"]
    assert cfg["data"]["eval_order_hash"] == base_cfg["data"]["eval_order_hash"]


def test_ts38pf_base_builds_a_manifest_with_correct_regime() -> None:
    train_target = load("train_target")
    cfg = load_config(CONFIGS / TS38PF_BASE_FILE, None)
    manifest = train_target.manifest_fields(
        cfg,
        n_train=1,
        n_trainable=1,
        epochs_total=1,
        schedule=[],
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="bf16",
    )
    assert manifest["regime"] == "elicit"


@pytest.mark.parametrize("path", TS38PF_OVERLAY_PATHS)
def test_ts38pf_overlay_builds_a_manifest(path: str) -> None:
    train_target = load("train_target")
    cfg = load_config(CONFIGS / TS38PF_BASE_FILE, CONFIGS / path)
    train_target.manifest_fields(
        cfg,
        n_train=1,
        n_trainable=1,
        epochs_total=1,
        schedule=[],
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="bf16",
    )


# ---- the format-only parent build config (NEW vs. ts38mw — that family
# pulled an already-built parent; this one has to validate its OWN parent
# config, mirroring the ts38 LoRA-parent section's essentials) --------------


def test_ts38pf_preteachfmt_parent_own_yaml_declares_lora() -> None:
    # The un-merged file, not the deep-merged cfg — this is what
    # own_lora_block actually reads to opt in.
    raw = yaml.safe_load((CONFIGS / TS38PF_PARENT_FILE).read_text())
    assert raw["lora"]["r"] == 128
    assert raw["lora"]["alpha"] == 32


def test_ts38pf_preteachfmt_parent_opts_into_lora_via_own_lora_block() -> None:
    train_sft = load("train_sft")
    path = CONFIGS / TS38PF_PARENT_FILE
    lora_cfg = train_sft.own_lora_block(load_config(path, None), path, None)
    assert lora_cfg is not None
    assert (lora_cfg["r"], lora_cfg["alpha"]) == (128, 32)


def test_ts38pf_preteachfmt_parent_min_steps_is_one_full_epoch() -> None:
    """Advisor-flagged fix: min_steps must be pinned to exactly one full
    epoch under train_sft.py's OWN step counting (n_val = round(val_fraction
    * n) via geode.train.packing.split_indices, then floor-division/
    drop-last batching — NOT train_target.py's ceil/no-drop-last
    convention), not left near-zero. On permuted labels there is no
    learnable mapping, so val loss can plateau from format acquisition
    alone in a handful of evals; eps/k must not be allowed to declare
    "converged" before the model has seen the full set once."""
    cfg = load_config(CONFIGS / TS38PF_PARENT_FILE, None)
    d, t = cfg["data"], cfg["train"]
    n_examples = 21544
    n_val = max(1, min(round(d["val_fraction"] * n_examples), n_examples - 1))
    n_train = n_examples - n_val
    steps_per_epoch = n_train // t["batch_size"]
    assert t["stopping"]["min_steps"] == steps_per_epoch == 167


def test_ts38pf_preteachfmt_parent_max_steps_covers_min_steps() -> None:
    cfg = load_config(CONFIGS / TS38PF_PARENT_FILE, None)
    t = cfg["train"]
    assert t["max_steps"] >= t["stopping"]["min_steps"] + t["stopping"]["k"] * t["eval_every"]
    assert t["max_steps"] >= 20 * t["stopping"]["min_steps"]  # >=20-epoch ceiling, never binds


def test_ts38pf_preteachfmt_parent_data_hash_pinned() -> None:
    cfg = load_config(CONFIGS / TS38PF_PARENT_FILE, None)
    assert cfg["data"]["file"] == "D_preteachfmt.parquet"
    assert (
        cfg["data"]["order_hash"]
        == "5b0b19a4c47375a4ada17cb1ee21292475b6ecaed22b2ef07aa560cf557b1bc1"
    )


def test_ts38pf_preteachfmt_parent_builds_a_lora_manifest() -> None:
    train_sft = load("train_sft")
    path = CONFIGS / TS38PF_PARENT_FILE
    cfg = load_config(path, None)
    lora_cfg = train_sft.own_lora_block(cfg, path, None)
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
    assert manifest["training"]["lora"]["rank"] == 128
    assert manifest["training"]["lora"]["alpha"] == 32


# =============================================================================
# ts38grid unified 3-arm grid extension (EXPERIMENTS.md §6.17, 2026-08-16):
# 8 NEW dataset sizes per arm — a small-n bracket {128, 256, 512} and a
# densification {2154, 10000, 46416, 146780, 215443} — on top of the 5 sizes
# already covered by TS38_OVERLAYS/TS38MW_OVERLAYS/TS38PF_OVERLAYS above
# ({1000, 4642, 21544, 100000, 316228}). Same three arms, same base configs
# (ts38_base.yaml/ts38mw_pretaught.yaml/ts38pf_preteachfmt.yaml), same
# run_id patterns (_TS38_RUN_ID_PATTERN/_TS38MW_RUN_ID_PATTERN/
# _TS38PF_RUN_ID_PATTERN) defined above. Purely additive: this section does
# not touch any TS38/TS38MW/TS38PF list, constant, or assert above it.
# =============================================================================

TS38GRID_SIZES = [128, 256, 512, 2154, 10000, 46416, 146780, 215443]

# n -> (eval_every, max_steps, min_steps=ceil(n/128)). Small-n bracket
# {128,256,512} shares the n=1000 overlays' cadence/ceiling (1 step/epoch,
# no matching-size fig2nl3 point to copy from). Densification
# {2154,10000,46416} copied verbatim from the matching-size
# llama_fig2nl3_noinst_n<size>.yaml overlay (20-epoch ceiling never binds
# there). {146780,215443} start from that same overlay's eval_every, but its
# max_steps was BELOW 20 epochs (146780: 14000 < 1147*20=22940; 215443:
# 20208 < 1684*20=33680), so max_steps is raised to 23000/34000 — the same
# >=20-epoch ceilings-must-never-bind rule commit 6b735f1 applied to
# ts38_base_n100000.yaml (->15625) and ts38_base_n316228.yaml (->50000).
TS38GRID_PINNED = {
    128: (5, 1000, 1),
    256: (5, 1000, 2),
    512: (5, 1000, 4),
    2154: (5, 1000, 17),
    10000: (10, 2000, 79),
    46416: (55, 11000, 363),
    146780: (175, 23000, 1147),
    215443: (250, 34000, 1684),
}

# (arm, n, eval_every, max_steps, min_steps) for every one of the 24 files.
TS38GRID_OVERLAYS = [
    (arm, n, *TS38GRID_PINNED[n]) for arm in ("ts38", "ts38mw", "ts38pf") for n in TS38GRID_SIZES
]

_TS38GRID_PATH_TMPL = {
    "ts38": "sweeps/ts38/ts38_base_n{n}.yaml",
    "ts38mw": "sweeps/ts38/ts38mw_pretaught_n{n}.yaml",
    "ts38pf": "sweeps/ts38/ts38pf_preteachfmt_n{n}.yaml",
}
_TS38GRID_BASE_FILE = {
    "ts38": "ts38_base.yaml",
    "ts38mw": "ts38mw_pretaught.yaml",
    "ts38pf": "ts38pf_preteachfmt.yaml",
}
_TS38GRID_RUN_ID_TMPL = {
    "ts38": "evt-ts38-base-n{n}",
    "ts38mw": "evt-ts38mw-pretaught-n{n}",
    "ts38pf": "evt-ts38pf-preteachfmt-n{n}",
}
# Same three patterns TS38_OVERLAYS/TS38MW_OVERLAYS/TS38PF_OVERLAYS already
# use above — a run id must match its OWN arm's pattern and no other arm's.
_TS38GRID_RUN_ID_PATTERN = {
    "ts38": _TS38_RUN_ID_PATTERN,
    "ts38mw": _TS38MW_RUN_ID_PATTERN,
    "ts38pf": _TS38PF_RUN_ID_PATTERN,
}
# train_target.py's ARM_REGIME (see the ARM-LETTER NOTE in ts38_base.yaml,
# read at the top of this file's ts38 section): base -> teach, mw/pf -> elicit.
_TS38GRID_REGIME = {"ts38": "teach", "ts38mw": "elicit", "ts38pf": "elicit"}


@pytest.mark.parametrize(("arm", "n", "eval_every", "max_steps", "min_steps"), TS38GRID_OVERLAYS)
def test_ts38grid_overlay_file_exists(
    arm: str, n: int, eval_every: int, max_steps: int, min_steps: int
) -> None:
    path = _TS38GRID_PATH_TMPL[arm].format(n=n)
    assert (CONFIGS / path).is_file(), path


@pytest.mark.parametrize(("arm", "n", "eval_every", "max_steps", "min_steps"), TS38GRID_OVERLAYS)
def test_ts38grid_run_id_matches_own_pattern_only(
    arm: str, n: int, eval_every: int, max_steps: int, min_steps: int
) -> None:
    path = _TS38GRID_PATH_TMPL[arm].format(n=n)
    raw = yaml.safe_load((CONFIGS / path).read_text())
    run_id = raw["run_id"]
    assert run_id == _TS38GRID_RUN_ID_TMPL[arm].format(n=n)
    assert _TS38GRID_RUN_ID_PATTERN[arm].match(run_id), run_id
    for other_arm, pattern in _TS38GRID_RUN_ID_PATTERN.items():
        if other_arm != arm:
            assert not pattern.match(run_id), (arm, other_arm, run_id)


@pytest.mark.parametrize(("arm", "n", "eval_every", "max_steps", "min_steps"), TS38GRID_OVERLAYS)
def test_ts38grid_overlay_pinned_values(
    arm: str, n: int, eval_every: int, max_steps: int, min_steps: int
) -> None:
    path = _TS38GRID_PATH_TMPL[arm].format(n=n)
    raw = yaml.safe_load((CONFIGS / path).read_text())
    assert raw["data"]["n_examples"] == n
    assert raw["train"]["eval_every"] == eval_every
    assert raw["train"]["max_steps"] == max_steps
    assert raw["train"]["stopping"]["min_steps"] == min_steps


@pytest.mark.parametrize(("arm", "n", "eval_every", "max_steps", "min_steps"), TS38GRID_OVERLAYS)
def test_ts38grid_min_steps_equals_ceil_n_over_128(
    arm: str, n: int, eval_every: int, max_steps: int, min_steps: int
) -> None:
    """min_steps must be the COMPUTED ceil(n/128), not just a literal that
    happens to match the table — cross-checked against the same
    require_full_epoch1_launch_check the launcher itself calls
    (test_full_epoch1_guard.py), not a reimplementation of its arithmetic."""
    train_target = load("train_target")
    steps_per_epoch = math.ceil(n / 128)
    assert min_steps == steps_per_epoch
    train_target.require_full_epoch1_launch_check(True, min_steps, steps_per_epoch)


@pytest.mark.parametrize(("arm", "n", "eval_every", "max_steps", "min_steps"), TS38GRID_OVERLAYS)
def test_ts38grid_overlay_ceiling_rules(
    arm: str, n: int, eval_every: int, max_steps: int, min_steps: int
) -> None:
    """Two independent ceiling checks on the MERGED config: the >=20-epoch
    cost-ceiling floor, and the earliest-possible-stop arithmetic
    TS38_OVERLAYS' own pinned-values test uses above (max_steps must cover
    min_steps PLUS k more eval_every-spaced evaluations, else the eps/k rule
    never gets a chance to fire before the ceiling — see that test's
    comment for the full reasoning)."""
    path = _TS38GRID_PATH_TMPL[arm].format(n=n)
    cfg = load_config(CONFIGS / _TS38GRID_BASE_FILE[arm], CONFIGS / path)
    t = cfg["train"]
    assert t["max_steps"] >= 20 * t["stopping"]["min_steps"]
    assert t["max_steps"] >= t["stopping"]["min_steps"] + t["stopping"]["k"] * t["eval_every"]


@pytest.mark.parametrize("n", TS38GRID_SIZES)
def test_ts38grid_arms_agree_on_merged_schedule(n: int) -> None:
    """The three arms at a given size share one recipe — only theta0
    differs — so their MERGED n_examples/eval_every/max_steps/min_steps
    must be identical, not just their own overlay's raw literals."""
    merged = {}
    for arm in ("ts38", "ts38mw", "ts38pf"):
        path = _TS38GRID_PATH_TMPL[arm].format(n=n)
        cfg = load_config(CONFIGS / _TS38GRID_BASE_FILE[arm], CONFIGS / path)
        merged[arm] = (
            cfg["data"]["n_examples"],
            cfg["train"]["eval_every"],
            cfg["train"]["max_steps"],
            cfg["train"]["stopping"]["min_steps"],
        )
    assert merged["ts38"] == merged["ts38mw"] == merged["ts38pf"], merged


@pytest.mark.parametrize("n", TS38GRID_SIZES)
def test_ts38grid_mw_pf_match_data_order_with_base_anchor(n: int) -> None:
    anchor = f"evt-ts38-base-n{n}"
    for arm in ("ts38mw", "ts38pf"):
        path = _TS38GRID_PATH_TMPL[arm].format(n=n)
        raw = yaml.safe_load((CONFIGS / path).read_text())
        assert raw["experiment"]["match_data_order_with"] == anchor


@pytest.mark.parametrize("n", TS38GRID_SIZES)
def test_ts38grid_base_overlay_carries_no_match_data_order_with_key(n: int) -> None:
    """The base overlay IS the G7 anchor — its own raw yaml must not set
    match_data_order_with at all; the merged value stays ts38_base.yaml's
    own null placeholder."""
    path = _TS38GRID_PATH_TMPL["ts38"].format(n=n)
    raw = yaml.safe_load((CONFIGS / path).read_text())
    assert "match_data_order_with" not in raw.get("experiment", {})
    cfg = load_config(CONFIGS / "ts38_base.yaml", CONFIGS / path)
    assert cfg["experiment"]["match_data_order_with"] is None


@pytest.mark.parametrize(("arm", "n", "eval_every", "max_steps", "min_steps"), TS38GRID_OVERLAYS)
def test_ts38grid_overlay_builds_a_manifest_with_correct_regime(
    arm: str, n: int, eval_every: int, max_steps: int, min_steps: int
) -> None:
    train_target = load("train_target")
    path = _TS38GRID_PATH_TMPL[arm].format(n=n)
    cfg = load_config(CONFIGS / _TS38GRID_BASE_FILE[arm], CONFIGS / path)
    manifest = train_target.manifest_fields(
        cfg,
        n_train=1,
        n_trainable=1,
        epochs_total=1,
        schedule=[],
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="bf16",
    )
    assert manifest["regime"] == _TS38GRID_REGIME[arm]


# =============================================================================
# ts38pp — PAPER-PROTOCOL pre-teach (decisions.md 2026-08-16 "ts38pp
# pre-registration"; scripts/launch_ts38pp_family.sh): a NEW full-FT parent
# (evt-ts38pp-parent, one epoch over the NEW 4M-row D_target_4M.parquet,
# correct labels, paper App. E.2's literal recipe) -> LoRA target on
# D_algo_bare, identical recipe to ts38mw_pretaught.yaml/ts38pf_
# preteachfmt.yaml except theta0. The base arm is REUSED (evt-ts38-base-
# n<size>, tested above) — only the parent and the new target arm are new.
# Mirrors the ts38pf section's style/helpers directly above; the parent
# section additionally pins the ONE pre-registered run-until-convergence
# exception (min_steps == max_steps == one full epoch), since — unlike
# ts38pf's parent — this parent trains on a set 185x larger than any other
# ts38(*) config's train block, so its own step-count arithmetic earns a
# dedicated check rather than reuse of ts38pf's 21,544-row numbers.
# =============================================================================

# (relative path, n_examples, eval_every, max_steps, min_steps=ceil(n/128)) —
# identical numbers to TS38MW_OVERLAYS/TS38PF_OVERLAYS (same grid, same
# recipe, different theta0); cross-checked against the sibling
# ts38_pretaught_n<size>.yaml overlay's own merged values below too, not
# just these literals.
TS38PP_OVERLAYS = [
    ("sweeps/ts38/ts38pp_pretaught_n1000.yaml", 1000, 5, 1000, 8),
    ("sweeps/ts38/ts38pp_pretaught_n4642.yaml", 4642, 5, 1000, 37),
    ("sweeps/ts38/ts38pp_pretaught_n21544.yaml", 21544, 25, 5000, 169),
    ("sweeps/ts38/ts38pp_pretaught_n100000.yaml", 100000, 125, 15625, 782),
    ("sweeps/ts38/ts38pp_pretaught_n316228.yaml", 316228, 375, 50000, 2471),
]
TS38PP_OVERLAY_PATHS = [row[0] for row in TS38PP_OVERLAYS]
TS38PP_BASE_FILE = "ts38pp_pretaught.yaml"
TS38PP_PARENT_FILE = "ts38pp_parent.yaml"
TS38PP_FILES = [TS38PP_BASE_FILE, TS38PP_PARENT_FILE, *TS38PP_OVERLAY_PATHS]
TS38PP_OVERLAY_ANCHORS = [
    ("sweeps/ts38/ts38pp_pretaught_n1000.yaml", "evt-ts38-base-n1000"),
    ("sweeps/ts38/ts38pp_pretaught_n4642.yaml", "evt-ts38-base-n4642"),
    ("sweeps/ts38/ts38pp_pretaught_n21544.yaml", "evt-ts38-base-n21544"),
    ("sweeps/ts38/ts38pp_pretaught_n100000.yaml", "evt-ts38-base-n100000"),
    ("sweeps/ts38/ts38pp_pretaught_n316228.yaml", "evt-ts38-base-n316228"),
]

# Same shape as _TS38MW_VS_PRETAUGHT_ALLOWED_DIFF_PATHS / _TS38PF_VS_
# PRETAUGHT_ALLOWED_DIFF_PATHS: theta0 (and its direct consequences) is the
# ONLY thing this arm changes relative to ts38_pretaught.yaml.
_TS38PP_VS_PRETAUGHT_ALLOWED_DIFF_PATHS = {
    "run_id",
    "experiment.parent_run_id",
    "experiment.parent_required_gates",
}

_TS38PP_RUN_ID_PATTERN = re.compile(r"^evt-ts38pp-pretaught(-n\d+)?$")
# The parent's own run id doesn't fit the above ("-parent", not "-n<size>");
# checked separately below.
_TS38PP_PARENT_RUN_ID = "evt-ts38pp-parent"

# The one-epoch pin's own arithmetic (header of ts38pp_parent.yaml):
# n_val = round(0.005 * 4,000,000) = 20,000 (geode.train.packing.
# split_indices' round-then-clamp rule), n_train = 3,980,000,
# steps_per_epoch = n_train // 128 = 31,093 (floor/drop-last — train_sft.py's
# OWN convention, not train_target.py's ceil/no-drop-last one used by the
# overlays above).
TS38PP_PARENT_N_EXAMPLES = 4_000_000
TS38PP_PARENT_ONE_EPOCH_STEPS = 31093
TS38PP_PARENT_SNAPSHOT_STEPS = [7773, 15546, 23319]
TS38PP_PARENT_ORDER_HASH = "ba2d6efdd939f63e6da75420a93362fcf86a6adeaa66bf5b5cce01532fbec54c"


def _ts38pp_sibling_pretaught_path(pp_path: str) -> str:
    """The same-size ts38_pretaught_n<size>.yaml overlay for a ts38pp overlay path."""
    return pp_path.replace("ts38pp_pretaught", "ts38_pretaught", 1)


def test_ts38pp_family_files_exist() -> None:
    assert len(TS38PP_FILES) == 7
    for rel in TS38PP_FILES:
        assert (CONFIGS / rel).is_file(), rel


@pytest.mark.parametrize("path", [TS38PP_BASE_FILE, *TS38PP_OVERLAY_PATHS])
def test_ts38pp_run_ids_match_pattern_and_out_of_other_families(path: str) -> None:
    raw = yaml.safe_load((CONFIGS / path).read_text())
    run_id = raw["run_id"]
    assert _TS38PP_RUN_ID_PATTERN.match(run_id), run_id
    # Must NOT match any earlier ts38(*) family's regex — a collision here
    # would silently mix this arm into another family's curve
    # (edl_converged_val_floor.py's FAMILIES["ts38"/"ts38mw"/"ts38pf"]).
    assert not _TS38_FAMILY_REGEX.match(run_id), run_id
    assert not _TS38MW_RUN_ID_PATTERN.match(run_id), run_id
    assert not _TS38PF_RUN_ID_PATTERN.match(run_id), run_id


def test_ts38pp_run_id_pattern_rejects_base_arm() -> None:
    # ts38pp has no base arm of its own — it reuses evt-ts38-base-n<size>
    # verbatim, never trains an evt-ts38pp-base-n<size> run.
    assert not _TS38PP_RUN_ID_PATTERN.match("evt-ts38pp-base-n1000")


def test_ts38pp_run_id_pattern_does_not_match_other_families() -> None:
    other_ids = [
        "evt-ts38-pretaught-parent",
        "evt-ts38-base-n1000",
        "evt-ts38-pretaught-n1000",
        "evt-ts38mw-pretaught-n1000",
        "evt-ts38pf-preteachfmt-n1000",
        "evt-ts38pf-preteachfmt-parent",
    ]
    for rid in other_ids:
        assert not _TS38PP_RUN_ID_PATTERN.match(rid), rid


def test_ts38pp_parent_run_id() -> None:
    raw = yaml.safe_load((CONFIGS / TS38PP_PARENT_FILE).read_text())
    assert raw["run_id"] == _TS38PP_PARENT_RUN_ID
    assert not _TS38_FAMILY_REGEX.match(raw["run_id"])
    assert not _TS38MW_RUN_ID_PATTERN.match(raw["run_id"])
    assert not _TS38PF_RUN_ID_PATTERN.match(raw["run_id"])
    assert not _TS38PP_RUN_ID_PATTERN.match(raw["run_id"])  # "-parent", not "-n<size>"


@pytest.mark.parametrize(
    ("path", "n_examples", "eval_every", "max_steps", "min_steps"), TS38PP_OVERLAYS
)
def test_ts38pp_overlay_pinned_values(
    path: str, n_examples: int, eval_every: int, max_steps: int, min_steps: int
) -> None:
    raw = yaml.safe_load((CONFIGS / path).read_text())
    assert raw["data"]["n_examples"] == n_examples
    assert raw["train"]["eval_every"] == eval_every
    assert raw["train"]["max_steps"] == max_steps
    assert raw["train"]["stopping"]["min_steps"] == min_steps
    assert max_steps >= min_steps


@pytest.mark.parametrize("path", TS38PP_OVERLAY_PATHS)
def test_ts38pp_overlay_matches_sibling_pretaught_overlay(path: str) -> None:
    """Same n_examples/eval_every/max_steps/min_steps as the same-size
    ts38_pretaught_n<size>.yaml overlay — the target-stage arms all share one
    recipe, only theta0 differs."""
    pp_cfg = load_config(CONFIGS / TS38PP_BASE_FILE, CONFIGS / path)
    sibling_path = _ts38pp_sibling_pretaught_path(path)
    sibling_cfg = load_config(CONFIGS / "ts38_pretaught.yaml", CONFIGS / sibling_path)
    for field in ("n_examples",):
        assert pp_cfg["data"][field] == sibling_cfg["data"][field], field
    for field in ("eval_every", "max_steps"):
        assert pp_cfg["train"][field] == sibling_cfg["train"][field], field
    assert pp_cfg["train"]["stopping"]["min_steps"] == sibling_cfg["train"]["stopping"]["min_steps"]
    assert pp_cfg["train"]["max_steps"] >= pp_cfg["train"]["stopping"]["min_steps"]


@pytest.mark.parametrize(("path", "anchor"), TS38PP_OVERLAY_ANCHORS)
def test_ts38pp_overlay_match_data_order_with_is_base_arm(path: str, anchor: str) -> None:
    raw = yaml.safe_load((CONFIGS / path).read_text())
    assert raw["experiment"]["match_data_order_with"] == anchor
    assert not anchor.startswith("evt-ts38-pretaught-")


def test_ts38pp_vs_ts38_base_allowed_diffs() -> None:
    base = yaml.safe_load((CONFIGS / "ts38_base.yaml").read_text())
    pp = yaml.safe_load((CONFIGS / TS38PP_BASE_FILE).read_text())
    diffs = _leaf_diffs(base, pp)
    unexplained = diffs - _TS38_ALLOWED_DIFF_PATHS
    assert not unexplained, f"ts38pp vs ts38_base differ in unexpected fields: {unexplained}"


def test_ts38pp_vs_ts38_pretaught_differs_only_in_theta0_fields() -> None:
    pretaught = yaml.safe_load((CONFIGS / "ts38_pretaught.yaml").read_text())
    pp = yaml.safe_load((CONFIGS / TS38PP_BASE_FILE).read_text())
    diffs = _leaf_diffs(pretaught, pp)
    assert diffs == _TS38PP_VS_PRETAUGHT_ALLOWED_DIFF_PATHS, diffs
    assert pretaught["run_id"] != pp["run_id"]
    assert pretaught["experiment"]["parent_run_id"] != pp["experiment"]["parent_run_id"]
    assert (
        pretaught["experiment"]["parent_required_gates"]
        != pp["experiment"]["parent_required_gates"]
    )
    assert pretaught["experiment"]["arm"] == pp["experiment"]["arm"]
    assert pretaught["experiment"]["match_data_order_with"] is None
    assert pp["experiment"]["match_data_order_with"] is None


def test_ts38pp_target_vs_ts38mw_and_ts38pf_identical_recipe() -> None:
    """The recipe (everything except theta0/run_id/parent bookkeeping) must
    be identical across all three "new parent" arms — a silent recipe drift
    between ts38pp_pretaught.yaml and its ts38mw/ts38pf siblings would break
    the "arms differ only in theta0" invariant the analysis tooling assumes."""
    pp = yaml.safe_load((CONFIGS / TS38PP_BASE_FILE).read_text())
    mw = yaml.safe_load((CONFIGS / TS38MW_BASE_FILE).read_text())
    pf = yaml.safe_load((CONFIGS / TS38PF_BASE_FILE).read_text())
    allowed = {"run_id", "experiment.parent_run_id"}
    assert _leaf_diffs(pp, mw) == allowed, _leaf_diffs(pp, mw)
    assert _leaf_diffs(pp, pf) == allowed, _leaf_diffs(pp, pf)


def test_ts38pp_merged_config_values() -> None:
    cfg = load_config(CONFIGS / TS38PP_BASE_FILE, None)
    base_cfg = load_config(CONFIGS / "ts38_base.yaml", None)
    assert cfg["experiment"]["parent_required_gates"] == []
    assert cfg["experiment"]["parent_run_id"] == _TS38PP_PARENT_RUN_ID
    assert cfg["experiment"]["require_full_epoch1"] is True
    assert cfg["lora"]["r"] == 128
    assert cfg["lora"]["alpha"] == 32
    assert cfg["train"]["lr"] == 1.0e-3
    assert cfg["train"]["lr"] == base_cfg["train"]["lr"]
    assert cfg["train"]["snapshots"]["n"] == 0
    # Target data is UNCHANGED D_algo_bare — only the parent differs from
    # base_cfg, not the target task.
    assert cfg["data"]["order_hash"] == base_cfg["data"]["order_hash"]
    assert cfg["data"]["eval_order_hash"] == base_cfg["data"]["eval_order_hash"]


def test_ts38pp_base_builds_a_manifest_with_correct_regime() -> None:
    train_target = load("train_target")
    cfg = load_config(CONFIGS / TS38PP_BASE_FILE, None)
    manifest = train_target.manifest_fields(
        cfg,
        n_train=1,
        n_trainable=1,
        epochs_total=1,
        schedule=[],
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="bf16",
    )
    assert manifest["regime"] == "elicit"


@pytest.mark.parametrize("path", TS38PP_OVERLAY_PATHS)
def test_ts38pp_overlay_builds_a_manifest(path: str) -> None:
    train_target = load("train_target")
    cfg = load_config(CONFIGS / TS38PP_BASE_FILE, CONFIGS / path)
    train_target.manifest_fields(
        cfg,
        n_train=1,
        n_trainable=1,
        epochs_total=1,
        schedule=[],
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="bf16",
    )


# ---- the paper-protocol full-FT parent build config (NEW vs. ts38mw/
# ts38pf — those families pulled/built a LoRA parent; this one is FULL FT,
# so it must NOT opt into own_lora_block, mirroring ts38_pretaught_parent.
# yaml's guard rather than ts38pf's) ----------------------------------------


def test_ts38pp_parent_own_yaml_has_no_lora_block() -> None:
    # The un-merged file, not the deep-merged cfg — common.yaml's shared
    # default lora block must NOT leak into this full-FT parent's own keys.
    raw = yaml.safe_load((CONFIGS / TS38PP_PARENT_FILE).read_text())
    assert "lora" not in raw


def test_ts38pp_parent_has_no_own_lora_block() -> None:
    """Mirror of test_ts38_parent_has_no_own_lora_block for the paper-
    protocol parent: the full-FT footgun this family is most exposed to
    (common.yaml's shared default lora block silently switching this run to
    LoRA and invalidating the paper-protocol comparison)."""
    train_sft = load("train_sft")
    path = CONFIGS / TS38PP_PARENT_FILE
    assert train_sft.own_lora_block(load_config(path, None), path, None) is None


def test_ts38pp_parent_lr_pinned_from_the_full_ft_ladder() -> None:
    cfg = load_config(CONFIGS / TS38PP_PARENT_FILE, None)
    assert cfg["train"]["lr"] == 3.0e-5


def test_ts38pp_parent_data_hash_and_local_path_pinned() -> None:
    cfg = load_config(CONFIGS / TS38PP_PARENT_FILE, None)
    assert cfg["data"]["file"] == "D_target_4M.parquet"
    assert cfg["data"]["order_hash"] == TS38PP_PARENT_ORDER_HASH
    # local_path takes precedence over hf_id at load time
    # (geode/arith/load.py::load_frozen_parquet) — D_target_4M is box-
    # generated, not on the HF dataset repo; hf_id is kept only because
    # manifest_fields reads it unconditionally for the dataset name string.
    assert cfg["data"]["local_path"] == "experiments/training-run/data/full/D_target_4M.parquet"
    assert cfg["data"]["hf_id"] == "mhieuuu/elicit-vs-teach-arith"


def test_ts38pp_parent_one_epoch_step_count_is_31093() -> None:
    """The ONE pre-registered exception to run-until-convergence (paper App.
    E.2's single-epoch protocol, launch_ts38pp_family.sh's hard-coded 31093
    check): min_steps == max_steps must equal the TRUE one-epoch step count
    under train_sft.py's OWN step counting (n_val = round(val_fraction * n)
    via geode.train.packing.split_indices, then floor-division/drop-last
    batching — NOT train_target.py's ceil/no-drop-last convention the
    overlays above use), not a hand-picked number."""
    cfg = load_config(CONFIGS / TS38PP_PARENT_FILE, None)
    d, t = cfg["data"], cfg["train"]
    n_val = max(
        1, min(round(d["val_fraction"] * TS38PP_PARENT_N_EXAMPLES), TS38PP_PARENT_N_EXAMPLES - 1)
    )
    n_train = TS38PP_PARENT_N_EXAMPLES - n_val
    steps_per_epoch = n_train // t["batch_size"]
    assert n_val == 20000
    assert n_train == 3980000
    assert steps_per_epoch == TS38PP_PARENT_ONE_EPOCH_STEPS
    assert t["max_steps"] == TS38PP_PARENT_ONE_EPOCH_STEPS
    assert t["stopping"]["min_steps"] == TS38PP_PARENT_ONE_EPOCH_STEPS


def test_ts38pp_parent_epochs_total_and_cost_estimate_are_one() -> None:
    cfg = load_config(CONFIGS / TS38PP_PARENT_FILE, None)
    assert cfg["train"]["epochs_total_planned"] == 1
    assert cfg["cost"]["assumed_epochs_for_estimate"] == 1


def test_ts38pp_parent_snapshot_steps_are_quarter_epoch_marks() -> None:
    cfg = load_config(CONFIGS / TS38PP_PARENT_FILE, None)
    steps = cfg["train"]["snapshot_steps"]
    assert steps == TS38PP_PARENT_SNAPSHOT_STEPS
    assert steps == sorted(steps)
    assert steps == sorted(set(steps))  # strictly increasing, no duplicates
    assert steps[-1] < cfg["train"]["max_steps"]
    # Each mark is floor(one_epoch_steps * {0.25, 0.5, 0.75}).
    one_epoch = TS38PP_PARENT_ONE_EPOCH_STEPS
    assert steps == [int(one_epoch * frac) for frac in (0.25, 0.5, 0.75)]


def test_ts38pp_parent_builds_a_full_ft_manifest() -> None:
    """No committed full-FT config before this one has ever exercised
    manifest_fields with a non-empty train.snapshot_steps — every earlier
    ts38(*) full-FT parent (ts38_pretaught_parent.yaml) leaves snapshots off
    entirely; this is the check that catches a manifest_fields regression on
    that combination before any GPU spend (module docstring's KeyError-deep-
    in-register_run failure mode)."""
    train_sft = load("train_sft")
    path = CONFIGS / TS38PP_PARENT_FILE
    cfg = load_config(path, None)
    lora_cfg = train_sft.own_lora_block(cfg, path, None)
    assert lora_cfg is None
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
    assert manifest["training"]["method"] == "full_ft"
    assert manifest["training"]["lora"]["rank"] is None
    assert manifest["snapshot_steps"] == TS38PP_PARENT_SNAPSHOT_STEPS


# =============================================================================
# ts1b (fig2ts) — paper-scale staged redo, Stage 0 build B0.4 (decisions.md
# 2026-08-19 "ts1b (fig2ts) staged redo: PRE-REGISTRATION"; scripts/
# launch_ts1b_stage12.sh). Two full-FT parents at TinyStories-1B (exact
# Llama-3.2-1B dims/tokenizer) mirroring ts38pp_parent.yaml's role/field
# conventions -- pp (App. E.2, correct labels, one epoch pinned) and pf
# (App. E.1.2, permuted labels, until convergence). Smoke-level per the
# brief: fields present, pins exact (derived from first principles, same
# style as TS38PP_PARENT_ONE_EPOCH_STEPS above), no lora block in either
# parent. Target-stage configs are NOT built yet (Stage 3+ needs owner
# re-confirmation) -- this section covers configs/ts1b_pp_parent.yaml,
# configs/ts1b_pf_parent.yaml, and configs/sweeps/ts1b/pp_lrsweep_*.yaml
# only; the pretrain-stage config (ts1b_pretrain.yaml) and its own
# lrsweep_*.yaml overlays predate this section and are untouched.
# =============================================================================

TS1B_PP_PARENT_FILE = "ts1b_pp_parent.yaml"
TS1B_PF_PARENT_FILE = "ts1b_pf_parent.yaml"
TS1B_PP_LRSWEEP_RUNGS = ["1e-4", "3e-5", "1e-5"]
TS1B_PP_LRSWEEP_FILES = [f"sweeps/ts1b/pp_lrsweep_{r}.yaml" for r in TS1B_PP_LRSWEEP_RUNGS]

# The one-epoch pin's own arithmetic (ts1b_pp_parent.yaml's header,
# BYTE-IDENTICAL derivation to ts38pp_parent.yaml's above -- same
# N=4,000,000, same val_fraction=0.005, same batch=128; model size never
# enters this arithmetic): n_val = round(0.005 * 4,000,000) = 20,000,
# n_train = 3,980,000, steps_per_epoch = n_train // 128 = 31,093. The
# pre-registration's own prose (decisions.md, EXPERIMENTS.md) quotes
# 31,250 (naive 4,000,000 / 128); that number is UNREACHABLE under
# geode.train.packing.split_indices (val_fraction must be > 0, so
# n_val = max(1, ...) >= 1 always), so 31,093 is the value actually
# committed below -- this test section is the correctness anchor for that
# correction, not a re-derivation done differently.
TS1B_N_EXAMPLES = 4_000_000
TS1B_VAL_FRACTION = 0.005
TS1B_ONE_EPOCH_STEPS = 31093
TS1B_PF_CEILING_STEPS = 3 * TS1B_ONE_EPOCH_STEPS  # 93,279 -- 3-epoch ceiling

# Copied from ts1b_pretrain.yaml's own model block (exact Llama-3.2-1B
# dims) -- both new parents must match the base checkpoint they warm-start
# from.
TS1B_MODEL = {
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rope_theta": 500000.0,
    "max_position_embeddings": 2048,
    "tie_word_embeddings": True,
}

_TS1B_PARENT_RUN_ID_PATTERN = re.compile(r"^evt-ts1b-p[pf]-parent$")
_TS1B_LRSWEEP_RUN_ID_PATTERN = re.compile(r"^evt-ts1b-pp-lrsweep-\d+e-\d+$")

# pp vs pf: labels permuted is the ONLY intended difference
# (ts1b_pf_parent.yaml's own header ruling, the 1B twin-parents design) --
# data identity and its step-count consequences change, nothing else
# (including the placeholder LR, which both configs must share).
_TS1B_PP_VS_PF_ALLOWED_DIFF_PATHS = {
    "run_id",
    "task.name",
    "data.file",
    "data.order_hash",
    "data.local_path",
    "train.max_steps",
    "train.epochs_total_planned",
    "cost.assumed_epochs_for_estimate",
}


def test_ts1b_family_files_exist() -> None:
    for rel in (TS1B_PP_PARENT_FILE, TS1B_PF_PARENT_FILE, *TS1B_PP_LRSWEEP_FILES):
        assert (CONFIGS / rel).is_file(), rel


@pytest.mark.parametrize("path", [TS1B_PP_PARENT_FILE, TS1B_PF_PARENT_FILE])
def test_ts1b_parent_run_id_matches_pattern(path: str) -> None:
    cfg = load_config(CONFIGS / path, None)
    assert _TS1B_PARENT_RUN_ID_PATTERN.match(cfg["run_id"]), cfg["run_id"]


@pytest.mark.parametrize("path", [TS1B_PP_PARENT_FILE, TS1B_PF_PARENT_FILE])
def test_ts1b_parent_arm_role_and_parent(path: str) -> None:
    cfg = load_config(CONFIGS / path, None)
    assert cfg["experiment"]["arm"] == "ts1b"
    # "parent" is not a role value this repo uses anywhere (only
    # pretrain/pre_teach/format_install/target exist, grep-verified) --
    # pre_teach matches ts38pp_parent.yaml's / ts38_preteachfmt_parent.
    # yaml's role value for the same App. E.2/E.1.2 concepts.
    assert cfg["experiment"]["role"] == "pre_teach"
    assert cfg["experiment"]["parent_run_id"] == "evt-ts1b-base"
    assert cfg["experiment"]["parent_required_gates"] == []


@pytest.mark.parametrize("path", [TS1B_PP_PARENT_FILE, TS1B_PF_PARENT_FILE])
def test_ts1b_parent_model_matches_ts1b_pretrain(path: str) -> None:
    """Model dims must match evt-ts1b-base's architecture exactly -- a
    mismatch here is also caught loudly by launch_ts1b_stage12.sh's own
    stage-0 dims check, but a config that silently drifts from
    ts1b_pretrain.yaml's dims before that check ever runs wastes a box
    boot finding out."""
    cfg = load_config(CONFIGS / path, None)
    for key, want in TS1B_MODEL.items():
        assert cfg["model"][key] == want, f"{path}: model.{key}"
    assert cfg["tokenizer"]["path"] == "meta-llama/Llama-3.2-1B"


@pytest.mark.parametrize("path", [TS1B_PP_PARENT_FILE, TS1B_PF_PARENT_FILE])
def test_ts1b_parent_own_yaml_has_no_lora_block(path: str) -> None:
    # The un-merged file, not the deep-merged cfg -- common.yaml's shared
    # default lora block must NOT leak into either full-FT parent's own
    # keys (same guard shape as test_ts38pp_parent_own_yaml_has_no_lora_block).
    raw = yaml.safe_load((CONFIGS / path).read_text())
    assert "lora" not in raw


@pytest.mark.parametrize("path", [TS1B_PP_PARENT_FILE, TS1B_PF_PARENT_FILE])
def test_ts1b_parent_has_no_own_lora_block(path: str) -> None:
    train_sft = load("train_sft")
    cfg_path = CONFIGS / path
    assert train_sft.own_lora_block(load_config(cfg_path, None), cfg_path, None) is None


def test_ts1b_pp_parent_one_epoch_step_count() -> None:
    """Same derivation as test_ts38pp_parent_one_epoch_step_count_is_31093,
    re-run here because the pre-registration's own prose quotes the
    unreachable 31,250 -- this is the correctness anchor pinning the value
    actually committed in ts1b_pp_parent.yaml."""
    cfg = load_config(CONFIGS / TS1B_PP_PARENT_FILE, None)
    d, t = cfg["data"], cfg["train"]
    assert d["val_fraction"] == TS1B_VAL_FRACTION
    n_val = max(1, min(round(d["val_fraction"] * TS1B_N_EXAMPLES), TS1B_N_EXAMPLES - 1))
    n_train = TS1B_N_EXAMPLES - n_val
    steps_per_epoch = n_train // t["batch_size"]
    assert n_val == 20000
    assert n_train == 3980000
    assert steps_per_epoch == TS1B_ONE_EPOCH_STEPS
    assert t["max_steps"] == TS1B_ONE_EPOCH_STEPS
    assert t["stopping"]["min_steps"] == TS1B_ONE_EPOCH_STEPS
    assert t["epochs_total_planned"] == 1
    assert cfg["cost"]["assumed_epochs_for_estimate"] == 1


def test_ts1b_pf_parent_stopping_regime() -> None:
    """pf trains UNTIL CONVERGENCE (unlike pp's pinned one-epoch): min_steps
    is exactly one full epoch (same arithmetic as pp -- do not declare
    'converged' before the model has seen the permuted-label corpus once),
    max_steps is a genuine 3-epoch ceiling eps/k is expected to beat."""
    cfg = load_config(CONFIGS / TS1B_PF_PARENT_FILE, None)
    d, t = cfg["data"], cfg["train"]
    assert d["val_fraction"] == TS1B_VAL_FRACTION
    assert t["stopping"]["min_steps"] == TS1B_ONE_EPOCH_STEPS
    assert t["max_steps"] == TS1B_PF_CEILING_STEPS == 93279
    # genuinely load-bearing here, unlike pp's inert min_steps==max_steps copy
    assert t["max_steps"] > t["stopping"]["min_steps"]
    assert t["epochs_total_planned"] == 3
    assert cfg["cost"]["assumed_epochs_for_estimate"] == 3


def test_ts1b_pp_vs_pf_differs_only_in_labels_and_step_count() -> None:
    """Labels permuted is the ONLY intended difference between pp and pf
    (ts1b_pf_parent.yaml's own header ruling, mirroring the owner's 38M
    twin-parents design) -- everything else, including the placeholder LR,
    must stay byte-identical."""
    pp = yaml.safe_load((CONFIGS / TS1B_PP_PARENT_FILE).read_text())
    pf = yaml.safe_load((CONFIGS / TS1B_PF_PARENT_FILE).read_text())
    diffs = _leaf_diffs(pp, pf)
    assert diffs == _TS1B_PP_VS_PF_ALLOWED_DIFF_PATHS, diffs
    assert pp["train"]["lr"] == pf["train"]["lr"]
    assert pp["experiment"]["parent_run_id"] == pf["experiment"]["parent_run_id"] == "evt-ts1b-base"
    assert pp["data"]["order_hash"] != pf["data"]["order_hash"]
    assert pp["data"]["file"] != pf["data"]["file"]


@pytest.mark.parametrize("path", [TS1B_PP_PARENT_FILE, TS1B_PF_PARENT_FILE])
def test_ts1b_parent_data_order_hash_pinned(path: str) -> None:
    """FLIPPED 2026-08-19 (was ...still_placeholder, by its own design):
    B0.1's block-render datagen landed, and both parents' data.order_hash
    are now real sha256 pins (filled from data/full/report.json's
    D_target_4M_block / D_target_4M_blockperm entries). A regression back
    to a TODO_* placeholder -- or to anything that is not a 64-char hex
    digest -- would make launch_ts1b_stage12.sh's stale-parent guard
    vacuous, so pin the SHAPE here (the values live in report.json and the
    configs; duplicating them here would just be a third copy to drift)."""
    cfg = load_config(CONFIGS / path, None)
    h = cfg["data"]["order_hash"]
    assert not h.startswith("TODO_")
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h), h


@pytest.mark.parametrize("path", [TS1B_PP_PARENT_FILE, TS1B_PF_PARENT_FILE])
def test_ts1b_parent_builds_a_full_ft_manifest(path: str) -> None:
    train_sft = load("train_sft")
    cfg_path = CONFIGS / path
    cfg = load_config(cfg_path, None)
    lora_cfg = train_sft.own_lora_block(cfg, cfg_path, None)
    assert lora_cfg is None
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
    assert manifest["training"]["method"] == "full_ft"
    assert manifest["training"]["lora"]["rank"] is None
    assert manifest["snapshot_steps"] == []  # no snapshot_steps key -- see header


# ---- LR mini-sweep overlays (Stage 1) --------------------------------------


@pytest.mark.parametrize("rung", TS1B_PP_LRSWEEP_RUNGS)
def test_ts1b_pp_lrsweep_overlay_values(rung: str) -> None:
    cfg = load_config(
        CONFIGS / TS1B_PP_PARENT_FILE, CONFIGS / f"sweeps/ts1b/pp_lrsweep_{rung}.yaml"
    )
    assert cfg["run_id"] == f"evt-ts1b-pp-lrsweep-{rung}"
    assert cfg["train"]["max_steps"] == 2000
    assert cfg["train"]["eval_every"] == 200
    assert cfg["train"]["lr"] == float(rung)
    # merged with the base config: min_steps (inherited, one full epoch) is
    # >> max_steps (this overlay's 2000), so the plateau rule stays inert
    # and stop_reason=max_steps is the EXPECTED outcome for every rung.
    assert cfg["train"]["stopping"]["min_steps"] == TS1B_ONE_EPOCH_STEPS
    assert cfg["train"]["stopping"]["min_steps"] > cfg["train"]["max_steps"]


def test_ts1b_pp_lrsweep_run_ids_distinct_and_out_of_other_families() -> None:
    run_ids = [f"evt-ts1b-pp-lrsweep-{r}" for r in TS1B_PP_LRSWEEP_RUNGS]
    assert len(run_ids) == len(set(run_ids))
    for rid in run_ids:
        assert _TS1B_LRSWEEP_RUN_ID_PATTERN.match(rid), rid
        # never collides with the pretrain-stage sweep's own run ids
        # (evt-ts1b-lrsweep-1e-3/3e-4/5e-4)
        assert not rid.startswith("evt-ts1b-lrsweep-")


@pytest.mark.parametrize("rung", TS1B_PP_LRSWEEP_RUNGS)
def test_ts1b_pp_lrsweep_overlay_builds_a_full_ft_manifest(rung: str) -> None:
    train_sft = load("train_sft")
    base_path = CONFIGS / TS1B_PP_PARENT_FILE
    override_path = CONFIGS / f"sweeps/ts1b/pp_lrsweep_{rung}.yaml"
    cfg = load_config(base_path, override_path)
    lora_cfg = train_sft.own_lora_block(cfg, base_path, override_path)
    assert lora_cfg is None
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
    assert manifest["training"]["method"] == "full_ft"
    assert manifest["run_id"] == f"evt-ts1b-pp-lrsweep-{rung}"
