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
