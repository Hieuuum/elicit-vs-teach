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
TS38_OVERLAYS = [
    ("sweeps/ts38/ts38_base_n1000.yaml", 1000, 5, 1000, 8),
    ("sweeps/ts38/ts38_base_n4642.yaml", 4642, 5, 1000, 37),
    ("sweeps/ts38/ts38_base_n21544.yaml", 21544, 25, 5000, 169),
    ("sweeps/ts38/ts38_base_n100000.yaml", 100000, 125, 10000, 782),
    ("sweeps/ts38/ts38_base_n316228.yaml", 316228, 375, 30000, 2471),
    ("sweeps/ts38/ts38_pretaught_n1000.yaml", 1000, 5, 1000, 8),
    ("sweeps/ts38/ts38_pretaught_n4642.yaml", 4642, 5, 1000, 37),
    ("sweeps/ts38/ts38_pretaught_n21544.yaml", 21544, 25, 5000, 169),
    ("sweeps/ts38/ts38_pretaught_n100000.yaml", 100000, 125, 10000, 782),
    ("sweeps/ts38/ts38_pretaught_n316228.yaml", 316228, 375, 30000, 2471),
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
