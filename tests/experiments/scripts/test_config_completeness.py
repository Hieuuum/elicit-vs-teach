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

from pathlib import Path

import pytest

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
    "llama_fig2_installer.yaml",
]
EMBEDDING_WARMSTART = [
    "archive/phase3/p3/warm_sum.yaml",
    "archive/phase3/p3/warm_colon.yaml",
    "archive/phase3/p3/warm_sum_colon.yaml",
]
LORA_TARGET = [
    ("archive/runs/run10_llama1b_target.yaml", None),
    ("llama_fig2_noinst.yaml", None),
    ("llama_fig2_inst.yaml", None),
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


def test_llama_fig2_installer_opts_into_lora_via_own_lora_block() -> None:
    """Pins the CRITICAL gating footgun train_sft.py's ``own_lora_block``
    docstring warns about: common.yaml merges a default ``lora:`` block into
    every config (including full-FT ones), so gating on the merged
    ``cfg["lora"]`` — instead of the run's OWN ``lora:`` key — would silently
    switch every full-FT launch to LoRA (and, post-2026-07-31, write a bogus
    adapter sidecar for weights that were never LoRA-trained). The
    redesigned fig-2 installer must opt in via its own key; a genuine
    full-FT config must not."""
    train_sft = load("train_sft")

    lora_path = CONFIGS / "llama_fig2_installer.yaml"
    lora_cfg = train_sft.own_lora_block(load_config(lora_path, None), lora_path, None)
    assert lora_cfg is not None
    assert (lora_cfg["r"], lora_cfg["alpha"]) == (64, 32)

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
