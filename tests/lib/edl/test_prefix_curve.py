"""V5.73 — geode.edl.prefix_edl_curve: required floor + moving-vs-fixed floor.

``floor`` is a required keyword (omitting it is a ``TypeError``); the ``"val"``
moving floor and the ``"test"`` fixed floor produce different EDL curves on the
same run — the floor-artifact the unified function makes explicit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geode.edl.metrics import prefix_edl_curve
from geode.zoo import PrequentialRecord, register_run, write_jsonl

HASH_A = "a1" * 32


def _manifest_fields(run_id: str) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": "2026-07-29T00:00:00+00:00",
        "git_commit": "deadbeef",
        "regime": "elicit",
        "base_model": {"hf_id": "tiny/llama", "revision": "main"},
        "task": {"name": "arith", "format_version": "1"},
        "dataset": {"name": "d", "n_unique_examples": 6, "seed": 0},
        "training": {
            "method": "lora",
            "lora": {
                "rank": 2,
                "alpha": 4.0,
                "target_modules": ["q_proj"],
                "dropout": 0.0,
                "sparse_param_count": None,
            },
            "optimizer": {
                "name": "adamw",
                "lr": 0.001,
                "batch_size": 2,
                "micro_batch_size": None,
                "betas": None,
                "weight_decay": 0.0,
                "grad_clip": None,
            },
            "lr_schedule": "constant",
            "min_lr": None,
            "precision": "fp32",
            "eval_every": 1,
            "max_steps": None,
            "stopping": {"eps_nats": None, "k": None, "min_steps": None},
            "epochs_total": 1,
            "seed": 0,
        },
        "trainable_param_count": 1000,
        "snapshot_steps": [],
        "cost": {"gpu_type": None, "est_usd": None, "actual_usd": None},
        "status": "complete",
        "masking_config_hash": HASH_A,
    }


def _build_run(geode_store: Path, run_id: str) -> Path:
    """A run with 3 epoch-1 records (steps 1-3), a moving eval log, and a fixed
    test floor. cum loss 8/12/14 over cum tokens 4/8/12, cum examples 2/4/6."""
    d = geode_store / "runs" / run_id
    register_run(_manifest_fields(run_id))
    write_jsonl(
        d / "logs" / "prequential.jsonl",
        [
            PrequentialRecord(step=1, epoch=1, example_ids=[0, 1], label_token_count=4, loss_sum_nats=8.0),
            PrequentialRecord(step=2, epoch=1, example_ids=[2, 3], label_token_count=4, loss_sum_nats=4.0),
            PrequentialRecord(step=3, epoch=1, example_ids=[4, 5], label_token_count=4, loss_sum_nats=2.0),
        ],
    )
    (d / "eval_log.jsonl").write_text(
        "\n".join(
            json.dumps({"step": s, "val_loss_nats": v})
            for s, v in [(1, 2.0), (2, 1.0), (3, 0.5)]
        )
        + "\n"
    )
    eval_dir = d / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "test_loss.json").write_text(
        json.dumps(
            {
                "n_test_examples": 8,
                "label_token_count": 16,
                "loss_sum_nats": 8.0,
                "loss_per_label_token_nats": 0.5,  # constant test floor
                "masking_config_hash": HASH_A,
            }
        )
    )
    return d


def test_v5_73_floor_is_required(geode_store: Path) -> None:
    _build_run(geode_store, "run-floor-req")
    with pytest.raises(TypeError):
        prefix_edl_curve("run-floor-req", store=geode_store)  # type: ignore[call-arg]


def test_v5_73_val_vs_test_floor_differ(geode_store: Path) -> None:
    _build_run(geode_store, "run-floors")
    dv = prefix_edl_curve("run-floors", floor="val", store=geode_store)
    dt = prefix_edl_curve("run-floors", floor="test", store=geode_store)

    assert dv["step"].tolist() == [1, 2, 3]
    # At step 2: MDL 12, tokens 8. Moving val floor 1.0 -> EDL 12-8*1.0 = 4.0;
    # fixed test floor 0.5 -> EDL 12-8*0.5 = 8.0. The floor choice changes the curve.
    assert dv["edl_nats"].iloc[1] == pytest.approx(4.0)
    assert dt["edl_nats"].iloc[1] == pytest.approx(8.0)
    assert not dv["edl_nats"].equals(dt["edl_nats"])
    # Full-epoch-1 totals ride in attrs (not the last eval point).
    assert dv.attrs["n_epoch1_steps"] == 3
    assert dv.attrs["epoch1_totals"] == pytest.approx((14.0, 12.0, 6.0))


def test_v5_73_invalid_floor_raises(geode_store: Path) -> None:
    _build_run(geode_store, "run-bad-floor")
    with pytest.raises(ValueError, match="floor must be"):
        prefix_edl_curve("run-bad-floor", floor="bogus", store=geode_store)  # type: ignore[arg-type]
