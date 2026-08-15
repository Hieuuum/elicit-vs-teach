"""``edl_converged_val_floor.py`` — the paper-floor (test-floor) columns.

Added 2026-08-15 when the OCV script grew ``edl_per_token_nats_test_floor`` /
``edl_per_token_bits_test_floor``: Donoway et al. Eq. 3 floors on the run's
own ``L_test(theta_T)`` (held-out test block), where OCV floors on the run's
own converged VAL loss (the rows the stopping rule watched). Everything else
— MDL, theta_T, D — is shared between the two columns, so their difference
must be exactly the floor difference. Silent failure here would put a
mislabelled curve on the paper-comparison figure, so the identities are
pinned on constructed numbers against a synthetic store (the same
``PrequentialRecord`` / ``eval/test_loss.json`` fabrication ``tests/lib/edl/
test_metrics.py`` uses). CPU-only, no network, no weights.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from geode.zoo import PrequentialRecord, register_run, write_jsonl

from tests._scriptloader import load

ecvf = load("edl_converged_val_floor")

HASH = "a1" * 32  # one well-formed masking hash shared by manifest + test_loss.json


def _manifest(run_id: str, n: int) -> dict:
    """Minimal valid spec 00 §2 manifest carrying the D-1 masking-hash extra."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": "2026-08-15T00:00:00+00:00",
        "git_commit": "deadbeef",
        "regime": "teach",
        "base_model": {"hf_id": "tiny/ts38", "revision": "main"},
        "task": {"name": "arith_target", "format_version": "1"},
        "dataset": {"name": "d_algo_bare", "n_unique_examples": n, "seed": 0},
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
                "lr": 1e-3,
                "batch_size": 2,
                "micro_batch_size": None,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
                "grad_clip": None,
            },
            "lr_schedule": "constant",
            "min_lr": None,
            "precision": "fp32",
            "eval_every": 1,
            "max_steps": None,
            "stopping": {"eps_nats": 0.002, "k": 5, "min_steps": 1},
            "epochs_total": 1,
            "seed": 0,
        },
        "trainable_param_count": 1000,
        "snapshot_steps": [],
        "cost": {"gpu_type": None, "est_usd": None, "actual_usd": None},
        "status": "complete",
        "masking_config_hash": HASH,
    }


def _write_run(
    store: Path,
    run_id: str,
    n: int,
    *,
    preq: list[tuple[int, float]],
    val_curve: list[float],
    l_test: float,
) -> tuple[float, int]:
    """Register ``run_id`` and hand-write prequential/eval/test artifacts.

    ``preq`` = [(label_token_count, loss_sum_nats), ...] epoch-1 records;
    ``val_curve`` = val_loss_nats per eval step (last one is theta_T's).
    Returns (mdl, D) so the test can compute expectations independently.
    """
    register_run(_manifest(run_id, n), store=store)
    run_dir = store / "runs" / run_id
    records = [
        PrequentialRecord(
            step=i,
            epoch=1,
            example_ids=[2 * i, 2 * i + 1],
            label_token_count=tok,
            loss_sum_nats=loss,
        )
        for i, (tok, loss) in enumerate(preq)
    ]
    write_jsonl(run_dir / "logs" / "prequential.jsonl", records)
    with (run_dir / "eval_log.jsonl").open("w") as fh:
        for step, v in enumerate(val_curve, start=1):
            fh.write(json.dumps({"step": step, "val_loss_nats": v, "stopping_eval": True}) + "\n")
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "test_loss.json").write_text(
        json.dumps(
            {
                "n_test_examples": 8,
                "label_token_count": 16,
                "loss_sum_nats": 16.0 * l_test,
                "loss_per_label_token_nats": l_test,
                "masking_config_hash": HASH,
            }
        )
    )
    mdl = sum(loss for _, loss in preq)
    d = sum(tok for tok, _ in preq)
    return mdl, d


def test_test_floor_column_is_paper_eq3_on_constructed_numbers(geode_store: Path) -> None:
    """``edl_per_token_nats_test_floor`` == (MDL − D·L_test)/D exactly, and its
    bits twin == nats/ln2 — the paper's Eq. 3 per label token."""
    mdl, d = _write_run(
        geode_store,
        "evt-ts38-base-n1000",
        1000,
        preq=[(4, 5.0), (6, 3.0), (2, 1.0)],  # MDL 9.0 over D 12
        val_curve=[2.0, 1.0, 0.7],  # theta_T val = 0.7
        l_test=0.5,
    )
    df = ecvf.collect("ts38mw", geode_store)
    assert len(df) == 1
    row = df.iloc[0]

    expected_nats = (mdl - d * 0.5) / d  # (9 - 6)/12 = 0.25
    assert row["edl_per_token_nats_test_floor"] == pytest.approx(expected_nats)
    assert row["edl_per_token_bits_test_floor"] == pytest.approx(expected_nats / math.log(2))
    # and OCV still floors on the converged VAL value, untouched by the addition
    assert row["edl_per_token_nats"] == pytest.approx((mdl - d * 0.7) / d)


def test_test_floor_matches_library_edl_nats(geode_store: Path) -> None:
    """The column reproduces ``geode.edl.metrics.edl_nats(run)/D`` — the
    library's canonical (test-floored) EDL — not a parallel definition."""
    from geode.edl.metrics import edl_nats, epoch1_totals

    _write_run(
        geode_store,
        "evt-ts38-base-n4642",
        4642,
        preq=[(3, 4.5), (5, 2.5), (4, 1.2), (4, 0.9)],
        val_curve=[3.0, 1.5, 0.9, 0.8],
        l_test=0.31,
    )
    df = ecvf.collect("ts38mw", geode_store)
    _, d, _ = epoch1_totals("evt-ts38-base-n4642", store=geode_store)
    assert df.iloc[0]["edl_per_token_nats_test_floor"] == pytest.approx(
        edl_nats("evt-ts38-base-n4642", store=geode_store) / d
    )


def test_ocv_minus_test_floor_equals_floor_gap(geode_store: Path) -> None:
    """OCV and paper columns share MDL and D; they may differ ONLY by the
    floor: (edl_test − edl_ocv)/token == L_val_conv − L_test, sign included.
    Pins the direction: a run whose val stops ABOVE its test loss gets a
    LOWER OCV EDL/D than paper EDL/D."""
    _write_run(
        geode_store,
        "evt-ts38-base-n21544",
        21544,
        preq=[(5, 6.0), (5, 4.0)],
        val_curve=[2.0, 1.3],  # theta_T val 1.3
        l_test=0.2,  # test 0.2 -> gap 1.1
    )
    df = ecvf.collect("ts38mw", geode_store)
    row = df.iloc[0]
    gap = row["edl_per_token_nats_test_floor"] - row["edl_per_token_nats"]
    assert gap == pytest.approx(row["l_val_converged_nats"] - row["l_test_nats"])
    assert gap == pytest.approx(1.1)
    assert row["edl_per_token_nats_test_floor"] > row["edl_per_token_nats"]


def test_test_floor_is_per_run_never_shared_across_n(geode_store: Path) -> None:
    """Two sizes with different L_test each subtract their OWN — the paper's
    per-n floor, not a shared constant. Guards against a refactor that
    accidentally broadcasts one run's test loss over the family."""
    _write_run(
        geode_store,
        "evt-ts38-base-n1000",
        1000,
        preq=[(4, 8.0)],  # avg 2.0
        val_curve=[1.9],
        l_test=1.5,
    )
    _write_run(
        geode_store,
        "evt-ts38-base-n100000",
        100000,
        preq=[(4, 4.0)],  # avg 1.0
        val_curve=[0.1],
        l_test=0.05,
    )
    df = ecvf.collect("ts38mw", geode_store).set_index("n")
    assert df.loc[1000, "edl_per_token_nats_test_floor"] == pytest.approx(2.0 - 1.5)
    assert df.loc[100000, "edl_per_token_nats_test_floor"] == pytest.approx(1.0 - 0.05)
    # the two floors actually used differ, i.e. nothing was broadcast
    assert df.loc[1000, "l_test_nats"] != df.loc[100000, "l_test_nats"]


def test_plot_draws_paper_floor_series_and_tolerates_old_csv(
    geode_store: Path, tmp_path: Path
) -> None:
    """The figure gains a dashed test-floor twin per arm when the column is
    present, and still renders (OCV-only) from a pre-2026-08-15 frame that
    lacks it — old committed CSVs must replot, not crash."""
    import pandas as pd

    _write_run(
        geode_store, "evt-ts38-base-n1000", 1000, preq=[(4, 8.0)], val_curve=[1.0], l_test=0.5
    )
    _write_run(
        geode_store, "evt-ts38-base-n4642", 4642, preq=[(4, 6.0)], val_curve=[0.5], l_test=0.3
    )
    df = ecvf.collect("ts38mw", geode_store)

    out_new = tmp_path / "with_test_floor.png"
    ecvf.plot(df, out_new, "ts38mw-test")
    assert out_new.is_file() and out_new.stat().st_size > 0

    old = df.drop(columns=["edl_per_token_nats_test_floor", "edl_per_token_bits_test_floor"])
    out_old = tmp_path / "ocv_only.png"
    ecvf.plot(pd.DataFrame(old), out_old, "ts38mw-old")
    assert out_old.is_file() and out_old.stat().st_size > 0
