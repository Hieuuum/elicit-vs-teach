"""``ts38fs_dose_curve.py`` cell enumeration, incl. the ts38dense extension
(EXPERIMENTS §6.21, decisions.md 2026-08-21 "ts38dense pre-registration").

ts38dense adds 5 NEW cells to the ts38fs 3-axis (install i x target n x
seed s) grid: the i=1000 dose point, densified target sizes n in
{2154, 10000, 46416, 146780, 215443}, seed 316 ONLY (not a cartesian
extension against every install or seed — those combinations were never
trained). This file checks the grid-construction primitives
(``DENSE_CELLS``/``expected_cells()``/``TOTAL_CELLS``/``run_id_for()``) in
isolation, plus ONE end-to-end ``collect()`` pass over a synthetic store
proving the dense cells round-trip through the real pipeline with the
right ``install_i``/``n``/``seed`` and are reported ``pending`` when absent,
matching the existing style in ``test_edl_converged_val_floor_families.py``.

CPU-only, no persistent store, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

from geode.zoo import PrequentialRecord, register_run, write_jsonl

from tests._scriptloader import load

dose = load("ts38fs_dose_curve")

_HASH = "a1" * 32  # one well-formed masking hash shared by manifest + test_loss.json


def test_dense_sizes_and_cells_are_exactly_five() -> None:
    assert dose.DENSE_SIZES == (2154, 10000, 46416, 146780, 215443)
    assert len(dose.DENSE_CELLS) == 5
    assert dose.DENSE_CELLS == tuple((1000, n, 316) for n in dose.DENSE_SIZES)


def test_dense_sizes_disjoint_from_shipped_sizes() -> None:
    """The 5 new sizes must not collide with the 5 shipped SIZES — a
    collision would silently duplicate a cell in expected_cells()."""
    assert not (set(dose.DENSE_SIZES) & set(dose.SIZES))


def test_expected_cells_is_the_60cell_grid_plus_dense_cells_only() -> None:
    """expected_cells() must equal the original 4x5x3 cartesian product plus
    exactly the 5 dense cells — NOT dense sizes crossed against any other
    install or seed (those (i, n, s) combinations were never trained and
    would sit "pending" forever)."""
    cells = dose.expected_cells()
    original = {(i, n, s) for s in dose.SEEDS for i in dose.INSTALLS for n in dose.SIZES}

    assert len(original) == 60
    assert set(cells) == original | set(dose.DENSE_CELLS)
    assert len(cells) == len(set(cells))  # no duplicates
    assert len(cells) == 65

    # No dense size appears at any install other than 1000, nor any seed
    # other than 316.
    dense_size_set = set(dose.DENSE_SIZES)
    for i, n, s in cells:
        if n in dense_size_set:
            assert i == 1000
            assert s == 316


def test_total_cells_is_65_and_matches_expected_cells_length() -> None:
    assert dose.TOTAL_CELLS == 65
    assert dose.TOTAL_CELLS == len(dose.expected_cells())


def test_run_id_for_dense_cells_uses_the_standard_ts38fs_naming() -> None:
    """Dense cells (i=1000) never hit the (i=21544, s=316) ts38pf-reuse
    special case, so run_id_for() must fall through to the generic
    evt-ts38fs-i<i>-n<n>-s<s> pattern for every one of them."""
    for i, n, s in dose.DENSE_CELLS:
        assert dose.run_id_for(i, n, s) == f"evt-ts38fs-i1000-n{n}-s316"


def test_run_id_for_still_reuses_ts38pf_for_the_original_special_case() -> None:
    """Adding the dense cells must not disturb the existing (i=21544,
    s=316) -> evt-ts38pf-preteachfmt-n<n> reuse mapping."""
    assert dose.run_id_for(21544, 4642, 316) == "evt-ts38pf-preteachfmt-n4642"
    assert dose.run_id_for(21544, 4642, 1316) == "evt-ts38fs-i21544-n4642-s1316"


def _ts38fs_manifest(run_id: str, n: int, *, stop_reason: str) -> dict:
    """Minimal valid spec 00 §2 manifest carrying the D-1 masking-hash extra
    AND the ``experiment.target_result.stop_reason`` field this script's
    ``collect()`` reads directly off the raw manifest JSON (unlike
    ``edl_converged_val_floor.py``, which never reads stop_reason)."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": "2026-08-21T00:00:00+00:00",
        "git_commit": "deadbeef",
        "regime": "elicit",
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
        "masking_config_hash": _HASH,
        "experiment": {"target_result": {"stop_reason": stop_reason}},
    }


def _write_ts38fs_run(store: Path, run_id: str, n: int, *, stop_reason: str = "converged") -> None:
    """Register ``run_id`` and hand-write the artifacts ``collect()`` needs;
    the specific loss numbers don't matter (only that the row lands with the
    right install/n/seed), so fixed placeholder values — same pattern as
    ``test_edl_converged_val_floor_families.py``'s ``_write_ts38pp_run``."""
    register_run(_ts38fs_manifest(run_id, n, stop_reason=stop_reason), store=store)
    run_dir = store / "runs" / run_id
    write_jsonl(
        run_dir / "logs" / "prequential.jsonl",
        [
            PrequentialRecord(
                step=0, epoch=1, example_ids=[0, 1], label_token_count=4, loss_sum_nats=8.0
            )
        ],
    )
    with (run_dir / "eval_log.jsonl").open("w") as fh:
        fh.write(json.dumps({"step": 1, "val_loss_nats": 1.0, "stopping_eval": True}) + "\n")
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "test_loss.json").write_text(
        json.dumps(
            {
                "n_test_examples": 8,
                "label_token_count": 16,
                "loss_sum_nats": 16.0 * 0.5,
                "loss_per_label_token_nats": 0.5,
                "masking_config_hash": _HASH,
            }
        )
    )


def test_collect_finds_dense_cells_with_correct_install_and_seed(geode_store: Path) -> None:
    """Writing exactly the 5 dense cells (nothing else) must round-trip
    through ``collect()`` with install_i=1000, seed=316, n in DENSE_SIZES —
    and every other one of the 60 non-dense cells must come back pending."""
    for i, n, s in dose.DENSE_CELLS:
        _write_ts38fs_run(geode_store, dose.run_id_for(i, n, s), n)

    df, pending = dose.collect(geode_store)

    assert len(df) == 5
    assert set(df["install_i"]) == {1000}
    assert set(df["seed"]) == {316}
    assert set(df["n"]) == set(dose.DENSE_SIZES)
    assert set(df["run_id"]) == {f"evt-ts38fs-i1000-n{n}-s316" for n in dose.DENSE_SIZES}
    assert not df["reused_ts38pf"].any()

    assert len(pending) == 60
    dense_size_set = set(dose.DENSE_SIZES)
    assert not any(n in dense_size_set for _, n, _ in pending)


def test_collect_reports_a_missing_dense_cell_as_pending(geode_store: Path) -> None:
    """4 of the 5 dense cells present, 1 missing -> that one comes back
    pending, not silently dropped or crashing the whole collect() pass."""
    present = dose.DENSE_CELLS[:4]
    missing = dose.DENSE_CELLS[4]
    for i, n, s in present:
        _write_ts38fs_run(geode_store, dose.run_id_for(i, n, s), n)

    df, pending = dose.collect(geode_store)

    assert len(df) == 4
    assert missing in pending
    assert all(cell not in pending for cell in present)
