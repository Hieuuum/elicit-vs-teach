"""Figure-2 dataset-size sweep driver (``analysis/dataset_size_sweep.py``).

Silent failure modes guarded, modeled on ``test_trajectory.py``: this driver
is the only thing standing between a partially-pulled store (some runs still
training, some not yet pulled at all) and a figure that looks complete when
it isn't. A synthetic store stands in for a real ``geode-store`` pull —
manifests with ``experiment.target_result``/``experiment.gates.G5`` and a
hand-written ``eval/test_loss.json``, no model weights anywhere (this driver
never touches them). CPU-only, fast.

Covers:

- rows read off a converged run carry every metric this driver claims to
  read (min_val_nats, the three edl_* fields, test_loss_per_label_token_nats,
  g5_zero_shot_em), with the spec 00 §7 / ZOO-6 required columns plus the
  condition/curve_label/stop_reason extras;
- a ``stop_reason != "converged"`` run is still included in the rows (never
  silently dropped) but prints a WARNING naming the run;
- a run whose manifest was never registered (not yet pulled) is skipped —
  ``None``, no exception — with its own warning;
- the assembled table round-trips through ``geode.zoo.write_results`` /
  ``read_results`` (the ZOO-6 long-format writer) without losing rows or
  columns;
- the figure file is created from a small multi-size, multi-condition table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geode.zoo import RunManifest, read_results, register_run
from geode.zoo.results import REQUIRED_COLUMNS
from geode.zoo.store import run_dir

from tests._scriptloader import load

dss = load("dataset_size_sweep")

BASE_MODEL = "meta-llama/Llama-3.2-1B"


def _manifest_fields(run_id: str, n: int) -> dict:
    """A valid spec 00 §2 manifest (only the fields this driver reads matter)."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": "2026-07-30T00:00:00+00:00",
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "regime": "elicit",
        "base_model": {"hf_id": BASE_MODEL, "revision": "main"},
        "task": {"name": "arith_target", "format_version": "1"},
        "dataset": {"name": "d_target", "n_unique_examples": n, "seed": 316},
        "training": {
            "method": "lora",
            "lora": {
                "rank": 64,
                "alpha": 32.0,
                "target_modules": ["q_proj", "v_proj"],
                "dropout": 0.0,
                "sparse_param_count": None,
            },
            "optimizer": {
                "name": "adamw",
                "lr": 3.53e-4,
                "batch_size": 128,
                "micro_batch_size": None,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
                "grad_clip": 1.0,
            },
            "lr_schedule": "constant",
            "min_lr": None,
            "precision": "bf16",
            "eval_every": 5,
            "max_steps": 500,
            "stopping": {"eps_nats": 0.002, "k": 5, "min_steps": 0},
            "epochs_total": 1,
            "seed": 316,
        },
        "trainable_param_count": 4_194_304,
        "snapshot_steps": [],
        "cost": {"gpu_type": "RTX4090", "est_usd": 0.1, "actual_usd": None},
        "status": "complete",
    }


def _write_run(
    store: Path,
    run_id: str,
    n: int,
    *,
    stop_reason: str = "converged",
    final_step: int = 42,
    min_val_nats: float = 0.5,
    edl_epoch1_nats: float | None = 120.0,
    edl_per_label_token_nats: float | None = 0.3,
    edl_per_example_nats: float | None = 1.2,
    zero_shot_accuracy: float | None = 0.87,
    write_test_loss: bool = True,
    test_loss_per_label_token_nats: float = 0.45,
) -> None:
    """Register a run and hand-write the ``experiment.*`` extras this driver reads."""
    fields = _manifest_fields(run_id, n)
    fields["experiment"] = {
        "target_result": {
            "final_step": final_step,
            "stop_reason": stop_reason,
            "best_val_nats": min_val_nats,
            "min_val_nats": min_val_nats,
            "edl_epoch1_nats": edl_epoch1_nats,
            "edl_per_label_token_nats": edl_per_label_token_nats,
            "edl_per_example_nats": edl_per_example_nats,
        },
        "gates": {
            "G5": (
                {"pass": True, "zero_shot_accuracy": zero_shot_accuracy}
                if zero_shot_accuracy is not None
                else {"pass": True}
            )
        },
    }
    register_run(fields, store=store)
    if write_test_loss:
        eval_dir = run_dir(run_id, store=store) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "test_loss.json").write_text(
            json.dumps(
                {
                    "n_test_examples": 2048,
                    "label_token_count": 6000,
                    "loss_sum_nats": test_loss_per_label_token_nats * 6000,
                    "loss_per_label_token_nats": test_loss_per_label_token_nats,
                    "masking_config_hash": "deadbeef",
                }
            )
        )


def test_run_rows_reads_all_expected_metrics(tmp_path: Path) -> None:
    """A fully-populated converged run yields one row per readable metric, correctly tagged."""
    run_id = "evt-llama-fig2-inst-n1000"
    _write_run(tmp_path, run_id, n=1000)

    rows = dss.run_rows(run_id, tmp_path)
    assert rows is not None
    by_metric = {r["metric_name"]: r for r in rows}

    expected = {
        "min_val_nats": 0.5,
        "edl_epoch1_nats": 120.0,
        "edl_per_label_token_nats": 0.3,
        "edl_per_example_nats": 1.2,
        "test_loss_per_label_token_nats": 0.45,
        "g5_zero_shot_em": 0.87,
    }
    assert set(by_metric) == set(expected)
    for name, value in expected.items():
        assert by_metric[name]["metric_value"] == value

    row = rows[0]
    for col in REQUIRED_COLUMNS:
        assert col in row
    assert row["run_id"] == run_id
    assert row["base_model_key"] == BASE_MODEL
    assert row["regime"] == "elicit"
    assert row["dataset_size"] == 1000
    assert row["checkpoint_step"] == 42
    assert row["layer"] == -1
    assert row["condition"] == "inst"
    assert row["curve_label"] == "format-installed"
    assert row["stop_reason"] == "converged"


def test_run_rows_never_fabricates_missing_metrics(tmp_path: Path) -> None:
    """A field genuinely absent (not yet written) is skipped, never emitted as 0/None."""
    run_id = "evt-llama-fig2-noinst-n1000"
    _write_run(
        tmp_path,
        run_id,
        n=1000,
        edl_epoch1_nats=None,
        edl_per_label_token_nats=None,
        edl_per_example_nats=None,
        zero_shot_accuracy=None,
        write_test_loss=False,
    )

    rows = dss.run_rows(run_id, tmp_path)
    assert rows is not None
    metric_names = {r["metric_name"] for r in rows}
    assert metric_names == {"min_val_nats"}
    assert rows[0]["condition"] == "noinst"
    assert rows[0]["curve_label"] == "base"


def test_non_converged_run_still_plots_with_warning(tmp_path, capsys) -> None:
    """stop_reason='max_steps' is a loud WARNING but the run's rows are still returned."""
    run_id = "evt-llama-fig2-noinst-n2154"
    _write_run(tmp_path, run_id, n=2154, stop_reason="max_steps")

    rows = dss.run_rows(run_id, tmp_path)
    out = capsys.readouterr().out

    assert rows is not None and len(rows) > 0
    assert all(r["stop_reason"] == "max_steps" for r in rows)
    assert "WARNING" in out
    assert run_id in out
    assert "max_steps" in out


def test_missing_run_skips_with_warning_not_exception(tmp_path, capsys) -> None:
    """A run never registered (not yet pulled) is skipped, not a crash."""
    run_id = "evt-llama-fig2-inst-n1000000"  # never written under tmp_path

    rows = dss.run_rows(run_id, tmp_path)
    out = capsys.readouterr().out

    assert rows is None
    assert run_id in out


def test_run_incomplete_manifest_skips_with_warning(tmp_path, capsys) -> None:
    """A registered run with no experiment.target_result yet (still training) is skipped."""
    run_id = "evt-llama-fig2-inst-n1000"
    fields = _manifest_fields(run_id, 1000)
    fields["status"] = "running"
    register_run(fields, store=tmp_path)

    rows = dss.run_rows(run_id, tmp_path)
    out = capsys.readouterr().out

    assert rows is None
    assert "WARNING" in out
    assert run_id in out


def test_default_run_ids_match_the_planned_sweep() -> None:
    """The 19-size x 2-condition id list matches the plan's formula and count."""
    ids = dss.default_run_ids()
    assert len(ids) == 38
    assert len(dss.SIZES) == 19
    assert dss.SIZES[0] == 1000
    assert dss.SIZES[-1] == 1000000
    assert "evt-llama-fig2-noinst-n1000" in ids
    assert "evt-llama-fig2-inst-n1000000" in ids


def test_zoo6_write_results_round_trip(tmp_path: Path) -> None:
    """The assembled table round-trips through the ZOO-6/spec 00 §7 results writer."""
    sizes = (1000, 10000)
    rows: list[dict] = []
    for n in sizes:
        for condition in ("noinst", "inst"):
            run_id = f"evt-llama-fig2-{condition}-n{n}"
            _write_run(tmp_path, run_id, n=n)
            rows.extend(dss.run_rows(run_id, tmp_path))

    df = pd.DataFrame(rows)
    from geode.zoo import write_results

    path = write_results(df, "dataset_size_sweep", store=tmp_path)
    assert path.is_file()

    read_back = read_results("dataset_size_sweep", store=tmp_path)
    assert len(read_back) == len(df)
    for col in REQUIRED_COLUMNS:
        assert col in read_back.columns
    assert {"condition", "curve_label", "stop_reason"} <= set(read_back.columns)
    assert set(read_back["condition"]) == {"noinst", "inst"}
    assert set(read_back["dataset_size"]) == set(sizes)


def test_plot_creates_figure_file(tmp_path: Path) -> None:
    """plot() writes a non-empty figure for a small multi-size, multi-condition table."""
    rows: list[dict] = []
    for n in (1000, 10000, 100000):
        for condition, label in dss.CONDITIONS.items():
            run_id = f"evt-llama-fig2-{condition}-n{n}"
            _write_run(
                tmp_path, run_id, n=n, stop_reason="converged" if n != 10000 else "max_steps"
            )
            rows.extend(dss.run_rows(run_id, tmp_path))

    df = pd.DataFrame(rows)
    out = tmp_path / "figures" / "dataset_size_sweep.png"
    dss.plot(df, out)

    assert out.is_file()
    assert out.stat().st_size > 0


def test_manifest_round_trip_unaffected(tmp_path: Path) -> None:
    """Sanity: RunManifest still validates the hand-built experiment.* extras (unknown fields, V0.2)."""
    run_id = "evt-llama-fig2-inst-n1000"
    _write_run(tmp_path, run_id, n=1000)
    manifest = RunManifest.load(run_dir(run_id, store=tmp_path) / "manifest.json")
    manifest.validate()  # unknown "experiment" extra must not break validation
    assert manifest.data["experiment"]["target_result"]["stop_reason"] == "converged"
