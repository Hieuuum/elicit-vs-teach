"""Arithmetic smoke/property fixtures for stage-specific full-run projection."""

import copy
import json

import pytest

from geode.circuits.checkpoints import CHECKPOINTS
from geode.circuits.stage_projection import build_projection, project_stage_timings


def fixture(stage="init", multiplier=1):
    return [
        {"stage": stage, "component": "startup", "seconds": 7 * multiplier},
        {
            "stage": stage,
            "component": "behavior",
            "task": "math",
            "n_rows": 10,
            "seconds": 20 * multiplier,
            "generated_tokens": 1000,
            "n_truncated": 1,
            "n_context_overflow": 0,
            "batch_size": 8,
            "max_new_tokens": 2048,
            "max_context": 4096,
        },
        {
            "stage": stage,
            "component": "circuits",
            "task": "math",
            "n_pairs": 8,
            "n_intervention_pairs": 4,
            "attribution_seconds": 40 * multiplier,
            "intervention_seconds": 60 * multiplier,
            "seconds": 103 * multiplier,
            "intervention_controls": ["uniform", "type_matched"],
        },
        {
            "stage": stage,
            "component": "probes",
            "task": "math",
            "n_rows": 40,
            "seconds": 20 * multiplier,
        },
    ]


def workload():
    return {
        "max_context": 4096,
        "elapsed_seconds": 244,
        "tasks": {
            "math": {
                "behavior_rows": 100,
                "attribution_pairs": 80,
                "intervention_pairs": 8,
                "probe_rows": 80,
            }
        },
    }


def test_seven_stage_projection_sums_own_rates_instead_of_reusing_a_mean_or_extreme():
    stages = tuple(f"stage-{i}" for i in range(7))
    rows = [row for i, stage in enumerate(stages) for row in fixture(stage, i + 1)]
    result = project_stage_timings(rows, workload(), hourly_rate=0.5, stages=stages)
    # One multiplier unit: behavior200 + attr400 + interventions120 + probes80
    # + overhead3 + startup7 = 810 seconds. Stage multipliers sum to28.
    assert result["point_hours"] == pytest.approx(810 * 28 / 3600)
    assert result["point_compute_usd"] == pytest.approx(810 * 28 / 3600 * 0.5)
    assert result["stages"]["stage-6"]["point_seconds"] == 810 * 7
    assert not result["is_confidence_interval"]


def test_attribution_and_interventions_scale_separately_without_extra_threefold_factor():
    result = project_stage_timings(fixture(), workload(), hourly_rate=1, stages=["init"])
    circuits = result["stages"]["init"]["tasks"]["math"]["circuits"]
    assert circuits["attribution_seconds"] == 400
    assert circuits["intervention_seconds"] == 120  # measured time already includes 3 patches
    assert circuits["fixed_overhead_proxy_seconds"] == 3
    assert circuits["point_seconds"] == 523


def test_probe_extra_shuffles_proxy_is_explicit_and_base_is_preserved():
    result = project_stage_timings(fixture(), workload(), hourly_rate=1, stages=["init"])
    probe = result["stages"]["init"]["tasks"]["math"]["probes"]
    assert probe["row_linear_seconds"] == 40
    assert probe["fit_count_proxy_multiplier"] == 2
    assert probe["point_seconds"] == 80
    assert result["row_linear_probe_base_hours"] == pytest.approx((810 - 40) / 3600)
    assert "Neither is an upper bound" in result["probe_assumption"]["interpretation"]


def test_observed_unattributed_overhead_stays_separate_from_full_projection():
    rows = fixture()
    elapsed = sum(row["seconds"] for row in rows) + 50
    result = project_stage_timings(
        rows, workload(), hourly_rate=1, stages=["init"], measured_run_elapsed_seconds=elapsed
    )
    assert result["overhead"]["observed_unattributed_run_seconds"] == 50
    assert result["point_hours"] == pytest.approx(810 / 3600)
    assert result["overhead"]["full_workload_preparation_cpu_seconds"] == 244


def test_truncations_are_reported_with_no_uncapped_runtime_claim():
    result = project_stage_timings(fixture(), workload(), hourly_rate=1, stages=["init"])
    behavior = result["stages"]["init"]["tasks"]["math"]["behavior"]
    assert behavior["generation"]["truncation_rate"] == 0.1
    assert behavior["generation"]["tokens_per_second_including_scoring"] == 50
    assert result["generation"]["n_truncated"] == 1
    assert "not an estimate of uncapped" in result["generation"]["interpretation"]


def test_missing_stage_never_borrows_a_different_checkpoint_rate():
    result = project_stage_timings(fixture(), workload(), hourly_rate=1, stages=["init", "sft"])
    assert result["status"] == "partial" and result["point_hours"] is None
    assert any("sft" in reason for reason in result["missing_measurements"])


def test_old_combined_circuit_timing_does_not_produce_a_fabricated_split():
    rows = fixture()
    rows[2].pop("attribution_seconds")
    rows[2].pop("intervention_seconds")
    result = project_stage_timings(rows, workload(), hourly_rate=1, stages=["init"])
    assert result["status"] == "partial" and result["point_hours"] is None


@pytest.mark.parametrize("mutation", ["seconds", "batch", "cap", "controls", "duplicate"])
def test_mismatched_or_double_counted_timings_fail_loudly(mutation):
    rows = copy.deepcopy(fixture())
    if mutation == "seconds":
        rows[2]["attribution_seconds"] = 200
    elif mutation == "batch":
        rows[1]["batch_size"] = 16
    elif mutation == "cap":
        rows[1]["max_new_tokens"] = 512
    elif mutation == "controls":
        rows[2]["intervention_controls"] = ["uniform"]
    else:
        rows.append(copy.deepcopy(rows[1]))
    with pytest.raises(ValueError):
        project_stage_timings(rows, workload(), hourly_rate=1, stages=["init"])


def test_completed_backup_round_trip_writes_projection_and_hashes(tmp_path):
    elapsed = 0
    for checkpoint in CHECKPOINTS:
        folder = tmp_path / checkpoint.stage
        folder.mkdir()
        rows = fixture(checkpoint.stage)
        elapsed += sum(row["seconds"] for row in rows)
        (folder / "timing.json").write_text(json.dumps(rows))
    full_path = tmp_path / "workload.json"
    full_path.write_text(json.dumps(workload()))
    (tmp_path / "run_metadata.json").write_text(
        json.dumps({"config": {"hourly_rate": 0.5}, "elapsed_seconds": elapsed + 12})
    )
    result = build_projection(tmp_path, full_path)
    assert result["point_hours"] == pytest.approx(810 * 7 / 3600)
    assert result["overhead"]["observed_unattributed_run_seconds"] == 12
    assert len(result["provenance"]["timing_files"]) == 7
    assert len(result["provenance"]["full_workload_sha256"]) == 64
    assert (
        json.loads((tmp_path / "full_run_projection.json").read_text())["status"]
        == "complete_projection"
    )
    assert "not confidence intervals" in (tmp_path / "full_run_projection.md").read_text()
