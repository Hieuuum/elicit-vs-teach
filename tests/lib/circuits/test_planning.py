"""Runtime planning properties: honest workload mixing, caps, and missing timing."""

import json

import pytest

from geode.circuits.planning import estimate_full_run


def test_pooled_throughput_is_ratio_of_sums_not_mean_of_rates():
    rows = [
        {"component": "behavior", "task": "math", "n_rows": 10, "seconds": 10, "stage": "base"},
        {"component": "behavior", "task": "math", "n_rows": 90, "seconds": 30, "stage": "sft"},
    ]
    result = estimate_full_run(rows, {"math": 1000}, checkpoint_count=2, hourly_rate=1)
    component = result["components"][0]
    assert component["effective_throughput_per_second"]["n_rows"] == 2.5
    assert component["pooled_rate_projected_seconds"] == 800
    assert component["projected_seconds_range"] == pytest.approx([2000 / 3, 2000])


def test_ethics_counts_are_already_rendered_no_hidden_control_multiplier():
    rows = [{"component": "behavior", "task": "ethics", "n_rows": 90, "seconds": 9}]
    result = estimate_full_run(rows, {"ethics": 900}, checkpoint_count=1)
    assert result["components"][0]["projected_seconds_range"] == [90, 90]


def test_combined_circuit_cost_respects_unidentified_operation_mix():
    rows = [
        {
            "component": "circuits",
            "task": "code",
            "n_pairs": 8,
            "n_intervention_pairs": 4,
            "seconds": 120,
        }
    ]
    result = estimate_full_run(
        rows,
        {},
        checkpoint_count=1,
        full_pair_counts={"code": {"n_pairs": 400, "n_intervention_pairs": 128}},
    )
    component = result["components"][0]
    assert component["projected_seconds_range"] == [120 * 32, 120 * 50]
    assert "pooled_rate_projected_seconds" not in component


def test_proportional_circuit_mix_has_single_identifiable_scaling_factor():
    rows = [
        {
            "component": "circuits",
            "task": "code",
            "n_pairs": 8,
            "n_intervention_pairs": 4,
            "seconds": 120,
        }
    ]
    result = estimate_full_run(
        rows,
        {},
        checkpoint_count=7,
        full_pair_counts={"code": {"n_pairs": 80, "n_intervention_pairs": 40}},
    )
    assert result["components"][0]["projected_seconds_range"] == [8400, 8400]


def test_unmeasured_required_interventions_do_not_get_a_fabricated_upper_bound():
    rows = [
        {
            "component": "circuits",
            "task": "code",
            "n_pairs": 8,
            "n_intervention_pairs": 0,
            "seconds": 120,
        }
    ]
    result = estimate_full_run(
        rows,
        {},
        checkpoint_count=1,
        full_pair_counts={"code": {"n_pairs": 80, "n_intervention_pairs": 40}},
    )
    assert result["components"][0]["projected_seconds_range"] == [1200, None]
    assert result["projected_hours_range"][1] is None
    assert result["status"] == "partial_unmeasured_workload"


def test_truncated_generations_make_uncapped_runtime_only_lower_bound_reference():
    rows = [
        {
            "component": "behavior",
            "task": "math",
            "n_rows": 10,
            "seconds": 100,
            "n_truncated": 3,
            "generated_tokens": 1000,
        }
    ]
    result = estimate_full_run(rows, {"math": 100}, checkpoint_count=1)
    assert result["status"] == "capped_lower_bound_reference"
    assert result["complete_uncapped_runtime_upper_bound"] is None
    assert not result["runtime_range_is_confidence_interval"]
    assert result["components"][0]["generation"]["effective_generated_tokens_per_second"] == 10


def test_startup_is_separate_and_scales_per_checkpoint_not_per_example():
    rows = [
        {"component": "behavior", "task": "math", "n_rows": 10, "seconds": 10},
        {"component": "startup", "seconds": 20, "scope": "checkpoint"},
        {"component": "startup", "seconds": 5, "scope": "run"},
    ]
    result = estimate_full_run(rows, {"math": 100}, checkpoint_count=2, hourly_rate=0.6)
    assert result["startup"]["projected_seconds_range"] == [45, 45]
    assert result["projected_hours_range"] == pytest.approx([245 / 3600] * 2)
    assert result["projected_compute_usd_range"] == pytest.approx([245 / 3600 * 0.6] * 2)


def test_missing_task_is_explicit_and_unknown_startup_is_not_zero():
    result = estimate_full_run([], {"math": 100}, checkpoint_count=1)
    assert result["status"] == "partial_unmeasured_workload"
    assert result["startup"]["status"] == "checkpoint_startup_unmeasured"
    assert result["projected_hours_range"] == [0, None]
    json.dumps(result, allow_nan=False)


def test_batch_headroom_recommends_a_measurement_not_assumed_speedup():
    rows = [
        {
            "component": "behavior",
            "task": "math",
            "n_rows": 10,
            "seconds": 100,
            "peak_allocated_gb": 10,
            "gpu_memory_gb": 48,
            "batch_size": 8,
        }
    ]
    result = estimate_full_run(rows, {"math": 100}, checkpoint_count=1)
    assert "benchmarking 2x" in result["cost_efficiency_recommendation"]
    assert "No speedup is assumed" in result["cost_efficiency_recommendation"]
    assert result["projected_hours_range"] == pytest.approx([1000 / 3600] * 2)


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_invalid_workload_counts_fail_loudly(bad):
    with pytest.raises(ValueError):
        estimate_full_run([], {"math": bad})


def test_probe_fitting_nonlinearity_is_not_hidden():
    rows = [{"component": "probes", "task": "math", "n_rows": 100, "seconds": 2}]
    result = estimate_full_run(rows, {}, full_probe_rows={"math": 1000})
    assert "superlinearly" in result["components"][0]["scaling_caveat"]
