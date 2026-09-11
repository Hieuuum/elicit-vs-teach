"""CPU-only, stage-specific workload projection from seven-checkpoint timings.

This is arithmetic extrapolation, not a confidence interval or a performance
guarantee. Every stage keeps its own measured rate; no checkpoint's timing is
silently substituted for another's. No models or GPU/cloud tools are invoked.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from .artifacts import sha256_file, write_json
from .checkpoints import CHECKPOINTS


def _number(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def project_stage_timings(
    timing_rows: Sequence[Mapping[str, Any]],
    workload: Mapping[str, Any],
    *,
    hourly_rate: float,
    stages: Sequence[str] = tuple(checkpoint.stage for checkpoint in CHECKPOINTS),
    pilot_shuffles: int = 2,
    full_shuffles: int = 5,
    measured_run_elapsed_seconds: float | None = None,
    expected_batch_size: int = 8,
    expected_max_new_tokens: int = 2048,
) -> dict[str, Any]:
    """Sum each checkpoint/task's own rate against exact rendered full counts.

    Circuit attribution and intervention seconds are independently scaled.
    Intervention time already includes top, uniform-random and type-matched
    patches; it must NOT be multiplied by the number of intervention variants.

    Probe point estimates multiply row-linear time by (full_shuffles+1) /
    (pilot_shuffles+1). Applying this fit-count ratio to the entire mixed probe
    phase is a conservative proxy for additional fits, not a measured isolation
    of classifier time, and is NOT an upper bound for larger eigendecompositions.
    The row-linear base is retained separately so the assumption is reviewable.
    """
    hourly_rate = _number(hourly_rate, "hourly_rate")
    pilot_shuffles = _count(pilot_shuffles, "pilot_shuffles")
    full_shuffles = _count(full_shuffles, "full_shuffles")
    if not stages or len(set(stages)) != len(stages):
        raise ValueError("stages must be nonempty and unique")
    tasks = workload.get("tasks", {})
    if not tasks:
        raise ValueError("Exact full workload must contain tasks")
    targets = {}
    for task, record in tasks.items():
        targets[task] = {
            key: _count(record[key], f"{task}/{key}")
            for key in (
                "behavior_rows",
                "attribution_pairs",
                "intervention_pairs",
                "probe_rows",
            )
        }
    indexed = {}
    for row in timing_rows:
        stage = row["stage"]
        if stage not in stages:
            raise ValueError(f"Unexpected timing stage: {stage}")
        component = row["component"]
        if component not in {"startup", "behavior", "circuits", "probes"}:
            raise ValueError(f"Unsupported timing component: {component}")
        if component != "startup" and row.get("task") not in tasks:
            raise ValueError("Timing task is absent from the exact full workload")
        key = (stage, component, row.get("task") if component != "startup" else None)
        if key in indexed:
            raise ValueError(f"Duplicate stage/component/task timing: {key}")
        _number(row["seconds"], "seconds")
        indexed[key] = row
    results, missing = {}, []
    fit_ratio = (full_shuffles + 1) / (pilot_shuffles + 1)
    total_by_component: dict[str, float] = defaultdict(float)
    baseline_by_component: dict[str, float] = defaultdict(float)
    truncated = overflow = 0
    known_generation_rows = unknown_truncation_rows = 0

    def scaling(seconds: float, observed: int, target: int, name: str) -> float | None:
        if target == 0:
            return 0.0
        if observed <= 0:
            missing.append(name + ": no measured positive workload")
            return None
        return seconds * target / observed

    for stage in stages:
        stage_result: dict[str, Any] = {"tasks": {}}
        stage_point = stage_base = 0.0
        startup = indexed.get((stage, "startup", None))
        if startup is None:
            missing.append(stage + "/startup: missing checkpoint download/load timing")
            stage_result["startup_seconds"] = None
        else:
            startup_seconds = _number(startup["seconds"], "startup seconds")
            stage_result["startup_seconds"] = startup_seconds
            stage_point += startup_seconds
            stage_base += startup_seconds
            total_by_component["startup"] += startup_seconds
            baseline_by_component["startup"] += startup_seconds
        for task, target in sorted(targets.items()):
            task_result: dict[str, Any] = {"full_counts": target}
            for component in ("behavior", "circuits", "probes"):
                row = indexed.get((stage, component, task))
                needed = any(
                    target[key]
                    for key in {
                        "behavior": ("behavior_rows",),
                        "circuits": ("attribution_pairs", "intervention_pairs"),
                        "probes": ("probe_rows",),
                    }[component]
                )
                if row is None:
                    if needed:
                        missing.append(f"{stage}/{task}/{component}: missing timing")
                    task_result[component] = {"status": "unmeasured" if needed else "not_requested"}
                    continue
                seconds = _number(row["seconds"], "component seconds")
                item: dict[str, Any] = {"measured_seconds": seconds, "status": "projected"}
                if component in {"behavior", "probes"}:
                    n_rows = _count(row["n_rows"], "n_rows")
                    if n_rows and seconds == 0:
                        raise ValueError("Nonempty workload cannot have zero measured seconds")
                    target_rows = target[
                        "behavior_rows" if component == "behavior" else "probe_rows"
                    ]
                    base = scaling(seconds, n_rows, target_rows, f"{stage}/{task}/{component}")
                    item.update(
                        measured_rows=n_rows,
                        full_rows=target_rows,
                        effective_rows_per_second=n_rows / seconds if seconds else None,
                        row_linear_seconds=base,
                    )
                    if base is None:
                        item.update(status="unmeasured", point_seconds=None)
                        task_result[component] = item
                        continue
                    point = base * fit_ratio if component == "probes" else base
                    item["point_seconds"] = point
                    if component == "probes":
                        item.update(
                            fit_count_proxy_multiplier=fit_ratio,
                            fitting_caveat="Combined extraction/eigendecomposition/fits are not separately timed; larger diagnostic pools can scale superlinearly.",
                        )
                    else:
                        expected = {
                            "batch_size": expected_batch_size,
                            "max_new_tokens": expected_max_new_tokens,
                            "max_context": workload.get("max_context", 4096),
                        }
                        for key, value in expected.items():
                            if row.get(key) != value:
                                raise ValueError(
                                    f"Behavior timing configuration mismatch for {stage}/{task}: {key}"
                                )
                        tokens = _count(row.get("generated_tokens", 0), "generated_tokens")
                        n_truncated = row.get("n_truncated")
                        n_overflow = _count(row.get("n_context_overflow", 0), "n_context_overflow")
                        if n_truncated is None:
                            unknown_truncation_rows += n_rows
                        else:
                            n_truncated = _count(n_truncated, "n_truncated")
                            if n_truncated > n_rows:
                                raise ValueError("Truncated row count exceeds behavioral rows")
                            truncated += n_truncated
                            known_generation_rows += n_rows
                        if n_overflow > n_rows:
                            raise ValueError("Context-overflow count exceeds behavioral rows")
                        overflow += n_overflow
                        item["generation"] = {
                            "tokens": tokens,
                            "tokens_per_second_including_scoring": tokens / seconds
                            if seconds
                            else None,
                            "n_truncated": n_truncated,
                            "truncation_rate": n_truncated / n_rows
                            if n_truncated is not None and n_rows
                            else None,
                            "n_context_overflow": n_overflow,
                        }
                        item["configuration"] = {
                            key: row[key]
                            for key in (
                                "batch_size",
                                "max_context",
                                "max_new_tokens",
                                "generation_seconds",
                                "reference_scoring_seconds",
                            )
                            if key in row
                        }
                    total_by_component[component] += point
                    baseline_by_component[component] += base
                    stage_point += point
                    stage_base += base
                else:
                    if not {"attribution_seconds", "intervention_seconds"}.issubset(row):
                        missing.append(
                            f"{stage}/{task}/circuits: separate attribution/intervention timing missing"
                        )
                        item.update(status="unmeasured", point_seconds=None)
                        task_result[component] = item
                        continue
                    attribution = _number(row["attribution_seconds"], "attribution_seconds")
                    intervention = _number(row["intervention_seconds"], "intervention_seconds")
                    if attribution + intervention > seconds + 0.001:
                        raise ValueError("Circuit subcomponent seconds exceed total component time")
                    n_pairs = _count(row["n_pairs"], "n_pairs")
                    n_interventions = _count(row["n_intervention_pairs"], "n_intervention_pairs")
                    if (n_pairs and attribution == 0) or (n_interventions and intervention == 0):
                        raise ValueError(
                            "Nonempty circuit workload cannot have zero subcomponent seconds"
                        )
                    if target["intervention_pairs"] and set(
                        row.get("intervention_controls", [])
                    ) != {"uniform", "type_matched"}:
                        raise ValueError(
                            "Intervention timing does not match top+uniform+type-matched full protocol"
                        )
                    point_a = scaling(
                        attribution,
                        n_pairs,
                        target["attribution_pairs"],
                        f"{stage}/{task}/attribution",
                    )
                    point_i = scaling(
                        intervention,
                        n_interventions,
                        target["intervention_pairs"],
                        f"{stage}/{task}/interventions",
                    )
                    overhead = max(0.0, seconds - attribution - intervention)
                    item.update(
                        measured_pairs=n_pairs,
                        measured_intervention_pairs=n_interventions,
                        attribution_seconds=point_a,
                        intervention_seconds=point_i,
                        fixed_overhead_proxy_seconds=overhead,
                        measured_attribution_pairs_per_second=n_pairs / attribution
                        if attribution
                        else None,
                        measured_intervention_pairs_per_second=n_interventions / intervention
                        if intervention
                        else None,
                        intervention_protocol="One measured intervention pair includes top, uniform-random and node-type-matched patches; no extra multiplicative factor",
                    )
                    if point_a is None or point_i is None:
                        item.update(status="unmeasured", point_seconds=None)
                    else:
                        point = point_a + point_i + overhead
                        item["point_seconds"] = point
                        stage_point += point
                        stage_base += point
                        for key, value in (
                            ("attribution", point_a),
                            ("interventions", point_i),
                            ("circuit_overhead", overhead),
                        ):
                            total_by_component[key] += value
                            baseline_by_component[key] += value
                task_result[component] = item
            stage_result["tasks"][task] = task_result
        stage_result.update(point_seconds=stage_point, row_linear_probe_base_seconds=stage_base)
        results[stage] = stage_result
    observed_component_seconds = sum(float(row["seconds"]) for row in timing_rows)
    residual = None
    if measured_run_elapsed_seconds is not None:
        elapsed = _number(measured_run_elapsed_seconds, "measured_run_elapsed_seconds")
        if elapsed + 1 < observed_component_seconds:
            raise ValueError("Measured component timings exceed run wall time")
        residual = max(0.0, elapsed - observed_component_seconds)
    point_seconds = sum(total_by_component.values())
    base_seconds = sum(baseline_by_component.values())
    return {
        "status": "partial" if missing else "complete_projection",
        "method": "Sum every measured stage/task rate separately using exact rendered full workload counts",
        "stages": results,
        "missing_measurements": missing,
        "point_hours": point_seconds / 3600 if not missing else None,
        "point_compute_usd": point_seconds / 3600 * hourly_rate if not missing else None,
        "measured_components_only_hours": point_seconds / 3600,
        "row_linear_probe_base_hours": base_seconds / 3600,
        "component_hours": {key: value / 3600 for key, value in total_by_component.items()},
        "hourly_rate_usd": hourly_rate,
        "generation": {
            "n_truncated": truncated,
            "n_context_overflow": overflow,
            "rows_with_truncation_recorded": known_generation_rows,
            "rows_with_truncation_unknown": unknown_truncation_rows,
            "interpretation": "Projection holds observed generation caps/batch/context fixed; it is not an estimate of uncapped completion time",
        },
        "overhead": {
            "measured_startup_included": all(
                result["startup_seconds"] is not None for result in results.values()
            ),
            "observed_unattributed_run_seconds": residual,
            "full_workload_preparation_cpu_seconds": workload.get("elapsed_seconds"),
            "unmeasured_full_run_overhead": "Full plan preparation, larger-scale I/O, final CPU statistics/reporting, transfer time and nonlinear probe fitting are not silently added as measured GPU work",
            "recommendation": "Prepare the full frozen plan before rental and perform final CPU statistics/reporting after releasing the GPU",
        },
        "probe_assumption": {
            "pilot_shuffles": pilot_shuffles,
            "full_shuffles": full_shuffles,
            "fit_count_proxy_multiplier": fit_ratio,
            "interpretation": "Point estimate applies the fit-count ratio to the entire probe phase as a conservative extra-fit proxy; row-linear base is also reported. Neither is an upper bound on nonlinear classifier fitting",
        },
        "is_confidence_interval": False,
        "caveats": [
            "No stage rate is borrowed from a different checkpoint; changing batch/context/dtype/hardware requires new measurements.",
            "Full-data prompt-length and generation-length distributions can differ from the small sanity sample.",
            "Circuit overhead is held fixed once per stage/task; serialization can grow with larger arrays.",
            "Intervention seconds already include all three patch variants, and ETHICS counts already include control renderings.",
            "Transfer fees are excluded; storage is included only when already part of the supplied hourly rate.",
        ],
    }


def build_projection(
    run_dir: str | Path,
    workload_path: str | Path,
    *,
    hourly_rate: float | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Read completed local backups and write full_run_projection.json/.md."""
    root, full_path = Path(run_dir), Path(workload_path)
    workload = json.loads(full_path.read_text())
    metadata_path = root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if hourly_rate is None:
        hourly_rate = metadata.get("config", {}).get("hourly_rate")
    if hourly_rate is None:
        raise ValueError("Supply hourly_rate or a saved run_metadata.json rate")
    timing_rows, sources = [], []
    for checkpoint in CHECKPOINTS:
        path = root / checkpoint.stage / "timing.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text()):
            if row.get("stage", checkpoint.stage) != checkpoint.stage:
                raise ValueError(f"Timing stage disagrees with its directory: {path}")
            timing_rows.append({**row, "stage": checkpoint.stage})
        sources.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
    report = project_stage_timings(
        timing_rows,
        workload,
        hourly_rate=hourly_rate,
        measured_run_elapsed_seconds=metadata.get("elapsed_seconds"),
    )
    if report["status"] == "partial" and not allow_partial:
        raise ValueError(
            "Seven-stage timing backup is incomplete: "
            + "; ".join(report["missing_measurements"][:5])
        )
    report["provenance"] = {
        "timing_files": sources,
        "full_workload_path": str(full_path),
        "full_workload_sha256": sha256_file(full_path),
        "projection_module_sha256": sha256_file(Path(__file__)),
    }
    write_json(root / "full_run_projection.json", report)
    lines = [
        f"Seven-stage point projection: {report['point_hours']:.2f} GPU rental-hours, ${report['point_compute_usd']:.2f} at ${hourly_rate:.4f}/hour."
        if report["point_hours"] is not None
        else "Projection is incomplete; unmeasured work is not assigned a guessed rate.",
        "",
        "These are workload extrapolations, not confidence intervals. Each checkpoint uses its own measured rate.",
        "",
        "| Component | Projected hours |",
        "|---|---:|",
        *[f"| {name} | {value:.3f} |" for name, value in report["component_hours"].items()],
        "",
        f"Without the conservative 2-to-5 shuffle fit-count proxy, the row-linear probe base gives {report['row_linear_probe_base_hours']:.2f} hours; nonlinear probe fitting remains unmeasured.",
        "",
        f"Recorded sanity truncations: {report['generation']['n_truncated']}; context overflows: {report['generation']['n_context_overflow']}. Full generation retains the measured cap, so no uncapped completion-time bound is claimed.",
        "",
        f"Observed sanity wall time outside component timers: {report['overhead']['observed_unattributed_run_seconds']} seconds; this is reported separately, not treated as a full-run overhead estimate.",
        "",
        report["overhead"]["unmeasured_full_run_overhead"] + ".",
        "",
        report["overhead"]["recommendation"] + ".",
    ]
    (root / "full_run_projection.md").write_text("\n".join(lines) + "\n")
    return report
