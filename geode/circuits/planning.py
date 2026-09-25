"""Transparent workload extrapolation from pilot timing, without launching compute.

Ranges are sensitivity ranges at observed configurations, NOT confidence
intervals or guarantees across GPUs, context lengths, batches, or checkpoints.
Counts always denote actual rendered rows (including ETHICS controls), not
independent statistical groups. Startup and combined circuit costs stay visible.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _seconds(row: Mapping[str, Any]) -> float:
    seconds = float(row["seconds"])
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Timing seconds must be finite and positive")
    return seconds


def estimate_full_run(
    timing_rows: Sequence[Mapping[str, Any]],
    full_counts: Mapping[str, int],
    checkpoint_count: int = 7,
    full_pair_counts: Mapping[str, int | Mapping[str, int]] | None = None,
    full_probe_rows: Mapping[str, int] | None = None,
    hourly_rate: float = 0.0,
) -> dict[str, Any]:
    """Estimate rendered full workloads using only measured component timings.

    ``full_counts`` maps tasks to rendered behavioral rows per checkpoint.
    ``full_pair_counts`` maps tasks to ``{'n_pairs': ..., 'n_intervention_pairs': ...}``.
    An integer pair count requests an attribution-only target (zero interventions),
    and is explicitly recorded as such. ``full_probe_rows`` counts activation rows,
    including answer-only controls only if the pilot n_rows does likewise.

    Combined circuit timing S=nA*cA+nI*cI cannot identify either per-unit cost.
    Conditional on stable per-unit costs, the new combined runtime lies between
    S*min(NA/nA, NI/nI) and S*max(...). An unmeasured required component instead
    makes the upper estimate unknown. Proportional workloads identify one rate.

    Optional timing fields: stage, n_truncated, n_context_overflow, batch_size,
    max_context, max_new_tokens, peak_allocated_gb, gpu_memory_gb. Startup rows
    have component='startup' and scope='checkpoint' (default) or 'run'.
    """
    checkpoint_count = _count(checkpoint_count, "checkpoint_count")
    if checkpoint_count == 0:
        raise ValueError("checkpoint_count must be positive")
    hourly_rate = float(hourly_rate)
    if not np.isfinite(hourly_rate) or hourly_rate < 0:
        raise ValueError("hourly_rate must be finite and nonnegative")
    targets: dict[tuple[str, str], dict[str, int]] = {}
    for task, count in full_counts.items():
        targets[("behavior", task)] = {"n_rows": _count(count, f"behavior/{task}")}
    for task, count in (full_probe_rows or {}).items():
        targets[("probes", task)] = {"n_rows": _count(count, f"probes/{task}")}
    for task, count in (full_pair_counts or {}).items():
        if isinstance(count, Mapping):
            if set(count) != {"n_pairs", "n_intervention_pairs"}:
                raise ValueError("Circuit targets need both n_pairs and n_intervention_pairs")
            targets[("circuits", task)] = {key: _count(value, key) for key, value in count.items()}
        else:
            targets[("circuits", task)] = {
                "n_pairs": _count(count, task),
                "n_intervention_pairs": 0,
            }
    if not targets:
        raise ValueError("At least one full-workload target is required")
    observed: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    startup = []
    for row in timing_rows:
        _seconds(row)
        if row["component"] == "startup":
            startup.append(row)
        elif row["component"] in {"behavior", "circuits", "probes"}:
            observed[(row["component"], row["task"])].append(row)

    components, warnings = [], []
    overall_low, overall_high = 0.0, 0.0
    missing, capped = False, False
    for (component, task), target in sorted(targets.items()):
        item: dict[str, Any] = {"component": component, "task": task, "full_per_checkpoint": target}
        rows = observed.get((component, task), [])
        if not any(target.values()):
            item.update(status="not_requested", projected_seconds_range=[0.0, 0.0])
            components.append(item)
            continue
        valid_rows = []
        for row in rows:
            counts = {key: _count(row.get(key, 0), key) for key in target}
            if any(counts.values()):
                valid_rows.append((row, counts))
        if not valid_rows:
            item.update(
                status="unmeasured",
                projected_seconds_range=[0.0, None],
                reason="No measured nonempty rows for this component/task",
            )
            missing = True
            components.append(item)
            continue
        total_seconds = sum(_seconds(row) for row, _ in valid_rows)
        totals = {key: sum(counts[key] for _, counts in valid_rows) for key in target}
        item.update(
            measured_seconds=total_seconds,
            measured_counts=totals,
            measured_stages=sorted({str(row.get("stage", "unspecified")) for row, _ in valid_rows}),
            effective_throughput_per_second={
                key: value / total_seconds for key, value in totals.items()
            },
            throughput_note="Total work divided by total elapsed component time; combined circuit rates each include both attribution and intervention overhead.",
        )
        ranges, unidentified = [], False
        for row, counts in valid_rows:
            ratios = [target[key] / count for key, count in counts.items() if count > 0]
            uncovered = any(target[key] > 0 and counts[key] == 0 for key in target)
            lower = _seconds(row) * min(ratios)
            upper = None if uncovered else _seconds(row) * max(ratios)
            ranges.append((lower, upper))
            unidentified |= uncovered
        low = min(value[0] for value in ranges) * checkpoint_count
        high = None if unidentified else max(value[1] for value in ranges) * checkpoint_count
        item.update(
            status="partial" if unidentified else "rough_projection",
            projected_seconds_range=[low, high],
            sensitivity="Observed per-stage rate range; circuit mixture adds unidentified relative-operation-cost sensitivity",
        )
        overall_low += low
        if high is None:
            missing = True
        else:
            overall_high += high
        if component != "circuits":
            item["pooled_rate_projected_seconds"] = (
                target["n_rows"] * total_seconds / totals["n_rows"] * checkpoint_count
            )
        elif any(counts.get("n_intervention_pairs", 0) for _, counts in valid_rows):
            item["cost_separation"] = (
                "Attribution and intervention seconds were not timed separately; differing workload ratios widen the range."
            )
        if component == "probes":
            item["scaling_caveat"] = (
                "Combined extraction and classifier fitting may scale superlinearly; a row-linear projection can underestimate larger eigendecompositions."
            )
        if component == "behavior":
            truncation_known = all("n_truncated" in row for row, _ in valid_rows)
            truncations = sum(
                _count(row.get("n_truncated", 0), "n_truncated") for row, _ in valid_rows
            )
            overflow = sum(
                _count(row.get("n_context_overflow", 0), "n_context_overflow")
                for row, _ in valid_rows
            )
            if truncations > totals["n_rows"] or overflow > totals["n_rows"]:
                raise ValueError("Truncation/overflow counts cannot exceed timed rows")
            item["generation"] = {
                "truncation_status": "observed"
                if truncations
                else "none_observed"
                if truncation_known
                else "unknown",
                "n_truncated": truncations if truncation_known else None,
                "n_context_overflow": overflow,
                "n_generated_tokens": sum(
                    _count(row.get("generated_tokens", 0), "generated_tokens")
                    for row, _ in valid_rows
                ),
                "uncapped_runtime_upper_bound": None,
            }
            item["generation"]["effective_generated_tokens_per_second"] = (
                item["generation"]["n_generated_tokens"] / total_seconds
            )
            if truncations or overflow:
                capped = True
                item["generation"]["interpretation"] = (
                    "This projection retains the pilot generation/context caps. It is only a lower-bound planning reference for complete uncapped evaluation, not a measured upper bound."
                )
            if not truncation_known:
                warnings.append(
                    f"{task}: truncation counts were not recorded; completion-adjusted runtime cannot be inferred."
                )
        configurations = [
            {
                key: row[key]
                for key in ("batch_size", "max_context", "max_new_tokens", "gpu", "dtype")
                if key in row
            }
            for row, _ in valid_rows
        ]
        item["observed_configurations"] = configurations
        components.append(item)

    checkpoint_startup = [
        _seconds(row) for row in startup if row.get("scope", "checkpoint") == "checkpoint"
    ]
    run_startup = [_seconds(row) for row in startup if row.get("scope") == "run"]
    if any(row.get("scope", "checkpoint") not in {"checkpoint", "run"} for row in startup):
        raise ValueError("Startup scope must be checkpoint or run")
    if checkpoint_startup:
        startup_range = [
            min(checkpoint_startup) * checkpoint_count + sum(run_startup),
            max(checkpoint_startup) * checkpoint_count + sum(run_startup),
        ]
        overall_low += startup_range[0]
        overall_high += startup_range[1]
        startup_report = {
            "status": "measured",
            "projected_seconds_range": startup_range,
            "n_checkpoint_measurements": len(checkpoint_startup),
        }
    else:
        # Known run-only setup does not imply zero checkpoint loading overhead.
        overall_low += sum(run_startup)
        overall_high += sum(run_startup)
        startup_report = {
            "status": "checkpoint_startup_unmeasured",
            "measured_run_setup_seconds": sum(run_startup),
            "projected_seconds_range": [sum(run_startup), None],
        }
        warnings.append(
            "Checkpoint download/load overhead is unmeasured and excluded from the finite component range."
        )
    memories = [
        (float(row["peak_allocated_gb"]), float(row["gpu_memory_gb"]))
        for row in timing_rows
        if row.get("peak_allocated_gb") is not None and row.get("gpu_memory_gb") is not None
    ]
    recommendation = "Keep the measured GPU/configuration as the cost reference; benchmark a larger batch before projecting its speedup."
    if memories:
        if any(not np.isfinite(used + total) or not 0 < used <= total for used, total in memories):
            raise ValueError(
                "Memory measurements must satisfy 0 < peak_allocated_gb <= gpu_memory_gb"
            )
        utilization = max(used / total for used, total in memories)
        if utilization < 0.6:
            recommendation = "Measured allocated-memory headroom suggests benchmarking 2x forward/generation batch size; keep attribution at the validated batch and verify reserved memory/OOM behavior. No speedup is assumed."
        else:
            recommendation = "Retain the measured batch size initially; group examples by prompt length to reduce padding and measure throughput again before changing GPU rental size."
    hours = [overall_low / 3600, None if missing else overall_high / 3600]
    return {
        "method": "rough observed-configuration workload extrapolation",
        "checkpoint_count": checkpoint_count,
        "components": components,
        "startup": startup_report,
        "projected_hours_range": hours,
        "projected_compute_usd_range": [
            value * hourly_rate if value is not None else None for value in hours
        ],
        "hourly_rate_usd": hourly_rate,
        "status": "partial_unmeasured_workload"
        if missing
        else "capped_lower_bound_reference"
        if capped
        else "rough_projection",
        "runtime_range_is_confidence_interval": False,
        "runtime_range_includes_all_startup": bool(checkpoint_startup),
        "complete_uncapped_runtime_upper_bound": None,
        "transfer_fees_included": False,
        "storage_fee_treatment": "Included only if already part of the supplied hourly rate",
        "cost_efficiency_recommendation": recommendation,
        "warnings": warnings
        + [
            "Scaling assumes the pilot GPU, batch size, dtype, context and generation protocol; performance is not assumed invariant to these settings.",
            "Observed checkpoint throughput variation is not a bound on unseen checkpoints; ranges are planning sensitivity, not guarantees.",
            "Probe classifier fitting and report/statistics CPU time may need separate measurements for larger diagnostic pools.",
            "Upload destinations do not by themselves eliminate provider network egress fees.",
        ],
    }
