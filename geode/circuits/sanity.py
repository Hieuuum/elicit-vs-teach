"""Read-only technical audit of saved checkpoint sanity runs, not capability tests."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import read_jsonl, validate_matched_rows
from .checkpoints import CHECKPOINTS


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _types(nodes: list[str]) -> Counter:
    return Counter("mlp" if node.endswith(".mlp") else "attention" for node in nodes)


def validate_interventions(rows: list[dict], circuit: dict, *, require_typed: bool) -> None:
    """Check disjoint source groups, node selections and all saved effect identities."""
    used = set(circuit["clean_group_ids"]) | set(circuit["corrupt_group_ids"])
    universe = set(circuit["node_names"])
    top = circuit["top16"]
    if len(top) != len(set(top)) or not set(top) <= universe:
        raise ValueError("invalid top-node set")
    for row in rows:
        if {row["clean_group"], row["corrupt_group"]} & used:
            raise ValueError("intervention source overlaps attribution")
        controls = ["random"]
        if require_typed or "random_type_matched" in row:
            controls.append("random_type_matched")
        for control in controls:
            nodes = row[control + "_nodes"]
            if (
                len(nodes) != len(top)
                or len(set(nodes)) != len(nodes)
                or not set(nodes) <= universe
            ):
                raise ValueError("invalid random intervention node set")
            if control == "random_type_matched" and _types(nodes) != _types(top):
                raise ValueError("random intervention node-type counts differ")
        for name in ["top", *controls]:
            patch = row[name]
            values = [patch[key] for key in ("clean_metric", "patched_metric", "effect")]
            if not np.isfinite(values).all():
                raise ValueError("nonfinite intervention metric")
            if not np.isclose(values[1] - values[0], values[2], rtol=1e-6, atol=1e-6):
                raise ValueError("patch effect is not patched-minus-clean")
            if not np.isclose(values[0], row["top"]["clean_metric"], rtol=1e-6, atol=1e-6):
                raise ValueError("intervention comparators use different clean metrics")


def validate_probe_splits(probe: dict, labels: np.ndarray, groups: np.ndarray) -> None:
    """All rows and source groups partition once; held-out labels match the arrays."""
    for layer in probe.get("layers", []):
        splits = layer["split_indices"]
        indices = [int(i) for values in splits.values() for i in values]
        if sorted(indices) != list(range(len(labels))):
            raise ValueError("probe splits do not partition rows")
        group_sets = [set(groups[values].tolist()) for values in splits.values()]
        for i, left in enumerate(group_sets):
            if any(left & right for right in group_sets[i + 1 :]):
                raise ValueError("probe source group crosses splits")
        if labels[splits["test"]].tolist() != layer["test_labels"]:
            raise ValueError("saved test labels do not match probe arrays")


def audit_run(
    root: Path,
    *,
    expected_stages: list[str] | None = None,
    require_typed: bool = True,
) -> dict[str, Any]:
    """Return technical failures and observed coverage separately from accuracy."""
    stages = expected_stages or [checkpoint.stage for checkpoint in CHECKPOINTS]
    failures, warnings, details = [], [], {}
    first_behavior, first_checkpoint = None, None
    first_circuits, first_probes = {}, {}
    metadata = _read(root / "run_metadata.json")
    if metadata.get("status") != "complete":
        failures.append("run metadata is not complete")
    if [cp["stage"] for cp in metadata["config"]["checkpoints"]] != stages:
        failures.append("configured checkpoint scope differs from expected stages")
    tasks = metadata["config"]["tasks"]
    probe_policy = metadata.get("probe_policy_by_stage", {})
    if set(probe_policy) - set(stages) or any(
        v not in {"complete", "skipped"} for v in probe_policy.values()
    ):
        failures.append("invalid probe policy")
    configured = {cp["stage"]: cp for cp in metadata["config"]["checkpoints"]}
    for stage in stages:
        try:
            base = root / stage
            cp = _read(base / "checkpoint.json")
            for key in ("stage", "repo", "revision"):
                if cp[key] != configured[stage][key]:
                    raise ValueError(f"loaded checkpoint {key} differs from pinned configuration")
            if first_checkpoint is not None:
                for key in ("tokenizer_hash", "parameter_count"):
                    if cp[key] != first_checkpoint[key]:
                        raise ValueError(f"checkpoint {key} mismatch")
            first_checkpoint = first_checkpoint or cp
            rows = read_jsonl(base / "behavior.jsonl")
            identities = [(r["id"], r["group"], r["prompt"], r["answer"]) for r in rows]
            if len({r[0] for r in identities}) != len(identities):
                raise ValueError("duplicate behavioral item IDs")
            if first_behavior is not None and identities != first_behavior:
                raise ValueError("behavioral inputs differ across checkpoints")
            first_behavior = first_behavior or identities
            if set(r["task"] for r in rows) != set(tasks):
                raise ValueError("missing behavioral task")
            for row in rows:
                for key in ("answer_log_prob_nats", "logit_margin"):
                    value = row.get(key)
                    if value is not None and not np.isfinite(value):
                        raise ValueError("nonfinite behavioral score")
            stage_info = {
                "behavior_rows": len(rows),
                "truncated": sum(r["truncated"] for r in rows),
                "context_overflow": sum(r["context_overflow"] for r in rows),
                "hardware": _read(base / "hardware.json"),
                "tasks": {},
            }
            for task in tasks:
                circuit = _read(base / "circuits" / (task + ".json"))
                with np.load(base / "circuits" / (task + ".npz")) as saved:
                    scores = saved["scores"]
                if scores.shape != (len(circuit["item_ids"]), len(circuit["node_names"])):
                    raise ValueError(f"{task}: circuit array shape mismatch")
                if not scores.size or not np.isfinite(scores).all():
                    raise ValueError(f"{task}: empty/nonfinite circuit scores")
                if task in first_circuits:
                    validate_matched_rows(first_circuits[task], circuit)
                else:
                    first_circuits[task] = circuit
                interventions = _read(base / "circuits" / (task + "_interventions.json"))
                if not interventions:
                    raise ValueError(f"{task}: no intervention examples")
                validate_interventions(interventions, circuit, require_typed=require_typed)
                if (
                    probe_policy.get(stage) == "skipped"
                    and not (base / "probes" / (task + ".json")).exists()
                ):
                    stage_info["tasks"][task] = {
                        "attribution_rows": len(scores),
                        "intervention_rows": len(interventions),
                        "probe_status": "skipped by user request",
                    }
                    continue
                probe = _read(base / "probes" / (task + ".json"))
                with np.load(base / "probes" / (task + ".npz")) as saved:
                    features, labels, groups = saved["features"], saved["labels"], saved["groups"]
                if len(features) != probe["n_rows"] or not np.isfinite(features).all():
                    raise ValueError(f"{task}: invalid probe feature array")
                identity = (probe["item_ids"], labels.tolist(), groups.tolist())
                if task in first_probes and identity != first_probes[task]:
                    raise ValueError(f"{task}: probe examples/labels differ across checkpoints")
                first_probes.setdefault(task, identity)
                validate_probe_splits(probe, labels, groups)
                if probe["status"] != "complete":
                    warnings.append(f"{stage}/{task}: probe fitting is {probe['status']}")
                stage_info["tasks"][task] = {
                    "attribution_rows": len(scores),
                    "intervention_rows": len(interventions),
                    "probe_rows": len(features),
                    "probe_groups": len(set(groups.tolist())),
                }
            details[stage] = stage_info
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            failures.append(f"{stage}: {type(exc).__name__}: {exc}")
    return {
        "status": "passed" if not failures else "failed",
        "expected_stages": stages,
        "failures": failures,
        "warnings": warnings,
        "checkpoints": details,
        "probe_policy_by_stage": probe_policy,
        "interpretation": "Technical integrity only; truncation, weak accuracy and noisy probes remain scientific limitations.",
    }
