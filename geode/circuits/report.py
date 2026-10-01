"""Offline reports from saved, matched circuit-evaluation artifacts.

No model execution or network access occurs here. Missing/insufficient analyses
remain explicit inconclusive entries; failed or truncated generations remain
in the behavioral denominator.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import json

import numpy as np

from .artifacts import fingerprint, read_jsonl, sha256_file, validate_matched_rows, write_json
from .checkpoints import CHECKPOINTS
from .data import native_aggregate
from .statistics import (
    node_type_counts,
    node_type_jaccard_baseline,
    paired_metric_ci,
    paired_overlap_ci,
    random_jaccard_baseline,
    split_half_stability,
)

ETHICS_TASKS = (
    "ethics_commonsense",
    "ethics_deontology",
    "ethics_justice",
    "ethics_virtue",
    "ethics_utilitarianism",
)


def _key(row: dict) -> str:
    return f"{row['task']}/{row['metadata']['split']}/{row['metadata'].get('variant', 'original')}"


def _group_values(rows: list[dict], field: str) -> tuple[np.ndarray, list[str]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            if not np.isfinite(float(value)):
                raise ValueError(f"Non-finite {field} for {row['id']}")
            groups[row["group"]].append(float(value))
    names = sorted(groups)
    exact = field == "correct" and rows[0]["task"] in {
        "ethics_justice",
        "ethics_deontology",
        "ethics_virtue",
    }
    return np.array([float(all(groups[g])) if exact else np.mean(groups[g]) for g in names]), names


def _behavior_summary(rows: list[dict], n_bootstrap: int, seed: int) -> dict:
    result = native_aggregate(rows)
    by_key: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_key[_key(row)].append(row)
    for key, selected in by_key.items():
        item = result[key]
        item["truncation_rate"] = sum(bool(r.get("truncated")) for r in selected) / len(selected)
        item["context_overflow_rate"] = sum(
            bool(r.get("context_overflow")) for r in selected
        ) / len(selected)
        for field in ("correct", "answer_log_prob_nats", "logit_margin"):
            values, groups = _group_values(selected, field)
            name = "accuracy_uncertainty" if field == "correct" else field
            valid = [float(r[field]) for r in selected if r.get(field) is not None]
            if not valid:
                item[name] = {"status": "unavailable", "n_scored": 0}
                continue
            diagnostics: dict[str, Any] = {
                "mean": float(np.mean(valid)),
                "n_scored": len(valid),
                "n_missing": len(selected) - len(valid),
                "n_groups": len(groups),
            }
            try:
                ci = paired_metric_ci(values, values, groups, n_bootstrap=n_bootstrap, seed=seed)
                diagnostics.update(status="ok", group_weighted_mean=ci["mean_a"], ci=ci["ci_a"])
            except ValueError as exc:
                diagnostics.update(status="inconclusive", reason=str(exc))
            item[name] = diagnostics
    # Equal-task macro uncertainty uses native group scores and stratified draws.
    for key, item in result.items():
        if not key.startswith("ethics_macro/"):
            continue
        _, split, variant = key.split("/")
        values, groups, strata = [], [], []
        for subkey, selected in by_key.items():
            task, task_split, task_variant = subkey.split("/")
            if task.startswith("ethics_") and (task_split, task_variant) == (split, variant):
                task_values, task_groups = _group_values(selected, "correct")
                values.extend(task_values)
                groups.extend(f"{task}/{g}" for g in task_groups)
                strata.extend([task] * len(task_groups))
        try:
            ci = paired_metric_ci(
                np.array(values),
                np.array(values),
                groups,
                strata=strata,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            item["accuracy_uncertainty"] = {
                "status": "ok",
                "ci": ci["ci_a"],
                "group_weighted_mean": ci["mean_a"],
            }
        except ValueError as exc:
            item["accuracy_uncertainty"] = {"status": "inconclusive", "reason": str(exc)}
    # Average native scores over complete control factorials BEFORE resampling
    # source groups; the eight renderings are never independent observations.
    balanced = {}
    source_values = {}
    task_splits = sorted(
        {
            (row["task"], row["metadata"]["split"])
            for row in rows
            if row["task"].startswith("ethics_")
        }
    )
    for task, split in task_splits:
        keys = sorted(key for key in by_key if key.startswith(f"{task}/{split}/control_"))
        if len(keys) != 8:
            continue
        variants = [_group_values(by_key[key], "correct") for key in keys]
        groups = variants[0][1]
        if any(group_ids != groups for _, group_ids in variants):
            raise ValueError("ETHICS control variants have unmatched independent source groups")
        values = np.mean([values for values, _ in variants], axis=0)
        item = {
            "accuracy": float(values.mean()),
            "n_groups": len(groups),
            "n_variants": 8,
            "metric": "mean native score across eight balanced label/position renderings",
            "variant_accuracies": [result[key]["accuracy"] for key in keys],
        }
        try:
            ci = paired_metric_ci(values, values, groups, n_bootstrap=n_bootstrap, seed=seed)
            item["accuracy_uncertainty"] = {
                "status": "ok",
                "ci": ci["ci_a"],
                "group_weighted_mean": ci["mean_a"],
            }
        except ValueError as exc:
            item["accuracy_uncertainty"] = {"status": "inconclusive", "reason": str(exc)}
        balanced[f"{task}/{split}/balanced_controls"] = item
        source_values[(task, split)] = (values, groups)
    for split in sorted({split for _, split in source_values}):
        if not all((task, split) in source_values for task in ETHICS_TASKS):
            continue
        values, groups, strata = [], [], []
        for task in ETHICS_TASKS:
            task_values, task_groups = source_values[(task, split)]
            values.extend(task_values)
            groups.extend(f"{task}/{group}" for group in task_groups)
            strata.extend([task] * len(task_values))
        item = {
            "accuracy": float(
                np.mean(
                    [
                        balanced[f"{task}/{split}/balanced_controls"]["accuracy"]
                        for task in ETHICS_TASKS
                    ]
                )
            ),
            "n_tasks": 5,
            "n_variants": 8,
            "variant_accuracies": np.mean(
                [
                    balanced[f"{task}/{split}/balanced_controls"]["variant_accuracies"]
                    for task in ETHICS_TASKS
                ],
                axis=0,
            ).tolist(),
        }
        try:
            ci = paired_metric_ci(
                np.array(values),
                np.array(values),
                groups,
                strata=strata,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            item["accuracy_uncertainty"] = {
                "status": "ok",
                "ci": ci["ci_a"],
                "group_weighted_mean": ci["mean_a"],
            }
        except ValueError as exc:
            item["accuracy_uncertainty"] = {"status": "inconclusive", "reason": str(exc)}
        balanced[f"ethics_macro/{split}/balanced_controls"] = item
    result.update(balanced)
    return result


def _paired_behavior(left: list[dict], right: list[dict], n_bootstrap: int, seed: int) -> dict:
    by_stage = []
    for records in (left, right):
        mapping: dict[str, list[dict]] = defaultdict(list)
        for row in records:
            mapping[_key(row)].append(row)
        by_stage.append(mapping)
    results = {}
    for key in sorted(set(by_stage[0]) | set(by_stage[1])):
        a, b = (sorted(mapping.get(key, []), key=lambda row: row["id"]) for mapping in by_stage)
        if not a or not b:
            results[key] = {"status": "inconclusive", "reason": "task missing from one checkpoint"}
            continue
        # Match semantic and rendered inputs, never predictions or correctness.
        fields = (
            "id",
            "task",
            "group",
            "prompt",
            "answer",
            "options",
            "label",
            "metadata",
            "diagnostic_prompt",
            "diagnostic_answer",
            "scoring_protocol",
        )
        if len(a) != len(b) or any(
            any(x.get(f) != y.get(f) for f in fields) for x, y in zip(a, b, strict=True)
        ):
            results[key] = {
                "status": "inconclusive",
                "reason": "unmatched behavioral example IDs or protocol",
            }
            continue
        field_results = {}
        for field in ("correct", "answer_log_prob_nats", "logit_margin"):
            # Confidence comparisons use the identical finite-scored rows at both stages.
            pairs = [
                (x, y)
                for x, y in zip(a, b, strict=True)
                if x.get(field) is not None and y.get(field) is not None
            ]
            if not pairs:
                field_results[field] = {"status": "unavailable", "n_paired_scored": 0}
                continue
            va, ga = _group_values([p[0] for p in pairs], field)
            vb, gb = _group_values([p[1] for p in pairs], field)
            if ga != gb:
                raise ValueError("Internal paired behavioral group mismatch")
            try:
                result = paired_metric_ci(va, vb, ga, n_bootstrap=n_bootstrap, seed=seed)
                result.update(status="ok", n_paired_scored=len(pairs))
            except ValueError as exc:
                result = {
                    "status": "inconclusive",
                    "reason": str(exc),
                    "n_paired_scored": len(pairs),
                }
            field_results[field] = result
        results[key] = field_results
    return results


def _load_circuits(stage_dir: Path) -> dict:
    result = {}
    for path in sorted((stage_dir / "circuits").glob("*.npz")):
        metadata_path = path.with_suffix(".json")
        if not metadata_path.exists():
            result[path.stem] = {"status": "inconclusive", "reason": "missing circuit metadata"}
            continue
        meta = json.loads(metadata_path.read_text())
        with np.load(path, allow_pickle=False) as data:
            scores = np.asarray(data["scores"], dtype=float)
        if scores.ndim != 2 or len(scores) != len(meta.get("item_ids", [])):
            raise ValueError(f"Circuit score/metadata shape mismatch: {path}")
        if len(set(meta["item_ids"])) != len(meta["item_ids"]):
            raise ValueError(f"Duplicate circuit pair IDs: {path}")
        result[path.stem] = {"status": "loaded", "scores": scores, "metadata": meta}
    if any(task in result for task in ETHICS_TASKS):
        missing = [task for task in ETHICS_TASKS if result.get(task, {}).get("status") != "loaded"]
        if missing:
            result["ethics_domain"] = {
                "status": "inconclusive",
                "reason": "Full five-task ETHICS domain map unavailable; missing: "
                + ", ".join(missing),
            }
        else:
            items = [result[task] for task in ETHICS_TASKS]
            nodes = items[0]["metadata"]["node_names"]
            token_hash = items[0]["metadata"]["tokenizer_hash"]
            if any(
                item["metadata"]["node_names"] != nodes
                or item["metadata"]["tokenizer_hash"] != token_hash
                for item in items
            ):
                raise ValueError("ETHICS task circuit node universes or tokenizers differ")
            meta = {
                "node_names": nodes,
                "tokenizer_hash": token_hash,
                "item_ids": [
                    f"{task}/{item_id}"
                    for task, item in zip(ETHICS_TASKS, items, strict=True)
                    for item_id in item["metadata"]["item_ids"]
                ],
                "group_ids": [
                    f"{task}/{group}"
                    for task, item in zip(ETHICS_TASKS, items, strict=True)
                    for group in item["metadata"]["group_ids"]
                ],
                "strata": [
                    task
                    for task, item in zip(ETHICS_TASKS, items, strict=True)
                    for _ in item["metadata"]["item_ids"]
                ],
                "protocol_hash": fingerprint(
                    [
                        (task, item["metadata"]["protocol_hash"])
                        for task, item in zip(ETHICS_TASKS, items, strict=True)
                    ]
                ),
            }
            result["ethics_domain"] = {
                "status": "loaded",
                "metadata": meta,
                "scores": np.concatenate([item["scores"] for item in items]),
            }
    return result


def _type_matched_intervention_summary(
    rows: list[dict], metadata: dict, groups: list[str], n_bootstrap: int, seed: int
) -> dict:
    """Validate and summarize an optional paired node-type-matched comparator."""
    summary: dict[str, Any] = {
        "status": "unavailable",
        "reason": "No node-type-matched random interventions were saved",
        "matching": "set size and exact MLP/head/other counts of the top-node set",
        "interpretation": "Stronger disruption than this control supports causal relevance, not proof of acquisition or a newly formed circuit.",
    }
    if not any("random_type_matched" in row for row in rows):
        return summary
    try:
        if not all("random_type_matched" in row for row in rows):
            raise ValueError("Node-type-matched controls are missing for some saved pairs")
        top_nodes = metadata.get("top16")
        universe = metadata["node_names"]
        if (
            not top_nodes
            or len(set(top_nodes)) != len(top_nodes)
            or len(top_nodes) != min(16, len(universe))
            or not set(top_nodes).issubset(universe)
        ):
            raise ValueError("Valid saved top-node IDs are required to verify type matching")
        expected_counts = node_type_counts(top_nodes)
        for row in rows:
            top, comparator = row["top"], row["random_type_matched"]
            sampled = row.get("random_type_matched_nodes", [])
            if (
                len(sampled) != len(top_nodes)
                or len(sampled) != len(set(sampled))
                or not set(sampled).issubset(universe)
                or node_type_counts(sampled) != expected_counts
            ):
                raise ValueError(
                    "Type-matched random nodes must preserve size and exact node-type counts"
                )
            if not np.isclose(
                top["clean_metric"], comparator["clean_metric"], rtol=1e-5, atol=1e-6
            ):
                raise ValueError("Top and type-matched controls use different clean baselines")
            if top.get("scale", 1.0) != comparator.get("scale", 1.0):
                raise ValueError("Top and type-matched controls use different patch scales")
            if not np.isclose(
                comparator["patched_metric"] - comparator["clean_metric"],
                comparator["effect"],
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("Type-matched effect disagrees with patched-minus-clean metric")
        top = np.array([row["top"]["effect"] for row in rows])
        random = np.array([row["random_type_matched"]["effect"] for row in rows])
        ci = paired_metric_ci(random, top, groups, n_bootstrap=n_bootstrap, seed=seed)
        summary.pop("reason")
        summary.update(
            status="ok",
            n_rows=len(rows),
            n_groups=ci["n_groups"],
            mean_top_effect=ci["mean_b"],
            mean_random_effect=ci["mean_a"],
            top_minus_random=ci["difference_b_minus_a"],
            ci_top=ci["ci_b"],
            ci_random=ci["ci_a"],
            ci_top_minus_random=ci["ci_difference"],
            top_node_type_counts=expected_counts,
        )
    except (ValueError, KeyError, TypeError) as exc:
        summary.update(status="inconclusive", reason=str(exc))
    return summary


def _intervention_summary(stage_dir: Path, circuits: dict, n_bootstrap: int, seed: int) -> dict:
    result = {}
    for path in sorted((stage_dir / "circuits").glob("*_interventions.json")):
        task = path.stem.removesuffix("_interventions")
        rows = json.loads(path.read_text())
        item: dict[str, Any] = {
            "n_rows": len(rows),
            "effect_definition": "patched-minus-clean; negative values reduce the correct-answer metric",
            "difference_definition": "top-node patch effect minus size-matched random-node patch effect",
            "random_matching": "The uniform 'random' comparator matches set size only; the separate random_type_matched comparator, when saved, also preserves node-type counts",
            "source_artifact": str(path.relative_to(stage_dir.parent)),
            "source_sha256": sha256_file(path),
        }
        try:
            if not rows:
                raise ValueError("No disjoint intervention pairs were saved")
            if any("clean_group" not in row or "corrupt_group" not in row for row in rows):
                raise ValueError(
                    "Interventions lack independent source-group IDs; item IDs cannot substitute"
                )
            ids = [(row["clean_id"], row["corrupt_id"]) for row in rows]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate intervention pair IDs")
            meta = circuits.get(task, {}).get("metadata", {})
            if "clean_group_ids" not in meta or "corrupt_group_ids" not in meta:
                raise ValueError(
                    "Scoring metadata lacks source groups to verify intervention disjointness"
                )
            scoring_groups = set(meta["clean_group_ids"]) | set(meta["corrupt_group_ids"])
            intervention_groups = {
                row[key] for row in rows for key in ("clean_group", "corrupt_group")
            }
            if scoring_groups & intervention_groups:
                raise ValueError(
                    "Intervention source groups overlap attribution-ranking source groups"
                )
            for row in rows:
                top, random = row["top"], row["random"]
                if not np.isclose(
                    top["clean_metric"], random["clean_metric"], rtol=1e-5, atol=1e-6
                ):
                    raise ValueError("Top and random interventions use different clean baselines")
                if top.get("scale", 1.0) != random.get("scale", 1.0):
                    raise ValueError("Top and random interventions use different patch scales")
                for scores in (top, random):
                    if not np.isclose(
                        scores["patched_metric"] - scores["clean_metric"],
                        scores["effect"],
                        rtol=1e-5,
                        atol=1e-6,
                    ):
                        raise ValueError(
                            "Saved intervention effect disagrees with patched-minus-clean metric"
                        )
                random_nodes = row["random_nodes"]
                if (
                    len(set(random_nodes)) != len(random_nodes)
                    or len(random_nodes) != min(16, len(meta["node_names"]))
                    or not set(random_nodes).issubset(meta["node_names"])
                ):
                    raise ValueError(
                        "Random intervention nodes are not size-matched unique circuit nodes"
                    )
            # Join connected source pairs, including reversed clean/corrupt roles.
            # This remains valid if future diagnostics reuse a corruption source.
            parents = {group: group for group in intervention_groups}

            def find(group: str) -> str:
                while parents[group] != group:
                    parents[group] = parents[parents[group]]
                    group = parents[group]
                return group

            for row in rows:
                a, b = find(row["clean_group"]), find(row["corrupt_group"])
                parents[max(a, b)] = min(a, b)
            groups = [find(row["clean_group"]) for row in rows]
            top = np.array([row["top"]["effect"] for row in rows])
            random = np.array([row["random"]["effect"] for row in rows])
            ci = paired_metric_ci(random, top, groups, n_bootstrap=n_bootstrap, seed=seed)
            example = min(rows, key=lambda row: (row["clean_id"], row["corrupt_id"]))
            item.update(
                status="ok",
                n_groups=ci["n_groups"],
                mean_top_effect=ci["mean_b"],
                mean_random_effect=ci["mean_a"],
                top_minus_random=ci["difference_b_minus_a"],
                ci_top=ci["ci_b"],
                ci_random=ci["ci_a"],
                ci_top_minus_random=ci["ci_difference"],
                example={
                    key: example[key]
                    for key in (
                        "clean_id",
                        "corrupt_id",
                        "clean_group",
                        "corrupt_group",
                        "top",
                        "random",
                    )
                },
            )
            item["random_type_matched"] = _type_matched_intervention_summary(
                rows, meta, groups, n_bootstrap, seed
            )
            if item["random_type_matched"]["status"] == "ok":
                item["example"]["random_type_matched"] = example["random_type_matched"]
        except ValueError as exc:
            item.update(status="inconclusive", reason=str(exc))
        result[task] = item
    return result


def _example(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    preferred = [row for row in rows if row["task"] == "gsm_symbolic"] or rows
    row = min(preferred, key=lambda r: r["id"])
    return {
        key: row.get(key)
        for key in (
            "id",
            "task",
            "group",
            "prompt",
            "answer",
            "generation",
            "correct",
            "answer_log_prob_nats",
            "logit_margin",
            "truncated",
            "context_overflow",
            "prediction",
            "diagnostic_answer",
        )
    }


def _compact_probe(probe: dict) -> dict:
    """Remove duplicated fitting arrays only from the derived report.

    Original per-task JSON/NPZ artifacts remain authoritative and unchanged;
    source path/hash in report.json identifies their complete provenance.
    """
    omitted = {
        "training_mean",
        "training_scale",
        "split_indices",
        "predictions",
        "test_labels",
        "test_group_ids",
        "item_ids",
        "group_ids",
        "candidates",
    }
    return {
        key: [_compact_probe(row) if isinstance(row, dict) else row for row in value]
        if isinstance(value, list)
        else _compact_probe(value)
        if isinstance(value, dict)
        else value
        for key, value in probe.items()
        if key not in omitted
    }


def _plots(root: Path, summary: dict) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = summary["stages"]
    plot_dir = root / "plots"
    plot_dir.mkdir(exist_ok=True)
    plots = []
    example = summary["example"]
    common = {
        "example": example,
        "example_selection": "First GSM problem ID lexicographically when available, otherwise first saved item ID; selected independently of performance and not claimed representative.",
    }

    def save(fig: Any, name: str, caption: str, takeaway: str, confusing: str) -> None:
        fig.tight_layout()
        fig.savefig(plot_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(
            {
                "path": f"plots/{name}",
                "caption": caption,
                "takeaway": takeaway,
                "confusing": confusing,
                **common,
            }
        )

    def facets(count: int, height: float = 3.0) -> tuple[Any, list[Any]]:
        columns = min(3, count)
        rows = (count + columns - 1) // columns
        fig, axes = plt.subplots(
            rows, columns, figsize=(4.3 * columns, height * rows), squeeze=False
        )
        flat = list(axes.flat)
        for ax in flat[count:]:
            ax.set_visible(False)
        return fig, flat[:count]

    def title(task: str) -> str:
        return (
            task.replace("ethics_", "ETHICS · ")
            .replace("cruxeval_", "CRUXEval · ")
            .replace("gsm_symbolic", "GSM-Symbolic")
        )

    # Primary behavior excludes robustness splits and ETHICS control variants.
    keys = sorted(
        {
            key
            for stage in stages
            for key in summary["behavior"][stage]
            if key.endswith("/test/original")
        }
    )
    if keys:
        fig, axes = facets(len(keys))
        for ax, key in zip(axes, keys, strict=True):
            values = [summary["behavior"][stage].get(key, {}) for stage in stages]
            y = [item.get("accuracy", np.nan) for item in values]
            ax.plot(range(len(stages)), y, marker="o", label="Original", color="C0")
            for i, item in enumerate(values):
                interval = item.get("accuracy_uncertainty", {}).get("ci")
                if interval:
                    ax.vlines(i, *interval, color="C0", alpha=0.6)
            balanced_key = key.removesuffix("original") + "balanced_controls"
            balanced = [summary["behavior"][stage].get(balanced_key, {}) for stage in stages]
            if any(item for item in balanced):
                ax.plot(
                    range(len(stages)),
                    [item.get("accuracy", np.nan) for item in balanced],
                    marker="s",
                    linestyle="--",
                    label="Balanced controls",
                    color="C1",
                )
                for i, item in enumerate(balanced):
                    variant_values = item.get("variant_accuracies", [])
                    ax.scatter(
                        i + np.linspace(-0.06, 0.06, len(variant_values)),
                        variant_values,
                        marker="x",
                        s=18,
                        color="C1",
                        alpha=0.4,
                    )
                    interval = item.get("accuracy_uncertainty", {}).get("ci")
                    if interval:
                        ax.vlines(i, *interval, color="C1", alpha=0.6)
            metric = (
                "Group exact-match accuracy"
                if any(item.get("metric") == "group_exact_match" for item in values)
                else "Native accuracy"
            )
            ax.set(
                title=title(key.split("/")[0]),
                ylabel=metric,
                ylim=(-0.03, 1.03),
                xticks=range(len(stages)),
                xticklabels=stages,
                xlim=(-0.2, max(0.2, len(stages) - 0.8)),
            )
            ax.grid(axis="y", alpha=0.2)
            if len(stages) >= 3:
                ax.tick_params(axis="x", labelrotation=35, labelsize=8)
            ax.legend(fontsize=7, loc="best")
        measured = [
            (summary["behavior"][stage][key]["accuracy"], stage, key)
            for stage in stages
            for key in keys
            if key in summary["behavior"][stage]
        ]
        low, high = min(measured), max(measured)
        behavior_takeaway = f"Measured primary accuracies range from {low[0]:.1%} ({low[1]}, {low[2]}) to {high[0]:.1%} ({high[1]}, {high[2]})."
        math_values = [
            summary["behavior"][stage].get("gsm_symbolic/test/original") for stage in stages
        ]
        control_values = [
            summary["behavior"][stage].get("ethics_macro/test/balanced_controls")
            for stage in stages
        ]
        if len(stages) >= 2 and all(math_values):
            behavior_takeaway = f"Math accuracy changes from {math_values[0]['accuracy']:.1%} to {math_values[-1]['accuracy']:.1%}"
            if all(control_values):
                behavior_takeaway += f", while the balanced-control ETHICS macro changes from {control_values[0]['accuracy']:.1%} to {control_values[-1]['accuracy']:.1%}"
            behavior_takeaway += "."
        save(
            fig,
            "behavior.png",
            "Native accuracy by task: blue is original format, orange averages all eight ETHICS controls, faint crosses show individual control variants, and bars are source-group 95% intervals.",
            behavior_takeaway,
            "ETHICS deontology, justice and virtue require every judgment in a native group to be correct, so group accuracy can be zero with useful row accuracy. Control variants are averaged within source groups before resampling. A collapsed percentile-bootstrap interval at zero reflects the small observed sample, not certainty that population accuracy is zero.",
        )
        confidence_keys = [key for key in keys if not key.startswith("ethics_macro/")]
        fig, axes = facets(len(confidence_keys))
        for ax, key in zip(axes, confidence_keys, strict=True):
            task = key.split("/")[0]
            field = "logit_margin" if task.startswith("ethics_") else "answer_log_prob_nats"
            values = [summary["behavior"][stage].get(key, {}).get(field, {}) for stage in stages]
            ax.plot(
                range(len(stages)),
                [item.get("group_weighted_mean", item.get("mean", np.nan)) for item in values],
                marker="o",
                color="C0",
            )
            for i, item in enumerate(values):
                if item.get("ci"):
                    ax.vlines(i, *item["ci"], color="C0", alpha=0.6)
            ax.set(
                title=title(task),
                ylabel="Correct-label logit margin"
                if field == "logit_margin"
                else "Reference log probability (nats/token)",
                xticks=range(len(stages)),
                xticklabels=stages,
                xlim=(-0.2, max(0.2, len(stages) - 0.8)),
            )
            ax.grid(axis="y", alpha=0.2)
            if len(stages) >= 3:
                ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        confidence_takeaway = "Reference-answer confidence is a separate diagnostic from native generated-answer accuracy."
        if len(stages) >= 2 and all(math_values):
            lp = [
                item.get("answer_log_prob_nats", {}).get("group_weighted_mean")
                for item in math_values
            ]
            if all(value is not None for value in lp):
                confidence_takeaway = f"Math direct-answer log probability changes from {lp[0]:.2f} to {lp[-1]:.2f} nats/token, compared with generated-answer accuracy {math_values[0]['accuracy']:.1%} → {math_values[-1]['accuracy']:.1%}."
        save(
            fig,
            "confidence.png",
            "Original-format answer-confidence diagnostics, faceted by task with source-group 95% intervals.",
            confidence_takeaway,
            "The math confidence diagnostic requests a number directly without the target's reasoning, whereas behavioral evaluation generates reasoning first; these scores can move in opposite directions. ETHICS uses label margins, and CRUXEval-I allows multiple correct inputs. Each panel has its own vertical scale.",
        )

    overlap = summary["circuit_comparisons"]
    available = [
        (boundary, task, item)
        for boundary, tasks in overlap.items()
        for task, item in tasks.items()
        if item.get("status") == "ok"
    ]
    if available:
        overlap_tasks = sorted({task for _, task, _ in available})
        faceted = len(stages) >= 3
        if faceted:
            fig, axes = facets(len(overlap_tasks), height=3.8)
            task_axes = dict(zip(overlap_tasks, axes, strict=True))
        else:
            fig, ax = plt.subplots(figsize=(max(8, len(available)), 5))
        positions = defaultdict(int)
        labels = defaultdict(list)
        for position, (boundary, task, item) in enumerate(available):
            if faceted:
                ax = task_axes[task]
                i = positions[task]
                positions[task] += 1
                labels[task].append(boundary.replace(" -> ", "→"))
            else:
                i = position
            ax.plot(i, item["jaccard"], "o", color="C0")
            ax.vlines(i, *item["ci"], color="C0")
            ax.plot(i, item["random_baseline"]["mean"], "x", color="C1")
            typed = item.get("node_type_baseline")
            if typed:
                ax.plot(i + 0.18, typed["mean"], "s", color="C4", markersize=4)
                ax.vlines(
                    i + 0.18, *typed["reference_interval_95"], color="C4", alpha=0.35, linewidth=4
                )
            for offset, stage in zip((-0.1, 0.1), boundary.split(" -> "), strict=True):
                stability = summary["circuit_stability"][stage].get(task, {})
                if stability.get("status") == "ok":
                    ax.plot(i + offset, stability["mean_jaccard"], "_", color="C2")
        if faceted:
            for task, ax in task_axes.items():
                ax.set(
                    title=title(task),
                    xticks=range(len(labels[task])),
                    xticklabels=labels[task],
                    ylabel="Jaccard@16",
                    ylim=(-0.03, 1.03),
                )
                ax.tick_params(axis="x", labelrotation=45, labelsize=8)
            fig.suptitle(
                "Blue: overlap · orange: uniform null · purple: type-preserving null · green: split-half reliability",
                fontsize=11,
            )
        else:
            ax.set(
                xticks=range(len(available)),
                xticklabels=[f"{boundary}\n{task}" for boundary, task, _ in available],
                ylabel="Jaccard@16",
                ylim=(-0.03, 1.03),
            )
            ax.tick_params(axis="x", labelrotation=35)
            ax.set_title(
                "Blue: overlap · orange: uniform null · purple: type-preserving null\nGreen: split-half reliability"
            )
        vals = [item["jaccard"] for _, _, item in available]
        typed = [item for _, _, item in available if item.get("node_type_baseline")]
        above = sum(
            item["jaccard"] > item["node_type_baseline"]["reference_interval_95"][1]
            for item in typed
        )
        save(
            fig,
            "overlap.png",
            "Top-16 overlap with paired group bootstrap intervals, uniform-node and node-type-preserving random references, and split-half reliability; purple bars are null reference ranges, not confidence intervals.",
            f"Overlap ranges from {min(vals):.3f} to {max(vals):.3f}; {above}/{len(typed)} comparisons exceed the type-preserving random reference range.",
            "Whole MLP blocks occupy far fewer nodes than attention heads but dominate many top-16 sets. The type-preserving null fixes each set's observed MLP/head counts, so exceeding the uniform-node baseline alone is not evidence of task-specific reuse. Split-half reliability is not a ceiling, and sampled checkpoints can skip phases. With few groups and discontinuous top-16 selection, a percentile bootstrap interval can exclude the observed point; this flags unstable uncertainty estimation, not an error-bar clipping rule.",
        )
        plots[-1]["circuit_example"] = available[0][2]["example_pair"]
        plots[-1]["example"] = None

    matrices = summary["overlap_matrices"]
    if matrices:
        fig, axes = facets(len(matrices), height=3.7)
        for ax, (task, matrix) in zip(axes, matrices.items(), strict=True):
            array = np.array(
                [[np.nan if value is None else value for value in row] for row in matrix]
            )
            im = ax.imshow(array, vmin=0, vmax=1, cmap="viridis")
            ax.set(
                title=title(task),
                xticks=range(len(stages)),
                yticks=range(len(stages)),
                xticklabels=stages,
                yticklabels=stages,
            )
            ax.tick_params(axis="x", labelrotation=45)
            if len(stages) <= 7:
                for i in range(len(stages)):
                    for j in range(len(stages)):
                        if np.isfinite(array[i, j]):
                            ax.text(
                                j,
                                i,
                                f"{array[i, j]:.2f}",
                                ha="center",
                                va="center",
                                color="white" if array[i, j] < 0.6 else "black",
                                fontsize=9,
                            )
            fig.colorbar(im, ax=ax, shrink=0.7)
        typed_comparisons = [item for _, _, item in available if item.get("node_type_baseline")]
        within = sum(
            item["node_type_baseline"]["reference_interval_95"][0]
            <= item["jaccard"]
            <= item["node_type_baseline"]["reference_interval_95"][1]
            for item in typed_comparisons
        )
        matrix_takeaway = (
            f"The matrix covers {len(matrices)} task(s) and {len(stages)} saved checkpoints."
        )
        if typed_comparisons:
            matrix_takeaway = f"{within}/{len(typed_comparisons)} adjacent available checkpoint/task overlaps lie inside the node-type-preserving random reference ranges; diagonal ones are identity by construction."
        save(
            fig,
            "overlap_matrix.png",
            "All available stage-pair top-16 overlaps after strict input and node matching.",
            matrix_takeaway,
            "Blank cells have insufficient groups or mismatched artifacts; the diagonal is identity by construction, not evidence of reliable circuitry.",
        )
        if available:
            plots[-1]["circuit_example"] = available[0][2]["example_pair"]
            plots[-1]["example"] = None

    probes = [
        (stage, task, report)
        for stage, tasks in summary["probes"].items()
        for task, report in tasks.items()
        if report.get("layers")
    ]
    if probes:
        tasks = sorted({task for _, task, _ in probes})
        fig, axes = facets(len(tasks), height=3.4)
        stage_handles = {}
        for ax, task in zip(axes, tasks, strict=True):
            group_counts = set()
            for stage, report_task, report in probes:
                if task != report_task:
                    continue
                color = f"C{stages.index(stage) % 10}"
                layer_ids = [row["layer"] for row in report["layers"]]
                heldout_groups = report["layers"][0].get("split_n_groups", {}).get("test", "?")
                group_counts.add(str(heldout_groups))
                stage_handles[stage] = ax.plot(
                    layer_ids,
                    [row["test_accuracy"] for row in report["layers"]],
                    label=f"{stage} (test groups: {heldout_groups})",
                    color=color,
                )[0]
                ax.plot(
                    layer_ids,
                    [row["shuffled_test_accuracy_mean"] for row in report["layers"]],
                    "--",
                    alpha=0.6,
                    color=color,
                )
                intervals = [
                    row.get("test_accuracy_uncertainty", {}).get("ci") for row in report["layers"]
                ]
                if all(interval is not None for interval in intervals):
                    ax.fill_between(
                        layer_ids,
                        [interval[0] for interval in intervals],
                        [interval[1] for interval in intervals],
                        color=color,
                        alpha=0.07,
                    )
                if report.get("answer_only", {}).get("layers"):
                    control = report["answer_only"]["layers"]
                    ax.plot(
                        [row["layer"] for row in control],
                        [row["test_accuracy"] for row in control],
                        ":",
                        alpha=0.8,
                        color=color,
                    )
                majority = report["layers"][0].get("majority_test_accuracy")
                if majority is not None:
                    ax.axhline(majority, color="gray", alpha=0.25, linewidth=1)
            ax.set(
                title=title(task)
                + (
                    f"\nHeld-out source groups: {', '.join(sorted(group_counts))}"
                    if len(stages) >= 3
                    else ""
                ),
                xlabel="Residual layer",
                ylabel="Held-out group accuracy",
                ylim=(-0.02, 1.02),
            )
            ax.grid(axis="y", alpha=0.15)
            if len(stages) < 3:
                ax.legend(fontsize=7, loc="best")
            else:
                ax.title.set_fontsize(10)
        if len(stages) >= 3:
            legend_stages = [stage for stage in stages if stage in stage_handles]
            fig.legend(
                [stage_handles[stage] for stage in legend_stages],
                legend_stages,
                loc="upper center",
                ncol=len(legend_stages),
                bbox_to_anchor=(0.5, 1.02),
                fontsize=9,
            )
        probe_takeaway = f"Held-out diagnostic probes were estimable for {len(probes)} checkpoint-task combinations."
        virtues = [item for _, task, item in probes if task == "ethics_virtue"]
        if virtues and all(
            abs(
                item["layers"][-1]["test_accuracy"]
                - item["layers"][-1].get("majority_test_accuracy", -1)
            )
            < 1e-12
            for item in virtues
        ):
            probe_takeaway = f"Final-layer virtue accuracy ({virtues[-1]['layers'][-1]['test_accuracy']:.1%}) matches its majority baseline, illustrating why raw probe scores need controls."
        save(
            fig,
            "probes.png",
            "Layerwise probes by task: solid lines are held-out accuracy, dashed lines shuffled-label controls, dotted lines answer-only controls, gray lines majority baselines, and faint bands source-group 95% intervals.",
            probe_takeaway,
            "A high score shared by shuffled-label or majority baselines can reflect label imbalance rather than usable capability. These small grouped test sets give noisy layerwise curves; selecting the best test layer is exploratory and is not a held-out model-selection result. Probe accuracy does not establish latent generation ability.",
        )
        chosen = next(
            (
                item
                for _, task, item in probes
                if task == "gsm_symbolic" and item.get("heldout_example")
            ),
            None,
        )
        chosen = chosen or next(
            (item for _, _, item in probes if item.get("heldout_example")), None
        )
        if chosen:
            plots[-1]["probe_example"] = chosen["heldout_example"]
            plots[-1]["example"] = None
    interventions = [
        (stage, task, item)
        for stage, tasks in summary["interventions"].items()
        for task, item in tasks.items()
        if item.get("status") == "ok"
    ]
    if interventions:
        intervention_tasks = sorted({task for _, task, _ in interventions})
        faceted = len(stages) >= 3
        if faceted:
            fig, axes = facets(len(intervention_tasks), height=3.6)
            task_axes = dict(zip(intervention_tasks, axes, strict=True))
        else:
            fig, ax = plt.subplots(figsize=(max(8, len(interventions)), 5))
        typed_items = [
            item["random_type_matched"]
            for _, _, item in interventions
            if item.get("random_type_matched", {}).get("status") == "ok"
        ]
        have_typed = bool(typed_items)
        labelled_typed = False
        for position, (stage, task, item) in enumerate(interventions):
            if faceted:
                ax = task_axes[task]
                i = stages.index(stage)
            else:
                i = position
            uniform_position = i - 0.12 if have_typed else i
            ax.plot(
                uniform_position,
                item["top_minus_random"],
                "o",
                color="C0",
                label="Uniform random: size only" if faceted or i == 0 else None,
            )
            ax.vlines(uniform_position, *item["ci_top_minus_random"], color="C0")
            typed = item.get("random_type_matched", {})
            if typed.get("status") == "ok":
                ax.plot(
                    i + 0.12,
                    typed["top_minus_random"],
                    "s",
                    color="C1",
                    label="Random: size and node type" if faceted or not labelled_typed else None,
                )
                ax.vlines(i + 0.12, *typed["ci_top_minus_random"], color="C1")
                labelled_typed = True
        if faceted:
            for task, ax in task_axes.items():
                ax.axhline(0, color="gray", linestyle="--")
                ax.set(
                    title=title(task),
                    xticks=range(len(stages)),
                    xticklabels=stages,
                    ylabel="Top-minus-random patch effect",
                    xlim=(-0.5, len(stages) - 0.5),
                )
                ax.tick_params(axis="x", labelrotation=35, labelsize=8)
            handles, legend_labels = axes[0].get_legend_handles_labels()
            unique_legend = dict(zip(legend_labels, handles))
            fig.legend(
                unique_legend.values(),
                unique_legend.keys(),
                loc="upper center",
                ncol=2,
                bbox_to_anchor=(0.5, 1.02),
                fontsize=9,
            )
        else:
            ax.axhline(0, color="gray", linestyle="--")
            ax.set(
                xticks=range(len(interventions)),
                xticklabels=[f"{stage}\n{task}" for stage, task, _ in interventions],
                ylabel="Top-minus-random patch effect (task-specific metric)",
                xlim=(-0.5, len(interventions) - 0.5),
            )
            ax.tick_params(axis="x", labelrotation=35)
            ax.legend(fontsize=8)
        negative = sum(item["ci_top_minus_random"][1] < 0 for _, _, item in interventions)
        if have_typed:
            typed_negative = sum(item["ci_top_minus_random"][1] < 0 for item in typed_items)
            takeaway = f"Top-node patching exceeds the size-and-node-type-matched control with an interval below zero in {typed_negative} of {len(typed_items)} available checkpoint-task checks; this does not prove acquisition."
            caption = "Disjoint-source top-node effects minus uniform size-matched controls (circles) and, where saved, size-and-node-type-matched controls (squares), with paired source-group 95% intervals."
            confusing = "Type matching controls the number of whole MLP blocks versus individual heads, but not every structural or task-independent difference between nodes. These exploratory intervals are unadjusted for multiple comparisons and use few independent pairs; units differ by task. Stronger disruption supports causal relevance, not proof of acquisition or a newly formed circuit. Missing type-matched controls are not inferred from uniform controls."
        else:
            takeaway = f"Top-node patching reduces the correct-answer metric more than random patching with an interval below zero in {negative} of {len(interventions)} saved checkpoint-task checks."
            caption = "Disjoint-source top-node versus size-matched random-node activation patch effects, with paired source-group 95% intervals."
            confusing = "Random patches match set size but not the top set's MLP/head composition, so stronger disruption can reflect node type rather than task-specific importance. These exploratory intervals are unadjusted for multiple comparisons and use few independent pairs; metric units differ by task. No type-matched interventions were saved, and these checks do not prove acquisition."
        save(
            fig,
            "interventions.png",
            caption,
            takeaway,
            confusing,
        )
        example_item = next(
            (
                item
                for _, _, item in interventions
                if item.get("random_type_matched", {}).get("status") == "ok"
            ),
            interventions[0][2],
        )
        plots[-1]["intervention_example"] = example_item["example"]
        plots[-1]["example"] = None
    return plots


def generate_report(run_dir: str | Path, n_bootstrap: int = 2000, seed: int = 0) -> dict:
    """Write report.json, report.md and plots from saved local evaluation outputs."""
    root = Path(run_dir)
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    stages = [cp.stage for cp in CHECKPOINTS if (root / cp.stage / "behavior.jsonl").exists()]
    if not stages:
        raise ValueError("No saved checkpoint behavior.jsonl artifacts found")
    run_metadata_path = root / "run_metadata.json"
    probe_policy = (
        json.loads(run_metadata_path.read_text()).get("probe_policy_by_stage", {})
        if run_metadata_path.exists()
        else {}
    )
    records = {stage: read_jsonl(root / stage / "behavior.jsonl") for stage in stages}
    if any(not rows for rows in records.values()):
        raise ValueError("A saved checkpoint has no behavioral records")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "probe_policy_by_stage": probe_policy,
        "analysis_provenance": {
            "report_module_sha256": sha256_file(Path(__file__)),
            "statistics_module_sha256": sha256_file(Path(__file__).with_name("statistics.py")),
            "note": "Derived offline analysis; original per-checkpoint measurement artifacts remain unchanged.",
        },
        "stages": stages,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "scope": "Saved evaluation artifacts only; scope is not assumed to be the full benchmark.",
        "n_behavioral_rows": {stage: len(rows) for stage, rows in records.items()},
        "behavior": {
            stage: _behavior_summary(rows, n_bootstrap, seed) for stage, rows in records.items()
        },
        "paired_behavior": {},
        "circuit_stability": {},
        "circuit_comparisons": {},
        "overlap_matrices": {},
        "probes": {},
        "interventions": {},
        "example": _example(records[stages[0]]),
    }
    if summary["example"]:
        summary["example"]["checkpoint_results"] = {
            stage: {key: row.get(key) for key in ("prediction", "correct", "answer_log_prob_nats")}
            for stage, rows in records.items()
            for row in rows
            if row["id"] == summary["example"]["id"]
        }
    circuits = {stage: _load_circuits(root / stage) for stage in stages}
    for stage in stages:
        summary["interventions"][stage] = _intervention_summary(
            root / stage, circuits[stage], n_bootstrap, seed
        )
        summary["probes"][stage] = {
            path.stem: json.loads(path.read_text())
            for path in sorted((root / stage / "probes").glob("*.json"))
        }
        for task, probe in summary["probes"][stage].items():
            if probe.get("layers") and probe.get("item_ids"):
                layer = probe["layers"][-1]
                indices = layer.get("split_indices", {}).get("test", [])
                if indices and layer.get("predictions"):
                    position = min(range(len(indices)), key=lambda i: probe["item_ids"][indices[i]])
                    item_id = probe["item_ids"][indices[position]]
                    candidate = next(
                        (
                            row.get("candidate")
                            for row in probe.get("candidates", [])
                            if row["id"] == item_id
                        ),
                        None,
                    )
                    probe["heldout_example"] = {
                        "stage": stage,
                        "task": task,
                        "item_id": item_id,
                        "layer": layer["layer"],
                        "predicted_label": layer["predictions"][position],
                        "true_label": layer["test_labels"][position],
                        "candidate": candidate,
                        "selection": "First heldout item ID lexicographically, at the fixed final residual layer; not selected using its outcome.",
                    }
            for layer in probe.get("layers", []) + probe.get("answer_only", {}).get("layers", []):
                try:
                    if "test_group_ids" not in layer:
                        raise ValueError("Saved probe lacks heldout source-group IDs")
                    correct = np.asarray(layer["predictions"]) == np.asarray(layer["test_labels"])
                    ci = paired_metric_ci(
                        correct,
                        correct,
                        layer["test_group_ids"],
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                    )
                    layer["test_accuracy_uncertainty"] = {
                        "status": "ok",
                        "ci": ci["ci_a"],
                        "scope": "heldout groups conditional on the fitted classifier",
                    }
                except ValueError as exc:
                    layer["test_accuracy_uncertainty"] = {
                        "status": "inconclusive",
                        "reason": str(exc),
                    }
            source = root / stage / "probes" / f"{task}.json"
            compact = _compact_probe(probe)
            compact["source_artifact"] = str(source.relative_to(root))
            compact["source_sha256"] = sha256_file(source)
            summary["probes"][stage][task] = compact
        summary["circuit_stability"][stage] = {}
        for task, item in circuits[stage].items():
            try:
                if item["status"] != "loaded":
                    raise ValueError(item["reason"])
                meta = item["metadata"]
                result = split_half_stability(
                    item["scores"],
                    meta["node_names"],
                    meta["group_ids"],
                    seed=seed,
                    strata=meta.get("strata"),
                )
                result["status"] = "ok"
            except ValueError as exc:
                result = {"status": "inconclusive", "reason": str(exc)}
            summary["circuit_stability"][stage][task] = result
    tasks = sorted({task for stage in stages for task in circuits[stage]})
    matrices = {task: [[None for _ in stages] for _ in stages] for task in tasks}
    for i, left in enumerate(stages):
        for j in range(i, len(stages)):
            right = stages[j]
            boundary = f"{left} -> {right}"
            if j == i + 1:
                summary["paired_behavior"][boundary] = _paired_behavior(
                    records[left], records[right], n_bootstrap, seed
                )
            pair_results = {}
            for task in tasks:
                try:
                    a, b = circuits[left].get(task), circuits[right].get(task)
                    if not a or not b or a["status"] != "loaded" or b["status"] != "loaded":
                        raise ValueError("Circuit data absent at one or both checkpoints")
                    validate_matched_rows(a["metadata"], b["metadata"])
                    meta = a["metadata"]
                    result = paired_overlap_ci(
                        a["scores"],
                        b["scores"],
                        meta["node_names"],
                        meta["group_ids"],
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                        strata=meta.get("strata"),
                    )
                    result.update(
                        status="ok",
                        random_baseline=random_jaccard_baseline(len(meta["node_names"]), seed=seed),
                        example_pair_id=min(meta["item_ids"]),
                        node_type_baseline=node_type_jaccard_baseline(
                            result["top_a"], result["top_b"], meta["node_names"], seed=seed
                        ),
                    )
                    pair_index = meta["item_ids"].index(result["example_pair_id"])
                    node = result["top_a"][0]
                    node_index = meta["node_names"].index(node)
                    result["example_pair"] = {
                        "id": result["example_pair_id"],
                        "node": node,
                        "signed_score_a": float(a["scores"][pair_index, node_index]),
                        "signed_score_b": float(b["scores"][pair_index, node_index]),
                        "selection": "First pair ID lexicographically, at the first aggregate-ranked node of the earlier checkpoint; not representative sampling.",
                    }
                    matrices[task][i][j] = matrices[task][j][i] = result["jaccard"]
                except ValueError as exc:
                    result = {"status": "inconclusive", "reason": str(exc)}
                pair_results[task] = result
            if j == i + 1:
                summary["circuit_comparisons"][boundary] = pair_results
    summary["overlap_matrices"] = {
        task: matrix
        for task, matrix in matrices.items()
        if any(value is not None for row in matrix for value in row)
    }
    summary["plots"] = _plots(root, summary)
    valid_comparisons = sum(
        item.get("status") == "ok"
        for tasks in summary["circuit_comparisons"].values()
        for item in tasks.values()
    )
    summary["tldr"] = [
        f"Measured {len(stages)} saved checkpoint(s), with {', '.join(f'{stage}: {len(records[stage])}' for stage in stages)} behavioral rows; this report covers only that saved scope.",
        f"Computed {valid_comparisons} matched consecutive-available-stage circuit comparison(s); unsupported analyses are explicitly inconclusive.",
        "These measurements describe behavior and circuit overlap; they do not by themselves prove elicitation or acquisition.",
    ]
    if len(stages) == 2:
        math_key = "gsm_symbolic/test/original"
        first = summary["behavior"][stages[0]].get(math_key)
        last = summary["behavior"][stages[1]].get(math_key)
        paired = (
            summary["paired_behavior"]
            .get(f"{stages[0]} -> {stages[1]}", {})
            .get(math_key, {})
            .get("correct", {})
        )
        if first and last and paired.get("status") == "ok":
            lo, hi = paired["ci_difference"]
            summary["tldr"][0] = (
                f"Two-checkpoint pilot: math accuracy {first['accuracy']:.1%} → {last['accuracy']:.1%} on {first['n_examples']} problems/{first['n_groups']} templates; paired 95% interval for the gain {100 * lo:.1f}–{100 * hi:.1f} percentage points."
            )
        comparisons = [
            item
            for tasks in summary["circuit_comparisons"].values()
            for item in tasks.values()
            if item.get("status") == "ok"
        ]
        if comparisons:
            above = sum(
                item["jaccard"] > item["node_type_baseline"]["reference_interval_95"][1]
                for item in comparisons
            )
            summary["tldr"][1] = (
                f"Observed circuit overlaps are {min(item['jaccard'] for item in comparisons):.3f}–{max(item['jaccard'] for item in comparisons):.3f}; {above}/{len(comparisons)} exceed the node-type-preserving 95% random reference range."
            )
        summary["tldr"][2] = (
            "Label controls, imbalanced probe baselines and small diagnostic samples keep circuit reuse inconclusive; these endpoints cannot locate the responsible intervening training phase."
        )
    elif len(stages) >= 3:
        math_results = [
            summary["behavior"][stage].get("gsm_symbolic/test/original") for stage in stages
        ]
        available_math = [item for item in math_results if item]
        if available_math:
            counts = {(item["n_examples"], item["n_groups"]) for item in available_math}
            shared_counts = len(counts) == 1 and len(available_math) == len(stages)
            sequence = []
            for stage, item in zip(stages, math_results):
                value = f"{item['accuracy']:.1%}" if item else "unavailable"
                if item and not shared_counts:
                    value += f" ({item['n_examples']} problems/{item['n_groups']} templates)"
                sequence.append(f"{stage} {value}")
            scope = ""
            if shared_counts:
                problems, templates = next(iter(counts))
                scope = f"; {problems} problems/{templates} templates per checkpoint"
            summary["tldr"][0] = f"Math accuracy: {' → '.join(sequence)}{scope}."
        comparisons = [
            item
            for tasks in summary["circuit_comparisons"].values()
            for item in tasks.values()
            if item.get("status") == "ok"
        ]
        if comparisons:
            above = sum(
                item["jaccard"] > item["node_type_baseline"]["reference_interval_95"][1]
                for item in comparisons
            )
            summary["tldr"][1] = (
                f"Adjacent available checkpoint/task Jaccard@16 is {min(item['jaccard'] for item in comparisons):.3f}–{max(item['jaccard'] for item in comparisons):.3f}; {above}/{len(comparisons)} exceed their node-type-preserving 95% random reference range (unadjusted for multiple comparisons)."
            )
        else:
            summary["tldr"][1] = (
                "No adjacent available checkpoint/task circuit comparison has sufficient matched independent groups; overlap is inconclusive."
            )
        metadata_path = root / "run_metadata.json"
        mode = (
            json.loads(metadata_path.read_text()).get("config", {}).get("mode")
            if metadata_path.exists()
            else None
        )
        scope = "Technical sanity sample" if mode == "pilot" else "Saved evaluation sample"
        summary["tldr"][2] = (
            f"{scope} across {len(stages)} checkpoints: intervals reflect sampled source groups, not independent training runs; label controls, probe baselines and patching comparisons must qualify any reuse claim, and overlap alone cannot establish acquisition or elicitation."
        )
    write_json(root / "report.json", summary)
    lines = summary["tldr"] + [
        "",
        "Intervals resample independent source groups, not training runs. Control variants stay grouped. Native ETHICS group scoring is preserved.",
        "",
    ]
    skipped = [stage for stage, policy in probe_policy.items() if policy == "skipped"]
    if skipped:
        lines.extend(
            [
                "Probe scope: further activation extraction and linear-probe fitting were stopped by user request for "
                + ", ".join(skipped)
                + ". Any already completed probe results are retained; missing probe results are intentional. Accuracy, answer scores, circuit top nodes and interventions retain their full scope.",
                "",
            ]
        )
    for plot in summary["plots"]:
        lines.extend(
            [plot["takeaway"], "", f"![{plot['caption']}]({plot['path']})", "", plot["caption"], ""]
        )
        if plot["example"]:
            ex = plot["example"]
            prompt = (
                str(ex["prompt"])
                .rsplit("\nQ:", 1)[-1]
                .split("\nA:", 1)[0]
                .strip()[:220]
                .replace("\n", " ")
            )
            reference = ex.get("diagnostic_answer") or ex["answer"]
            outcomes = "; ".join(
                f"{stage}: parsed prediction `{row['prediction']}`, correct={row['correct']}"
                for stage, row in ex.get("checkpoint_results", {}).items()
            )
            lines.extend(
                [
                    f"Example `{ex['id']}`: “{prompt}” Reference `{str(reference)[:80]}`; {outcomes}. {plot['example_selection']}",
                    "",
                ]
            )
            if ex["task"] == "cruxeval_input":
                lines.extend(
                    [
                        "CRUXEval-I grades the recovered input against the real function; an asserted output printed in the generation is ignored by this input-prediction checker.",
                        "",
                    ]
                )
        if plot.get("circuit_example"):
            ex = plot["circuit_example"]
            lines.extend(
                [
                    f"Circuit example pair `{ex['id']}`, node `{ex['node']}`: signed attribution {ex['signed_score_a']:.5g} → {ex['signed_score_b']:.5g}. {ex['selection']}",
                    "",
                ]
            )
        if plot.get("intervention_example"):
            ex = plot["intervention_example"]
            typed_text = (
                f", type-matched random effect {ex['random_type_matched']['effect']:.5g}"
                if "random_type_matched" in ex
                else ""
            )
            lines.extend(
                [
                    f"Intervention example `{ex['clean_id']}|{ex['corrupt_id']}`: top-node patch effect {ex['top']['effect']:.5g}, uniform random-node patch effect {ex['random']['effect']:.5g}{typed_text}. First saved pair lexicographically; not selected as representative.",
                    "",
                ]
            )
        if plot.get("probe_example"):
            ex = plot["probe_example"]
            lines.extend(
                [
                    f"Probe example `{ex['item_id']}` ({ex['stage']}, residual layer {ex['layer']}): candidate `{ex.get('candidate')}`, predicted label {ex['predicted_label']}, true label {ex['true_label']}. {ex['selection']}",
                    "",
                ]
            )
        lines.extend([plot["confusing"], ""])
    skipped = [
        (f"{boundary}/{task}", item["reason"])
        for boundary, tasks in summary["circuit_comparisons"].items()
        for task, item in tasks.items()
        if item.get("status") != "ok"
    ]
    if skipped:
        lines.append(
            "Inconclusive circuit comparisons: "
            + "; ".join(f"{name}: {reason}" for name, reason in skipped)
        )
    if not summary["plots"]:
        lines.append(
            "No plots are estimable from the saved artifacts. See report.json for unavailable analyses."
        )
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return summary
