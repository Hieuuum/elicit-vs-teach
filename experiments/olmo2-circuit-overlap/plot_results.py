"""Create the requested CPU-only plots from completed artifacts, one stage at a time."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from geode.circuits.artifacts import sha256_file, validate_matched_rows, write_json
from geode.circuits.checkpoints import CHECKPOINTS
from geode.circuits.report import _behavior_summary
from geode.circuits.sanity import validate_probe_splits
from geode.circuits.statistics import jaccard, paired_overlap_ci, topk_nodes

TASKS = [
    "ethics_commonsense",
    "ethics_deontology",
    "ethics_justice",
    "ethics_virtue",
    "ethics_utilitarianism",
    "cruxeval_input",
    "cruxeval_output",
    "gsm_symbolic",
]
TITLES = [
    "ETHICS · Commonsense",
    "ETHICS · Deontology",
    "ETHICS · Justice",
    "ETHICS · Virtue",
    "ETHICS · Utilitarianism",
    "CRUXEval · Input",
    "CRUXEval · Output",
    "GSM-Symbolic",
]
STAGES = [cp.stage for cp in CHECKPOINTS]
LABELS = ["Init", "Stage 1", "Stage 2", "SFT", "DPO", "RLVR1", "RLVR2"]
COLORS = ["#778899", "#4c78a8", "#72b7b2", "#f2a541", "#b279a2", "#e45756", "#238b45"]


def csv_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def original_rows(path):
    """Retain only native test examples; raw results and control variants stay untouched."""
    result = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            meta = row["metadata"]
            if meta["split"] == "test" and meta.get("variant", "original") == "original":
                result.append(row)
    return result


def collect(run, pilot, output, n_bootstrap):
    metadata = json.loads((run / "run_metadata.json").read_text())
    audit = json.loads((run / "technical_audit.json").read_text())
    if metadata["status"] != "complete" or audit["status"] != "passed" or audit["warnings"]:
        raise ValueError("Completed, technically audited results are required")
    data = {
        "stages": STAGES,
        "tasks": TASKS,
        "behavior": {},
        "overlap": {},
        "probes": {},
        "probe_coverage": [],
        "examples": {},
        "n_bootstrap": n_bootstrap,
        "plan_fingerprint": metadata["plan_fingerprint"],
        "source_files": {},
        "script_sha256": sha256_file(Path(__file__)),
        "created_epoch": time.time(),
    }
    reference_identity = None
    behavior_csv = []
    for stage in STAGES:
        path = run / stage / "behavior.jsonl"
        rows = original_rows(path)
        identity = [(r["id"], r["group"], r["prompt"], r["answer"], r["metadata"]) for r in rows]
        if reference_identity is not None and identity != reference_identity:
            raise ValueError("Native behavioral inputs differ across stages")
        if reference_identity is None:
            reference_identity = identity
        metrics = _behavior_summary(rows, n_bootstrap, 0)
        data["behavior"][stage] = {}
        for task in TASKS:
            item = metrics[f"{task}/test/original"]
            data["behavior"][stage][task] = item
            score_field = "logit_margin" if task.startswith("ethics_") else "answer_log_prob_nats"
            score = item[score_field]
            behavior_csv.append(
                {
                    "stage": stage,
                    "task": task,
                    "n_examples": item["n_examples"],
                    "n_groups": item["n_groups"],
                    "accuracy": item["accuracy"],
                    "accuracy_ci_low": item["accuracy_uncertainty"]["ci"][0],
                    "accuracy_ci_high": item["accuracy_uncertainty"]["ci"][1],
                    "row_accuracy": item["row_accuracy"],
                    "score_field": score_field,
                    "score_mean": score["group_weighted_mean"],
                    "score_ci_low": score["ci"][0],
                    "score_ci_high": score["ci"][1],
                    "parse_failure_rate": item["parse_failure_rate"],
                    "truncation_rate": item["truncation_rate"],
                }
            )
        ex = min((r for r in rows if r["task"] == "gsm_symbolic"), key=lambda r: r["id"])
        data["examples"][stage] = {
            k: ex.get(k) for k in ["id", "prompt", "answer", "prediction", "correct", "generation"]
        }
        data["source_files"][str(path)] = sha256_file(path)
        print(json.dumps({"event": "behavior_summarized", "stage": stage}), flush=True)
        del rows, identity
    csv_rows(output / "accuracy_and_scores.csv", behavior_csv)
    overlap_csv = []
    for task in TASKS:
        data["overlap"][task] = {}
        for left, right in zip(STAGES[:-1], STAGES[1:]):
            paths = [run / s / "circuits" / task for s in [left, right]]
            metas = [json.loads(p.with_suffix(".json").read_text()) for p in paths]
            validate_matched_rows(*metas)
            arrays = []
            for p in paths:
                with np.load(p.with_suffix(".npz"), allow_pickle=False) as saved:
                    arrays.append(saved["scores"])
            sensitivity = paired_overlap_ci(
                *arrays,
                metas[0]["node_names"],
                metas[0]["group_ids"],
                k=16,
                n_bootstrap=n_bootstrap,
                seed=0,
            )
            # Plot the actual saved sets used in interventions. Equal-group
            # weighting is a separate sensitivity when cluster sizes differ.
            for array, meta in zip(arrays, metas, strict=True):
                if topk_nodes(array, meta["node_names"], 16) != meta["top16"]:
                    raise ValueError(f"Saved top nodes disagree with attribution scores: {task}")
            result = {
                "jaccard": jaccard(metas[0]["top16"], metas[1]["top16"]),
                "top_a": metas[0]["top16"],
                "top_b": metas[1]["top16"],
                "n_groups": sensitivity["n_groups"],
                "n_examples": sensitivity["n_examples"],
                "group_weighted_sensitivity": sensitivity,
            }
            data["overlap"][task][f"{left} -> {right}"] = result
            overlap_csv.append(
                {
                    "task": task,
                    "from_stage": left,
                    "to_stage": right,
                    "jaccard16": result["jaccard"],
                    "group_weighted_jaccard": sensitivity["jaccard"],
                    "group_weighted_ci_low": sensitivity["ci"][0],
                    "group_weighted_ci_high": sensitivity["ci"][1],
                    "n_groups": result["n_groups"],
                    "shared_nodes": len(set(result["top_a"]) & set(result["top_b"])),
                }
            )
        print(json.dumps({"event": "overlap_summarized", "task": task}), flush=True)
    csv_rows(output / "jaccard16_transitions.csv", overlap_csv)
    # RLVR2's full-size pilot used the same checkpoint, frozen rows and five-shuffle protocol.
    gate = json.loads((pilot / "pilot_status.json").read_text())
    if gate["status"] != "passed" or gate["full_plan_fingerprint"] != metadata["plan_fingerprint"]:
        raise ValueError("RLVR2 full-size pilot does not match this run")
    cp = next(cp for cp in metadata["config"]["checkpoints"] if cp["stage"] == "rlvr2")
    for key in ["repo", "revision"]:
        if gate["checkpoint"][key] != cp[key]:
            raise ValueError("Pilot checkpoint mismatch")
    identities = {}
    probe_csv = []
    for stage in STAGES:
        data["probes"][stage] = {}
        for task in TASKS:
            path = run / stage / "probes" / f"{task}.json"
            source = "full_run"
            if not path.exists() and stage == "rlvr2":
                path = pilot / stage / "probes" / f"{task}.json"
                source = "full_size_pilot"
            if not path.exists():
                data["probe_coverage"].append(
                    {"stage": stage, "task": task, "source": "unavailable"}
                )
                continue
            probe = json.loads(path.read_text())
            if probe["status"] != "complete":
                raise ValueError(f"Incomplete probe report: {path}")
            if source == "full_size_pilot":
                expected = gate["probe_audits"][task]
                if (
                    sha256_file(path) != expected["report_sha256"]
                    or sha256_file(path.with_suffix(".npz")) != expected["feature_artifact_sha256"]
                ):
                    raise ValueError(f"Pilot artifact hash mismatch: {task}")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
                labels, groups = saved["labels"], saved["groups"]
            validate_probe_splits(probe, labels, groups)
            identity = (
                probe["item_ids"],
                labels.tolist(),
                groups.tolist(),
                [layer["split_indices"] for layer in probe["layers"]],
            )
            if task in identities and identity != identities[task]:
                raise ValueError(f"Probe rows or splits differ across checkpoints: {task}")
            identities[task] = identity
            layers = []
            for layer in probe["layers"]:
                if len(layer["shuffled_test_accuracies"]) != 5:
                    raise ValueError("Probe shuffle protocol mismatch")
                by_group = defaultdict(list)
                for pred, truth, group in zip(
                    layer["predictions"], layer["test_labels"], layer["test_group_ids"], strict=True
                ):
                    by_group[group].append(pred == truth)
                measured = np.mean([np.mean(values) for values in by_group.values()])
                if not np.isclose(measured, layer["test_accuracy"]):
                    raise ValueError("Saved probe accuracy disagrees with predictions")
                item = {
                    k: layer[k]
                    for k in [
                        "layer",
                        "test_accuracy",
                        "shuffled_test_accuracy_mean",
                        "majority_test_accuracy",
                    ]
                }
                layers.append(item)
                probe_csv.append(
                    {
                        "stage": stage,
                        "task": task,
                        "source": source,
                        **item,
                        "test_groups": layer["split_n_groups"]["test"],
                    }
                )
            data["probes"][stage][task] = {
                "source": source,
                "layers": layers,
                "n_rows": probe["n_rows"],
                "test_groups": probe["layers"][0]["split_n_groups"]["test"],
            }
            data["probe_coverage"].append({"stage": stage, "task": task, "source": source})
            data["source_files"][str(path)] = sha256_file(path)
        print(json.dumps({"event": "probes_loaded", "stage": stage}), flush=True)
    csv_rows(output / "linear_probe_layers.csv", probe_csv)
    csv_rows(output / "probe_coverage.csv", data["probe_coverage"])
    write_json(output / "plot_data.json", data)
    return data


def plot(data, output):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )

    def save(fig, name):
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(
                output / f"{name}.{extension}", dpi=180, bbox_inches="tight", facecolor="white"
            )
        plt.close(fig)

    transitions = [f"{a} →\n{b}" for a, b in zip(LABELS[:-1], LABELS[1:])]
    matrix = np.array(
        [
            [
                data["overlap"][task][f"{a} -> {b}"]["jaccard"]
                for a, b in zip(STAGES[:-1], STAGES[1:])
            ]
            for task in TASKS
        ]
    )
    fig, ax = plt.subplots(figsize=(11, 6.3))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set(xticks=range(6), xticklabels=transitions, yticks=range(8), yticklabels=TITLES)
    ax.tick_params(length=0, pad=9)
    for i in range(8):
        for j in range(6):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > 0.55 else "#192b3a",
                fontsize=12,
            )
    fig.colorbar(im, ax=ax, pad=0.025, label="Jaccard@16 · |intersection| / |union|")
    ax.set_title(
        "Top-node overlap between consecutive training stages",
        loc="left",
        pad=18,
        fontweight="bold",
    )
    fig.text(
        0.02,
        -0.015,
        "Actual saved top-16 sets used for interventions. CSV includes a separate equal-source-group weighting sensitivity.",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, "jaccard16_transitions")
    x = np.arange(7)
    for kind in ["accuracy", "answer_scores"]:
        fig, axes = plt.subplots(2, 4, figsize=(17, 8.6), squeeze=False)
        for ax, task, title in zip(axes.flat, TASKS, TITLES, strict=True):
            vals = [data["behavior"][s][task] for s in STAGES]
            if kind == "accuracy":
                y = [v["accuracy"] for v in vals]
                ci = [v["accuracy_uncertainty"]["ci"] for v in vals]
                ylabel = (
                    "Group exact-match accuracy"
                    if task in ["ethics_deontology", "ethics_justice", "ethics_virtue"]
                    else "Native accuracy"
                )
                ax.set_ylim(-0.025, 1.025)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            else:
                field = "logit_margin" if task.startswith("ethics_") else "answer_log_prob_nats"
                y = [v[field]["group_weighted_mean"] for v in vals]
                ci = [v[field]["ci"] for v in vals]
                ylabel = (
                    "Correct-label logit margin"
                    if field == "logit_margin"
                    else "Reference log-probability (nats/token)"
                )
                ax.axhline(0, color="#aab0b6", linewidth=0.7)
            ax.plot(x, y, color="#245c85", linewidth=1.7, zorder=3)
            ax.scatter(x, y, c=COLORS, s=36, zorder=4, edgecolors="white", linewidth=0.5)
            ax.vlines(
                x,
                [c[0] for c in ci],
                [c[1] for c in ci],
                color="#245c85",
                alpha=0.75,
                linewidth=1.5,
            )
            ax.set(title=title, xticks=x, xticklabels=LABELS, ylabel=ylabel)
            ax.tick_params(axis="x", labelrotation=40)
            ax.grid(axis="y", alpha=0.16)
            for index in [0, 6]:
                text = f"{y[index]:.1%}" if kind == "accuracy" else f"{y[index]:.2f}"
                ax.annotate(
                    text,
                    (x[index], y[index]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        title = (
            "Native accuracy across training stages"
            if kind == "accuracy"
            else "Answer scores across training stages"
        )
        fig.suptitle(title, x=0.055, y=0.995, ha="left", fontsize=18, fontweight="bold")
        foot = (
            "Original test prompts; all failures remain in the denominator. Bars: 95% source-group bootstrap intervals."
            if kind == "accuracy"
            else "ETHICS: correct-minus-incorrect label logits. CRUX/GSM: mean token log-probability of the reference answer, not a raw logit. Higher is better."
        )
        fig.text(0.055, 0.012, foot, fontsize=10)
        fig.tight_layout(rect=[0, 0.055, 1, 0.955], h_pad=2.5)
        save(fig, kind)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6), squeeze=False)
    handles = {}
    for ax, task, title in zip(axes.flat, TASKS, TITLES, strict=True):
        missing = []
        for i, stage in enumerate(STAGES):
            p = data["probes"][stage].get(task)
            if p is None:
                missing.append(LABELS[i])
                continue
            layers = p["layers"]
            xs = [layer["layer"] for layer in layers]
            label = LABELS[i] + (" *" if p["source"] == "full_size_pilot" else "")
            handles[label] = ax.plot(
                xs,
                [layer["test_accuracy"] for layer in layers],
                color=COLORS[i],
                lw=1.8,
                label=label,
            )[0]
            ax.plot(
                xs,
                [layer["shuffled_test_accuracy_mean"] for layer in layers],
                color=COLORS[i],
                lw=0.8,
                ls=":",
                alpha=0.55,
            )
        ax.set(
            title=title,
            ylim=(-0.025, 1.025),
            xlim=(0, 16),
            xticks=[0, 4, 8, 12, 16],
            xlabel="Residual layer (0 = embedding)",
            ylabel="Held-out group accuracy",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.16)
        if missing:
            ax.text(
                0.03,
                0.04,
                "Unavailable: " + ", ".join(missing),
                transform=ax.transAxes,
                fontsize=8,
                color="#666666",
            )
    fig.suptitle(
        "Linear-probe accuracy by layer and checkpoint",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="lower center",
        bbox_to_anchor=(0.52, 0.033),
        ncol=7,
        frameon=False,
    )
    fig.text(
        0.055,
        0.005,
        "Solid: held-out accuracy. Dotted: five-shuffle baseline. * RLVR2 uses its verified full-size pilot with identical probe inputs/splits. No new probes fitted.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.105, 1, 0.955], h_pad=2.5)
    save(fig, "linear_probe_layers")


def report(data, output):
    math_values = [data["behavior"][s]["gsm_symbolic"]["accuracy"] for s in STAGES]
    values = [r["jaccard"] for task in data["overlap"].values() for r in task.values()]
    lines = [
        f"GSM-Symbolic accuracy: {' → '.join(f'{100 * x:.1f}%' for x in math_values)} (Init → Stage 1 → Stage 2 → SFT → DPO → RLVR1 → RLVR2).",
        f"Consecutive-stage Jaccard@16 ranges from {min(values):.2f} to {max(values):.2f}; overlap describes node-set stability and does not by itself establish acquisition or elicitation.",
        "Probe coverage: all eight tasks through DPO; three tasks at RLVR1; all eight RLVR2 tasks from the verified full-size pilot. No new activation extraction or fitting was performed.",
        "",
        "All seven checkpoints passed the saved-artifact technical audit. Each has 26,568 original test examples plus ETHICS controls, totaling 186,312 behavioral rows per checkpoint (1,304,184 total). Raw outputs, native extraction failures, likelihood scores, top-node arrays, interventions and available probes are preserved.",
        "",
        "| Dataset | Init | Stage 1 | Stage 2 | SFT | DPO | RLVR1 | RLVR2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, title in zip(TASKS, TITLES):
        lines.append(
            "| "
            + title
            + " | "
            + " | ".join(f"{data['behavior'][s][task]['accuracy']:.1%}" for s in STAGES)
            + " |"
        )
    lines.extend(
        [
            "",
            "Jaccard@16 between major training stages",
            "",
            "![Jaccard@16](jaccard16_transitions.png)",
            "",
            "The heatmap uses the actual saved top-16 sets used for interventions, ranked by absolute mean signed attribution over saved examples. The CSV separately reports equal-source-group weighting and its 2,000-draw paired bootstrap interval. That sensitivity can change GSM rankings because source groups have unequal sizes; its interval is not an interval for the primary saved-set statistic. Overlap can also reflect a shared preference for whole MLP nodes rather than task-specific reuse.",
            "",
            "Native accuracy across training stages",
            "",
            "![Accuracy](accuracy.png)",
            "",
            "Original test prompts only. ETHICS deontology, justice and virtue use native group exact-match accuracy. Failed extraction and incorrect execution count as wrong. Intervals resample source groups, not independent training runs.",
            "",
            "Answer scores across training stages",
            "",
            "![Answer scores](answer_scores.png)",
            "",
            "ETHICS uses the correct-label minus incorrect-label logit margin. CRUXEval and GSM use mean token log-probability of the correct reference under the diagnostic answer prompt, in nats/token. These are not raw sequence logits or probabilities of correctness. The GSM diagnostic excludes the target rationale. These scores are distinct from generated-answer correctness.",
            "",
            "Available layerwise linear probes",
            "",
            "![Linear probes](linear_probe_layers.png)",
            "",
            "Solid lines show saved held-out source-group accuracy; dotted lines show shuffled-label controls. High scores shared by controls may reflect label imbalance. Layer 0 is the embedding residual and layers 1–16 follow transformer blocks. The RLVR2 full-size pilot has the same pinned checkpoint, frozen probe identities, labels and held-out splits; its files were checked against the original pilot audit hashes. Five RLVR1 tasks are intentionally absent.",
            "",
            "A concrete saved example",
            "",
            "The first GSM example by ID is shown below for reproducibility, not as a representative sample.",
        ]
    )
    ex = data["examples"]["init"]
    question = ex["prompt"].rsplit("\nQ:", 1)[-1].split("\nA:", 1)[0].strip()
    lines.extend(["", f"`{ex['id']}`: {question}", f"Reference: `{ex['answer']}`.", ""])
    for stage in STAGES:
        e = data["examples"][stage]
        lines.append(f"- {stage}: extracted `{e['prediction']}`; correct={e['correct']}.")
    lines.extend(
        [
            "",
            "Files: PNG, PDF and SVG for every figure; CSV tables for accuracy/scores, stage overlap, probe layers and coverage; `plot_data.json` records metrics, sources and hashes.",
            "",
            "Recovery note: the original completion monitor read a stale running status immediately before observing exit code 0, then stopped the instance. The local backup contains completed metadata and passed structural/numerical audits. A fresh local SHA-256 manifest records the saved files; a final cross-host hash comparison was unavailable because the original GPU could not be restarted. The instance disk remains preserved.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    d = collect(a.run, a.pilot, a.output, a.n_bootstrap)
    plot(d, a.output)
    report(d, a.output)
    print(json.dumps({"status": "complete", "output": str(a.output)}), flush=True)
