"""Saved-artifact report integration: native scoring, matching, plots and scope."""

import copy
import json
import shutil

import numpy as np
import pytest

from geode.circuits.artifacts import write_json, write_jsonl
from geode.circuits.report import (
    ETHICS_TASKS,
    _intervention_summary,
    _load_circuits,
    generate_report,
)


def make_stage(root, stage, *, n_groups=8, protocol="fixed", flip=False):
    rows = []
    for group in range(n_groups):
        for item in range(2):
            rows.append(
                {
                    "id": f"justice-{group:02d}-{item}",
                    "task": "ethics_justice",
                    "group": f"group-{group}",
                    "prompt": f"Judge statement {group}/{item}",
                    "answer": "yes",
                    "label": 1,
                    "options": ["yes", "no"],
                    "metadata": {"split": "test", "variant": "original", "group_size": 2},
                    "correct": bool(item == 0 or (flip and group < n_groups // 2)),
                    "generation": "yes" if item == 0 else "no",
                    "answer_log_prob_nats": -0.7,
                    "logit_margin": 0.2,
                    "truncated": item == 1,
                    "context_overflow": False,
                }
            )
    write_jsonl(root / stage / "behavior.jsonl", rows)
    (root / stage / "circuits").mkdir()
    scores = np.tile(np.arange(20, dtype=float), (n_groups, 1))
    np.savez(root / stage / "circuits/ethics_justice.npz", scores=scores)
    write_json(
        root / stage / "circuits/ethics_justice.json",
        {
            "item_ids": [f"pair-{i}" for i in range(n_groups)],
            "group_ids": [f"group-{i}" for i in range(n_groups)],
            "node_names": [f"node-{i}" for i in range(20)],
            "protocol_hash": protocol,
            "tokenizer_hash": "fixed-tokenizer",
        },
    )
    return rows


def test_report_preserves_native_group_metric_and_failed_denominators(tmp_path):
    make_stage(tmp_path, "init")
    make_stage(tmp_path, "stage1", flip=True)
    report = generate_report(tmp_path, n_bootstrap=100)
    item = report["behavior"]["init"]["ethics_justice/test/original"]
    assert item["accuracy"] == 0  # one wrong answer invalidates a native group
    assert item["row_accuracy"] == 0.5
    assert item["n_examples"] == 16 and item["truncation_rate"] == 0.5
    assert report["behavior"]["stage1"]["ethics_justice/test/original"]["accuracy"] == 0.5
    comparison = report["paired_behavior"]["init -> stage1"]["ethics_justice/test/original"]
    assert comparison["correct"]["difference_b_minus_a"] == 0.5
    assert report["circuit_comparisons"]["init -> stage1"]["ethics_justice"]["jaccard"] == 1
    assert report["overlap_matrices"]["ethics_justice"] == [[1, 1], [1, 1]]
    assert len(report["tldr"]) == 3
    assert (tmp_path / "report.md").read_text().splitlines()[:3] == report["tldr"]
    assert json.loads((tmp_path / "report.json").read_text())["stages"] == ["init", "stage1"]
    for plot in report["plots"]:
        assert (tmp_path / plot["path"]).stat().st_size > 100
        assert plot["caption"] and plot["takeaway"] and plot["confusing"]
        if plot["example"]:
            assert plot["example"]["id"] == "justice-00-0"
        else:
            assert (
                plot.get("circuit_example")
                or plot.get("probe_example")
                or plot.get("intervention_example")
            )


def test_report_never_compares_mismatched_circuit_protocols(tmp_path):
    make_stage(tmp_path, "init", protocol="one")
    make_stage(tmp_path, "sft", protocol="two")
    report = generate_report(tmp_path, n_bootstrap=100)
    compared = report["circuit_comparisons"]["init -> sft"]["ethics_justice"]
    assert compared["status"] == "inconclusive"
    assert "protocol_hash" in compared["reason"]
    assert report["overlap_matrices"]["ethics_justice"] == [[1, None], [None, 1]]


def _add_math_rows(root, stage, rows, *, n_problems=16, n_correct=8):
    math_rows = [
        {
            "id": f"math-{i}",
            "task": "gsm_symbolic",
            "group": f"template-{i // 2}",
            "prompt": f"What is {i} + 1?",
            "answer": str(i + 1),
            "generation": str(i + 1 if i < n_correct else -1),
            "correct": i < n_correct,
            "metadata": {"split": "test", "variant": "original"},
            "answer_log_prob_nats": -1.0,
        }
        for i in range(n_problems)
    ]
    write_jsonl(root / stage / "behavior.jsonl", rows + math_rows)


def test_multistage_tldr_reports_math_sequence_and_only_adjacent_typed_nulls(tmp_path, monkeypatch):
    monkeypatch.setattr("geode.circuits.report._plots", lambda *_: [])
    for stage, correct in (("init", 0), ("stage1", 8), ("stage2", 12)):
        rows = make_stage(tmp_path, stage)
        _add_math_rows(tmp_path, stage, rows, n_correct=correct)
    write_json(tmp_path / "run_metadata.json", {"config": {"mode": "pilot"}})
    report = generate_report(tmp_path, n_bootstrap=100)
    assert report["tldr"][0] == (
        "Math accuracy: init 0.0% → stage1 50.0% → stage2 75.0%; "
        "16 problems/8 templates per checkpoint."
    )
    assert "Jaccard@16 is 1.000–1.000; 2/2 exceed" in report["tldr"][1]
    assert "node-type-preserving" in report["tldr"][1]
    assert "unadjusted for multiple comparisons" in report["tldr"][1]
    assert report["tldr"][2].startswith("Technical sanity sample across 3 checkpoints:")
    assert "cannot establish acquisition or elicitation" in report["tldr"][2]
    assert (tmp_path / "report.md").read_text().splitlines()[:3] == report["tldr"]


def test_multistage_tldr_keeps_missing_math_and_different_denominators_explicit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("geode.circuits.report._plots", lambda *_: [])
    for stage in ("init", "stage1", "stage2"):
        rows = make_stage(tmp_path, stage, n_groups=2)
        if stage != "stage1":
            _add_math_rows(
                tmp_path, stage, rows, n_problems=8 if stage == "init" else 16, n_correct=4
            )
    report = generate_report(tmp_path, n_bootstrap=100)
    assert report["tldr"][0] == (
        "Math accuracy: init 50.0% (8 problems/4 templates) → stage1 unavailable → "
        "stage2 25.0% (16 problems/8 templates)."
    )
    assert report["tldr"][1].startswith("No adjacent available checkpoint/task")
    assert report["tldr"][2].startswith("Saved evaluation sample across 3 checkpoints:")


def test_two_stage_math_tldr_remains_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr("geode.circuits.report._plots", lambda *_: [])
    for stage, correct in (("stage2", 8), ("rlvr2", 12)):
        rows = make_stage(tmp_path, stage)
        _add_math_rows(tmp_path, stage, rows, n_correct=correct)
    report = generate_report(tmp_path, n_bootstrap=100)
    assert report["tldr"][0].startswith(
        "Two-checkpoint pilot: math accuracy 50.0% → 75.0% on 16 problems/8 templates; "
        "paired 95% interval for the gain "
    )
    assert report["tldr"][1].startswith("Observed circuit overlaps are 1.000–1.000; 1/1 exceed")
    assert report["tldr"][2].endswith(
        "these endpoints cannot locate the responsible intervening training phase."
    )


def test_report_rejects_incomplete_native_groups(tmp_path):
    rows = make_stage(tmp_path, "init")
    write_jsonl(tmp_path / "init/behavior.jsonl", rows[:-1])
    with pytest.raises(ValueError, match="incomplete native"):
        generate_report(tmp_path, n_bootstrap=100)


def test_report_marks_small_group_samples_inconclusive_without_bogus_ci(tmp_path):
    make_stage(tmp_path, "init", n_groups=2)
    make_stage(tmp_path, "stage1", n_groups=2)
    report = generate_report(tmp_path, n_bootstrap=100)
    uncertainty = report["behavior"]["init"]["ethics_justice/test/original"]["accuracy_uncertainty"]
    assert uncertainty["status"] == "inconclusive" and "ci" not in uncertainty
    assert (
        report["circuit_comparisons"]["init -> stage1"]["ethics_justice"]["status"]
        == "inconclusive"
    )
    assert not report["overlap_matrices"]


def test_ethics_controls_remain_separate_and_do_not_replicate_primary_groups(tmp_path):
    rows = make_stage(tmp_path, "init")
    control = copy.deepcopy(rows)
    for row in control:
        row["id"] += ":control"
        row["metadata"]["variant"] = "control_v0_p0_m0"
        row["correct"] = True
    write_jsonl(tmp_path / "init/behavior.jsonl", rows + control)
    report = generate_report(tmp_path, n_bootstrap=100)
    primary = report["behavior"]["init"]["ethics_justice/test/original"]
    secondary = report["behavior"]["init"]["ethics_justice/test/control_v0_p0_m0"]
    assert primary["accuracy"] == 0 and secondary["accuracy"] == 1
    assert primary["n_groups"] == secondary["n_groups"] == 8


def test_balanced_ethics_controls_average_native_group_scores_before_bootstrap(tmp_path):
    rows = make_stage(tmp_path, "init")
    controls = []
    for variant in range(8):
        for row in copy.deepcopy(rows):
            row["id"] += f":control-{variant}"
            row["metadata"]["variant"] = f"control_{variant}"
            row["correct"] = int(row["group"].split("-")[-1]) < 4 and variant % 2 == 0
            controls.append(row)
    write_jsonl(tmp_path / "init/behavior.jsonl", rows + controls)
    report = generate_report(tmp_path, n_bootstrap=100)
    item = report["behavior"]["init"]["ethics_justice/test/balanced_controls"]
    assert item["accuracy"] == 0.25
    assert item["n_groups"] == 8 and item["n_variants"] == 8
    assert item["accuracy_uncertainty"]["ci"][1] > item["accuracy_uncertainty"]["ci"][0]
    assert "faint crosses" in next(
        plot["caption"] for plot in report["plots"] if plot["path"] == "plots/behavior.png"
    )


def test_missing_confidence_and_skipped_probes_remain_explicit(tmp_path):
    rows = make_stage(tmp_path, "init")
    for row in rows:
        row["answer_log_prob_nats"] = None
    write_jsonl(tmp_path / "init/behavior.jsonl", rows)
    write_json(
        tmp_path / "init/probes/ethics_justice.json",
        {"status": "skipped", "reason": "too few groups"},
    )
    report = generate_report(tmp_path, n_bootstrap=100)
    assert (
        report["behavior"]["init"]["ethics_justice/test/original"]["answer_log_prob_nats"]["status"]
        == "unavailable"
    )
    assert report["probes"]["init"]["ethics_justice"]["reason"] == "too few groups"


def test_unmatched_behavioral_prompts_prevent_paired_metric_comparison(tmp_path):
    make_stage(tmp_path, "init")
    rows = make_stage(tmp_path, "stage1")
    rows[0]["prompt"] = "a different instruction"
    write_jsonl(tmp_path / "stage1/behavior.jsonl", rows)
    report = generate_report(tmp_path, n_bootstrap=100)
    compared = report["paired_behavior"]["init -> stage1"]["ethics_justice/test/original"]
    assert compared["status"] == "inconclusive"
    assert "unmatched behavioral" in compared["reason"]


def test_report_refuses_empty_run(tmp_path):
    with pytest.raises(ValueError, match="No saved"):
        generate_report(tmp_path, n_bootstrap=100)


def test_probe_plot_ci_resamples_heldout_source_groups(tmp_path):
    make_stage(tmp_path, "init")
    write_json(
        tmp_path / "init/probes/ethics_justice.json",
        {
            "layers": [
                {
                    "layer": 0,
                    "test_accuracy": 0.5,
                    "shuffled_test_accuracy_mean": 0.5,
                    "predictions": [0, 1] * 4,
                    "test_labels": [0, 0] * 4,
                    "test_group_ids": ["a", "a", "b", "b", "c", "c", "d", "d"],
                }
            ],
        },
    )
    report = generate_report(tmp_path, n_bootstrap=100)
    uncertainty = report["probes"]["init"]["ethics_justice"]["layers"][0][
        "test_accuracy_uncertainty"
    ]
    assert uncertainty["ci"] == [0.5, 0.5]
    assert "conditional" in uncertainty["scope"]
    assert any(plot["path"] == "plots/probes.png" for plot in report["plots"])
    summary = report["probes"]["init"]["ethics_justice"]
    assert "predictions" not in summary["layers"][0]
    assert len(summary["source_sha256"]) == 64
    original = json.loads((tmp_path / summary["source_artifact"]).read_text())
    assert original["layers"][0]["predictions"] == [0, 1] * 4


def test_ethics_domain_top_nodes_weight_tasks_equally_not_by_number_of_rows(tmp_path, monkeypatch):
    for stage in ("init", "stage1"):
        make_stage(tmp_path, stage)
        for task in ETHICS_TASKS:
            n = 4 if task == ETHICS_TASKS[0] else 20
            scores = np.zeros((n, 20))
            scores[:, :15] = 1000
            if task == ETHICS_TASKS[0] and stage == "init":
                scores[:, 15] = 30
            elif task != ETHICS_TASKS[0]:
                scores[:, 16] = 4
            path = tmp_path / stage / "circuits" / f"{task}.npz"
            np.savez(path, scores=scores)
            write_json(
                path.with_suffix(".json"),
                {
                    "item_ids": [f"pair-{i}" for i in range(n)],
                    "group_ids": [f"group-{i}" for i in range(n)],
                    "node_names": [f"node-{i}" for i in range(20)],
                    "protocol_hash": task,
                    "tokenizer_hash": "fixed-tokenizer",
                },
            )
    monkeypatch.setattr("geode.circuits.report._plots", lambda *args: [])
    report = generate_report(tmp_path, n_bootstrap=100)
    domain = report["circuit_comparisons"]["init -> stage1"]["ethics_domain"]
    assert domain["status"] == "ok"
    assert "node-15" in domain["top_a"] and "node-16" not in domain["top_a"]
    assert "node-16" in domain["top_b"] and "node-15" not in domain["top_b"]
    assert domain["jaccard"] == pytest.approx(15 / 17)
    assert domain["n_groups"] == 84


def test_partial_ethics_tasks_do_not_masquerade_as_full_domain_map(tmp_path):
    make_stage(tmp_path, "init")
    circuits = _load_circuits(tmp_path / "init")
    assert circuits["ethics_domain"]["status"] == "inconclusive"
    assert "missing" in circuits["ethics_domain"]["reason"]


def make_interventions(root, repeats=1):
    make_stage(root, "init")
    meta_path = root / "init/circuits/ethics_justice.json"
    meta = json.loads(meta_path.read_text())
    meta["clean_group_ids"] = [f"ranking-clean-{i}" for i in range(8)]
    meta["corrupt_group_ids"] = [f"ranking-corrupt-{i}" for i in range(8)]
    write_json(meta_path, meta)
    rows = []
    for i in range(8):
        for variant in range(repeats):
            rows.append(
                {
                    "clean_id": f"check-{i}-clean-v{variant}",
                    "corrupt_id": f"check-{i}-corrupt-v{variant}",
                    "clean_group": f"diagnostic-clean-{i}",
                    "corrupt_group": f"diagnostic-corrupt-{i}",
                    "top": {"clean_metric": 3, "patched_metric": 1, "effect": -2, "scale": 1},
                    "random": {
                        "clean_metric": 3,
                        "patched_metric": 2.5,
                        "effect": -0.5,
                        "scale": 1,
                    },
                    "random_nodes": [f"node-{i}" for i in range(16)],
                }
            )
    path = root / "init/circuits/ethics_justice_interventions.json"
    write_json(path, rows)
    return path, rows


def test_interventions_report_paired_disjoint_group_ci_and_actual_example(tmp_path):
    make_interventions(tmp_path, repeats=8)
    report = generate_report(tmp_path, n_bootstrap=100)
    item = report["interventions"]["init"]["ethics_justice"]
    assert item["status"] == "ok" and item["n_groups"] == 8 and item["n_rows"] == 64
    assert item["top_minus_random"] == -1.5
    assert item["ci_top_minus_random"] == [-1.5, -1.5]
    plot = next(plot for plot in report["plots"] if plot["path"] == "plots/interventions.png")
    assert plot["intervention_example"]["clean_id"] == "check-0-clean-v0"


@pytest.mark.parametrize(
    "failure", ["overlap", "baseline", "random_size", "missing_groups", "shared_corrupt"]
)
def test_invalid_or_dependent_causal_checks_are_not_reported_as_independent_evidence(
    tmp_path, failure
):
    path, rows = make_interventions(tmp_path)
    if failure == "overlap":
        rows[0]["clean_group"] = "ranking-clean-0"
    elif failure == "baseline":
        rows[0]["random"]["clean_metric"] = 4
    elif failure == "random_size":
        rows[0]["random_nodes"] = ["node-0"]
    elif failure == "missing_groups":
        rows[0].pop("clean_group")
    elif failure == "shared_corrupt":
        for row in rows:
            row["corrupt_group"] = "shared-corruption"
    write_json(path, rows)
    report = _intervention_summary(tmp_path / "init", _load_circuits(tmp_path / "init"), 100, 0)
    assert report["ethics_justice"]["status"] == "inconclusive"
    assert "ci_top_minus_random" not in report["ethics_justice"]


def make_type_matched_interventions(root, repeats=1):
    path, rows = make_interventions(root, repeats=repeats)
    meta_path = root / "init/circuits/ethics_justice.json"
    meta = json.loads(meta_path.read_text())
    mlps = [f"layer.{i}.mlp" for i in range(4)]
    heads = [f"layer.0.attn.{i}" for i in range(16)]
    meta["node_names"] = mlps + heads
    meta["top16"] = mlps[:3] + heads[:13]
    write_json(meta_path, meta)
    for row in rows:
        index = int(row["clean_id"].split("-")[1])
        effect = -0.5 - index / 10
        row["random_nodes"] = meta["node_names"][:16]
        row["random_type_matched_nodes"] = mlps[1:] + heads[3:]
        row["random_type_matched"] = {
            "clean_metric": 3,
            "patched_metric": 3 + effect,
            "effect": effect,
            "scale": 1,
        }
    write_json(path, rows)
    return path, rows, meta_path


def test_type_matched_interventions_report_both_comparators_and_paired_cluster_ci(tmp_path):
    make_type_matched_interventions(tmp_path, repeats=8)
    report = generate_report(tmp_path, n_bootstrap=100)
    item = report["interventions"]["init"]["ethics_justice"]
    assert item["status"] == "ok"
    assert item["top_minus_random"] == -1.5  # Legacy uniform comparator unchanged.
    typed = item["random_type_matched"]
    assert typed["status"] == "ok"
    assert typed["n_groups"] == 8 and typed["n_rows"] == 64
    assert typed["top_minus_random"] == pytest.approx(-1.15)
    assert typed["mean_random_effect"] == pytest.approx(-0.85)
    assert typed["ci_top_minus_random"][0] < -1.15 < typed["ci_top_minus_random"][1]
    assert typed["top_node_type_counts"] == {"attention_head": 13, "mlp": 3}
    plot = next(p for p in report["plots"] if p["path"] == "plots/interventions.png")
    assert "size-and-node-type-matched" in plot["caption"]
    assert "does not prove acquisition" in plot["takeaway"]
    assert "random_type_matched" in plot["intervention_example"]


def test_multistage_facets_preserve_each_overlap_and_intervention_point(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    make_type_matched_interventions(tmp_path)
    write_json(
        tmp_path / "init/probes/ethics_justice.json",
        {
            "layers": [
                {
                    "layer": 0,
                    "test_accuracy": 0.5,
                    "shuffled_test_accuracy_mean": 0.5,
                    "predictions": [0, 1] * 4,
                    "test_labels": [0, 0] * 4,
                    "test_group_ids": ["a", "a", "b", "b", "c", "c", "d", "d"],
                    "split_n_groups": {"test": 4},
                }
            ]
        },
    )
    for stage in ("stage1", "stage2"):
        shutil.copytree(tmp_path / "init", tmp_path / stage)
    saved = {}
    original_save = Figure.savefig

    def capture(fig, path, **kwargs):
        if path.name == "probes.png":
            assert fig.axes[0].get_legend() is None
            assert [text.get_text() for text in fig.legends[0].get_texts()] == [
                "init",
                "stage1",
                "stage2",
            ]
            assert "Held-out source groups: 4" in fig.axes[0].get_title()
            saved[path.name] = True
        if path.name in {"overlap.png", "interventions.png"}:
            assert fig.get_figwidth() <= 13
            saved[path.name] = [
                (line.get_marker(), list(line.get_xdata()), list(line.get_ydata()))
                for line in fig.axes[0].lines
            ]
        return original_save(fig, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    generate_report(tmp_path, n_bootstrap=100)
    overlap = [(x, y) for marker, x, y in saved["overlap.png"] if marker == "o"]
    assert overlap == [([0], [1.0]), ([1], [1.0])]
    interventions = saved["interventions.png"]
    assert saved["probes.png"]
    assert [x[0] for marker, x, _ in interventions if marker == "o"] == [-0.12, 0.88, 1.88]
    assert [x[0] for marker, x, _ in interventions if marker == "s"] == [0.12, 1.12, 2.12]
    assert all(y == [-1.5] for marker, _, y in interventions if marker == "o")
    assert "type-matched random effect" in (tmp_path / "report.md").read_text()


def test_type_matched_ci_does_not_treat_repeated_control_rows_as_independent(tmp_path):
    results = []
    for repeats in (1, 8):
        root = tmp_path / str(repeats)
        make_type_matched_interventions(root, repeats=repeats)
        summary = _intervention_summary(root / "init", _load_circuits(root / "init"), 100, 0)
        results.append(summary["ethics_justice"]["random_type_matched"])
    assert results[0]["n_groups"] == results[1]["n_groups"] == 8
    assert results[0]["ci_top_minus_random"] == pytest.approx(
        results[1]["ci_top_minus_random"], abs=1e-14
    )


def test_legacy_interventions_keep_uniform_results_and_mark_missing_type_control(tmp_path):
    make_interventions(tmp_path)
    summary = _intervention_summary(tmp_path / "init", _load_circuits(tmp_path / "init"), 100, 0)
    item = summary["ethics_justice"]
    assert item["status"] == "ok" and item["top_minus_random"] == -1.5
    assert item["random_type_matched"]["status"] == "unavailable"
    assert "ci_top_minus_random" not in item["random_type_matched"]


@pytest.mark.parametrize(
    "failure", ["composition", "missing_row", "baseline", "scale", "effect", "top_ids"]
)
def test_invalid_type_control_does_not_replace_valid_uniform_results(tmp_path, failure):
    path, rows, meta_path = make_type_matched_interventions(tmp_path)
    if failure == "composition":
        rows[0]["random_type_matched_nodes"] = [f"layer.0.attn.{i}" for i in range(16)]
    elif failure == "missing_row":
        rows[0].pop("random_type_matched")
    elif failure == "baseline":
        rows[0]["random_type_matched"]["clean_metric"] = 4
    elif failure == "scale":
        rows[0]["random_type_matched"]["scale"] = 0.5
    elif failure == "effect":
        rows[0]["random_type_matched"]["effect"] = 0
    elif failure == "top_ids":
        meta = json.loads(meta_path.read_text())
        meta.pop("top16")
        write_json(meta_path, meta)
    write_json(path, rows)
    summary = _intervention_summary(tmp_path / "init", _load_circuits(tmp_path / "init"), 100, 0)
    item = summary["ethics_justice"]
    assert item["status"] == "ok" and item["top_minus_random"] == -1.5
    assert item["random_type_matched"]["status"] == "inconclusive"
    assert "ci_top_minus_random" not in item["random_type_matched"]
