"""Tests for `experiments/eapig-circuit-check/compare.py` (PLAN.md "Parent vs
child comparison" + "Plots").

`compare.py` lives under a hyphenated experiment directory, so it is loaded by
file path (the project's `_scriptloader.py` only covers `training-run`; this
file is self-contained instead of extending it).

Uses the real 195,865-edge graph (fixed by model architecture; there's no
smaller substitute) with random score vectors, and small per-example counts
to keep fixtures cheap. CPU only, matplotlib Agg (set inside `compare.py`).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
COMPARE_PATH = ROOT / "experiments" / "eapig-circuit-check" / "compare.py"


def _load_compare():
    spec = importlib.util.spec_from_file_location("eapig_compare", COMPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load_compare()
et = compare.et
SIZES = compare.SIZES


@pytest.fixture(scope="module")
def ctx():
    return compare.build_graph_ctx()


def _rand_mean(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(compare.build_graph_ctx()["graph"].n_edges
                                                         ).astype(np.float32)


E = compare.build_graph_ctx()["graph"].n_edges


def _write_model_dir(
    base: Path,
    tag: str,
    mean: np.ndarray,
    *,
    mean_a: np.ndarray | None = None,
    mean_b: np.ndarray | None = None,
    em: float = 0.9,
    gate_pass: bool = True,
    include_tests: bool = True,
    selected_size: float | None = None,
    parent_mean: np.ndarray | None = None,
) -> None:
    """Write a minimal results/<tag>/{sanity,scores,evaluate} triple."""
    d = base / tag
    d.mkdir(parents=True, exist_ok=True)
    n_val = 8
    n_true = int(round(em * n_val))
    sanity = {
        "n_edges": E,
        "n_val": n_val,
        "exact_match": em,
        "em_per_example": [True] * n_true + [False] * (n_val - n_true),
        "nonstandard_split_share": 0.0,
        "nonstandard_per_example": [False] * n_val,
        "m_full": 5.0,
        "m_full_terms": [2.5, 2.5],
        "m_empty": 0.0,
        "m_empty_terms": [0.0, 0.0],
        "identity_err_full": 1e-5,
        "identity_err_empty": 1e-5,
        "sec_per_circuit_eval": 0.05,
        "token_ids_10": [],
    }
    (d / "sanity.json").write_text(json.dumps(sanity))

    mean_a = mean if mean_a is None else mean_a
    mean_b = mean if mean_b is None else mean_b
    n_disc = 4  # keep fixtures fast; schema wants (512, E) in the real run
    torch.save(
        {
            "mean": torch.as_tensor(mean, dtype=torch.float32),
            "mean_a": torch.as_tensor(mean_a, dtype=torch.float32),
            "mean_b": torch.as_tensor(mean_b, dtype=torch.float32),
            "per_example": torch.randn(n_disc, E, dtype=torch.float32).half(),
        },
        d / "scores.pt",
    )

    rng = np.random.default_rng(0)
    sizes = {}
    for frac in SIZES:
        f_random = rng.normal(0.1, 0.05, size=20).tolist()
        sizes[str(frac)] = {
            "k": et.size_to_k(frac, E),
            "m_circuit": 4.5,
            "m_circuit_terms": [2.2, 2.3],
            "f": 0.9,
            "f_random": f_random,
            "sufficiency": {"pass": True, "win_fraction": 1.0},
            "f_removed": 0.1,
            "f_removed_random": f_random,
        }
    evaluate: dict = {
        "m_full": 5.0,
        "m_empty": 0.0,
        "n_draws": 20,
        "n_draws_ts": 10,
        "sizes": sizes,
        "f_log_mean": 0.9,
        "validity_gate": {"pass": gate_pass, "m_full_pos": True, "f_degenerate": False},
    }
    if include_tests:
        tests = {}
        for i, frac in enumerate(SIZES):
            tests[str(frac)] = {
                "sufficiency": True,
                "equivalence": i >= 1,
                "partial_necessity": True,
                "consistency": i >= 2,
                "specificity": i >= 2,
            }
        evaluate["tests"] = tests
        evaluate["selected_size"] = selected_size if selected_size is not None else SIZES[2]
    if parent_mean is not None:
        evaluate["parent_in_child"] = {
            str(frac): {"f_parent_circuit": 0.5, "f_own": 0.9, "f_random_band": [0.05, 0.1, 0.2]}
            for frac in SIZES
        }
    (d / "evaluate.json").write_text(json.dumps(evaluate))


# --------------------------------------------------------------------------
# Edge/node Jaccard and change rate: identical and disjoint circuits
# --------------------------------------------------------------------------


def test_overlap_identical_scores_full_jaccard_and_zero_change(ctx):
    mean = _rand_mean(1)
    for frac in SIZES:
        ov = compare.overlap_at_size(mean, mean, frac, ctx)
        assert ov["edge_jaccard"] == pytest.approx(1.0)
        assert ov["node_jaccard_all"] == pytest.approx(1.0)
        assert ov["node_jaccard_heads"] == pytest.approx(1.0)
        assert ov["edge_change"] == {"added": 0, "removed": 0, "rate": pytest.approx(0.0)}
        assert ov["node_change"]["rate"] == pytest.approx(0.0)
        assert sum(ov["layer_hist_added"]) == 0
        assert sum(ov["layer_hist_removed"]) == 0


def test_overlap_negated_scores_are_disjoint(ctx):
    # Negating a vector of distinct values swaps top-k for bottom-k; with
    # k << E/2 (true for every SIZES entry on a 195,865-edge graph) the two
    # top-k sets cannot share an index.
    mean = _rand_mean(2)
    neg = -mean
    for frac in SIZES:
        k = et.size_to_k(frac, E)
        ov = compare.overlap_at_size(mean, neg, frac, ctx)
        assert ov["edge_jaccard"] == pytest.approx(0.0)
        assert ov["edge_change"]["added"] == k
        assert ov["edge_change"]["removed"] == k
        assert ov["edge_change"]["rate"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Split-half ceiling
# --------------------------------------------------------------------------


def _fake_entry(mean, mean_a, mean_b):
    return {
        "scores": {
            "mean": torch.as_tensor(mean, dtype=torch.float32),
            "mean_a": torch.as_tensor(mean_a, dtype=torch.float32),
            "mean_b": torch.as_tensor(mean_b, dtype=torch.float32),
        },
        "evaluate": {
            "sizes": {str(f): {"f": 0.5, "sufficiency": {"pass": True}} for f in SIZES},
            "validity_gate": {"pass": True},
        },
        "sanity": {
            "exact_match": 0.9, "n_val": 8, "m_full": 5.0, "m_empty": 0.0,
            "nonstandard_split_share": 0.0,
        },
    }


def test_ceiling_identical_halves_is_one_and_reliable(ctx):
    mean = _rand_mean(3)
    ms = compare.model_summary(_fake_entry(mean, mean, mean), ctx)
    for frac in SIZES:
        assert ms["ceiling"][str(frac)] == pytest.approx(1.0)
        assert ms["ceiling_unreliable"][str(frac)] is False


def test_ceiling_disjoint_halves_is_zero_and_flagged_unreliable(ctx):
    mean = _rand_mean(4)
    ms = compare.model_summary(_fake_entry(mean, mean, -mean), ctx)
    for frac in SIZES:
        assert ms["ceiling"][str(frac)] == pytest.approx(0.0)
        assert ms["ceiling_unreliable"][str(frac)] is True


# --------------------------------------------------------------------------
# Failed-gate route labelling
# --------------------------------------------------------------------------


def test_route_labelled_no_parent_circuit_when_gate_fails(tmp_path, ctx):
    _write_model_dir(tmp_path, "elicit_parent", _rand_mean(10), gate_pass=False,
                      include_tests=False)
    _write_model_dir(tmp_path, "elicit_child", _rand_mean(11), parent_mean=_rand_mean(10))
    models = compare.load_all(tmp_path)
    model_summaries = {t: (compare.model_summary(e, ctx) if e else None) for t, e in models.items()}
    route = compare.route_summary(
        "elicit", "elicit_parent", "elicit_child", models, model_summaries, ctx
    )
    assert route["available"] is True
    assert route["label"] == "no parent circuit"
    assert route["reference_only"] is True
    # overlaps are still computed ("for reference only"), not suppressed
    assert route["sizes"][str(SIZES[0])]["edge_jaccard"] is not None


def test_route_labelled_ok_when_gate_passes(tmp_path, ctx):
    _write_model_dir(tmp_path, "elicit_parent", _rand_mean(12), gate_pass=True)
    _write_model_dir(tmp_path, "elicit_child", _rand_mean(13), parent_mean=_rand_mean(12))
    models = compare.load_all(tmp_path)
    model_summaries = {t: (compare.model_summary(e, ctx) if e else None) for t, e in models.items()}
    route = compare.route_summary(
        "elicit", "elicit_parent", "elicit_child", models, model_summaries, ctx
    )
    assert route["label"] == "ok"
    assert route["reference_only"] is False


# --------------------------------------------------------------------------
# Missing models must not crash
# --------------------------------------------------------------------------


def test_run_compare_missing_route_reports_available(tmp_path):
    results_dir, figures_dir = tmp_path / "results", tmp_path / "figures"
    ep_mean = _rand_mean(20)
    _write_model_dir(results_dir, "elicit_parent", ep_mean)
    _write_model_dir(results_dir, "elicit_child", _rand_mean(21), parent_mean=ep_mean)
    # fmt_parent / teach_child deliberately absent

    out = compare.run_compare(results_dir, figures_dir)

    assert set(out["missing_models"]) == {"fmt_parent", "teach_child"}
    assert out["routes"]["elicit"]["available"] is True
    assert out["routes"]["teach"]["available"] is False
    assert set(out["routes"]["teach"]["missing"]) == {"fmt_parent", "teach_child"}
    assert out["children_reference"]["available"] is False
    assert out["accuracy_gap"]["available"] is False
    assert (results_dir / "compare.json").exists()
    assert (results_dir / "summary.md").exists()
    pngs = {p.name for p in figures_dir.glob("*.png")}
    assert len(pngs) == 5


# --------------------------------------------------------------------------
# Full four-model run, accuracy-gap flag, plots
# --------------------------------------------------------------------------


def test_run_compare_full_four_models(tmp_path):
    results_dir, figures_dir = tmp_path / "results", tmp_path / "figures"
    ep_mean, fp_mean = _rand_mean(30), _rand_mean(32)
    _write_model_dir(results_dir, "elicit_parent", ep_mean)
    _write_model_dir(results_dir, "elicit_child", _rand_mean(31), parent_mean=ep_mean)
    _write_model_dir(results_dir, "fmt_parent", fp_mean, gate_pass=False, include_tests=False)
    _write_model_dir(results_dir, "teach_child", _rand_mean(33), parent_mean=fp_mean, em=0.3)

    out = compare.run_compare(results_dir, figures_dir)

    assert out["missing_models"] == []
    assert out["routes"]["elicit"]["label"] == "ok"
    assert out["routes"]["teach"]["label"] == "no parent circuit"
    assert out["routes"]["teach"]["reference_only"] is True
    assert out["children_reference"]["available"] is True

    gap = out["accuracy_gap"]
    assert gap["available"] is True
    assert gap["gap"] == pytest.approx(0.6, abs=1e-6)  # 0.9 - 0.3
    assert gap["flag"] is True
    assert gap["note"] is not None

    assert (results_dir / "compare.json").exists()
    compare_json = json.loads((results_dir / "compare.json").read_text())
    assert compare_json["routes"]["elicit"]["label"] == "ok"
    assert (results_dir / "summary.md").exists()
    summary = (results_dir / "summary.md").read_text()
    assert "no parent circuit" in summary
    assert "FLAGGED" in summary

    pngs = sorted(p.name for p in figures_dir.glob("*.png"))
    assert pngs == [
        "1_f_vs_size.png", "2_test_grid.png", "3_overlap_vs_size.png",
        "4_change_and_layers.png", "5_parent_in_child.png",
    ]
    for name in pngs:
        assert (figures_dir / name).stat().st_size > 0
