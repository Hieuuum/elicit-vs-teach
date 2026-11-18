"""experiments/unlearning/relearn_predict.py: the relearning measures and the rank correlation (CPU)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT / "experiments" / "unlearning"))
_spec = importlib.util.spec_from_file_location("relearn_predict", _ROOT / "experiments" / "unlearning" / "relearn_predict.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def _pt(test, edl=0.0, mdl=10.0):
    return {"test_loss_nats": test, "edl_ocv_per_token_nats": edl, "mdl_nats": mdl, "label_tokens": 5}


def test_spearman_handles_order_and_ties():
    assert R.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert R.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert abs(R.spearman([1, 1, 2, 2], [1, 2, 1, 2])) < 1e-9


def test_outcomes_are_means_over_seeds_against_the_models_own_endpoint():
    pts = {0: {8: _pt(3.0, 1.0), 32: _pt(2.0), 128: _pt(1.5), 573: _pt(1.0)},
           1: {8: _pt(5.0, 3.0), 32: _pt(2.0), 128: _pt(1.5, mdl=20.0), 573: _pt(1.0)}}
    rot = {0: {8: _pt(0, 0.5)}, 1: {8: _pt(0, 0.5)}}
    o = R.outcomes(pts, rot)
    assert o["gap_8"] == 3.0 and o["gap_32"] == 1.0 and o["mdl128"] == 3.0 and o["kedl"] == 1.5
    assert R.outcomes({}, {}) == {}


def test_report_states_the_preregistered_checks():
    rows = [{"set": "wmdp", "tag": f"m{i}", "s": 0.5 - 0.1 * i, "margin": 1.0, "loss": 2.0, "gap_32": 0.1 * i}
            for i in range(5)]
    out = "\n".join(R.report(rows))
    assert "R1 (s vs gap_32 negative): PASS (-1.00, n=5)" in out and "R2" in out


def test_report_checks_the_tofu_replication_on_gap_8():
    rows = [{"set": "tofu", "tag": t, "s": 0.5 - 0.1 * i, "margin": float(i % 2), "loss": 1.0 + (i % 3), "gap_8": 0.1 * i}
            for i, t in enumerate(("orig", "npo", "graddiff", "rmu", "retain"))]
    out = "\n".join(R.report(rows))
    assert "T1 TOFU (s vs gap_8 negative): PASS (-1.00, n=5)" in out and "T1 without the retain model: -1.00 (n=4)" in out
