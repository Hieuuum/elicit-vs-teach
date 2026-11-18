"""edl_sweep.py: the knowledge-specific curve (true minus rotated, seed-paired) and the M21 series.

Property (decisions.md 2026-10-15): whatever cost both sweeps pay alike (format, the unlock of a
safeguard) cancels point by point, so adding any seed-paired curve to both the true and the
rotated sweep leaves the knowledge curve and its call unchanged; a hump that lives only in the
shared part disappears from it, one that lives in the true sweep alone survives.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
_DIR = _ROOT / "experiments" / "unlearning"


def _load():
    if "edl_sweep" in sys.modules:
        return sys.modules["edl_sweep"]
    spec = importlib.util.spec_from_file_location("edl_sweep", _DIR / "edl_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["edl_sweep"] = mod
    spec.loader.exec_module(mod)
    return mod


ES = _load()
NS = [8, 16, 32, 64, 128]


def _write(store: Path, tag: str, seed: int, n: int, ocv: float, test: float, m21: float | None = None) -> None:
    d = store / "runs" / f"wmdp-edl-{tag}-n{n}-s{seed}"
    d.mkdir(parents=True)
    r = {"edl_ocv_per_token_nats": ocv, "edl_test_per_token_nats": test, "label_tokens": 10 * n, "mdl_nats": 25.0 * n}
    if m21 is not None:
        r.update({"m21_test_nats": m21, "m21_rotations": 4})
    (d / "manifest.json").write_text(json.dumps({"result": r}))


def _populate(store: Path, tag: str, true_curve, rot_curve, m21_true=None, m21_rot=None, seeds=(1, 2)) -> None:
    for s in seeds:
        for i, n in enumerate(NS):
            _write(store, tag, s, n, true_curve[i], true_curve[i] - 0.3, None if m21_true is None else m21_true[i])
            _write(store, f"{tag}-shuf", s, n, rot_curve[i], rot_curve[i] - 0.3, None if m21_rot is None else m21_rot[i])


def test_knowledge_curve_is_the_seed_paired_difference_and_ignores_what_both_sweeps_pay(tmp_path):
    facts = [1.0, 0.8, 0.6, 0.5, 0.4]                     # falling: the facts are read out
    unlock = [0.5, 1.5, 3.0, 1.0, 0.2]                    # a hump both sweeps pay (the safeguard)
    _populate(tmp_path, "u", [f + u for f, u in zip(facts, unlock)], unlock)
    k = ES.knowledge(ES.load(tmp_path, "u"), ES.load(tmp_path, "u-shuf"))
    assert k["ns"] == NS and k["seeds"] == [1, 2]
    for f in ("ocv", "test"):
        for c in k["floors"][f]["curves_nats"].values():
            assert c == pytest.approx(facts)
        assert k["floors"][f]["call"] == "decreasing"
        assert k["floors"][f]["sign"]["read"] == "gate" and k["floors"][f]["sign"]["crossing_n"] is None
    assert k["mdl_diff"]["mean_nats"] == pytest.approx([0.0] * len(NS))    # the synthetic points share one MDL
    # the true sweep alone reads the hump
    t = ES.summarize(ES.load(tmp_path, "u"))
    assert t["floors"]["ocv"]["call"] == "increasing"
    # adding one more seed-paired cost to both leaves the knowledge curve untouched
    _populate(tmp_path, "v", [f + u + 2.0 * i for i, (f, u) in enumerate(zip(facts, unlock))],
              [u + 2.0 * i for i, u in enumerate(unlock)])
    k2 = ES.knowledge(ES.load(tmp_path, "v"), ES.load(tmp_path, "v-shuf"))
    for s_, c in k["floors"]["ocv"]["curves_nats"].items():
        assert k2["floors"]["ocv"]["curves_nats"][s_] == pytest.approx(c)


def test_a_hump_in_the_true_sweep_alone_survives_the_subtraction(tmp_path):
    taught = [1.0, 1.5, 2.5, 1.2, 0.6]                    # the facts themselves are taught
    rot = [0.4, 0.4, 0.4, 0.4, 0.4]
    _populate(tmp_path, "t", [a + b for a, b in zip(taught, rot)], rot)
    k = ES.knowledge(ES.load(tmp_path, "t"), ES.load(tmp_path, "t-shuf"))
    assert k["floors"]["ocv"]["call"] == "increasing"
    assert k["floors"]["ocv"]["rise_at"] == (16, 32)


def test_knowledge_needs_shared_seeds_and_sizes_and_m21_series_pass_through(tmp_path):
    _populate(tmp_path, "w", [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], m21_true=[0.1, 0.2, 0.4, 0.8, 0.9],
              m21_rot=[0.1, 0.1, 0.1, 0.1, 0.1], seeds=(1,))
    for n in NS:
        _write(tmp_path, "w-shuf", 2, n, 1.0, 0.7)        # a rotated seed with no true partner
    pt, pr = ES.load(tmp_path, "w"), ES.load(tmp_path, "w-shuf")
    k = ES.knowledge(pt, pr)
    assert k["seeds"] == [1]
    assert ES.knowledge(pt, {}) is None
    s = ES.summarize(pt)
    assert s["m21"]["mean_nats"] == pytest.approx([0.1, 0.2, 0.4, 0.8, 0.9]) and s["m21"]["rotations"] == 4
    assert "m21" not in ES.summarize(pr)                   # seed 2 has no M21: the series is not reported


def test_knowledge_block_carries_the_v1_13_read_and_the_m21_half_rise(tmp_path):
    taught = [0.1, 0.05, 0.4, 1.2, 1.6]                   # the hump survives: teach
    rot = [0.4] * 5
    _populate(tmp_path, "t", [a + b for a, b in zip(taught, rot)], rot, m21_true=[0.1, 0.1, 0.5, 1.0, 1.1])
    k = ES.knowledge(ES.load(tmp_path, "t"), ES.load(tmp_path, "t-shuf"))
    assert k["floors"]["ocv"]["sign"]["read"] == "teach" and k["floors"]["ocv"]["sign"]["max_n"] == 128
    assert ES.summarize(ES.load(tmp_path, "t"))["m21"]["half_rise_n"] == 64
