"""Smoke tests for experiments/eapig-circuit-check/base_control_report.py (synthetic inputs)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[2] / "experiments/eapig-circuit-check/base_control_report.py"
_spec = importlib.util.spec_from_file_location("base_control_report", _P)
bcr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bcr)

K = 100_000  # large k so a 3-decimal Jaccard round-trips through an integer overlap


def _ov(j: float) -> int:
    return round(2 * K * j / (1 + j))


def _entry(js: tuple[float, ...]) -> dict:
    rows = {}
    for f, j in zip(bcr.FRACS, js, strict=True):
        rows[bcr.frac_key(f)] = {
            "overlap": _ov(j),
            "k": K,
            "null": {"n1": {"edge": {"p5": 0.99}}},
        }
    return {"rankings": {"signed": rows}}


def make(ref_scale: float = 1.0) -> tuple[dict, dict]:
    pairs = {}
    for pair, _, _ in bcr.PAIRS:
        exp = bcr.EXPECTED_REF.get(pair, (0.3,) * 6)
        pairs[pair] = _entry(tuple(x * ref_scale if pair in bcr.EXPECTED_REF else x for x in exp))
    for lab in bcr.LABELS:
        pairs[f"{lab}:halves"] = _entry((0.9,) * 6)
    en = {"pairs": pairs}
    boot = {"meta": {}, "pairs": {}}  # overlap_boot.json layout: pairs."A:B".str(frac)
    for pair in pairs:
        if pair.endswith(":halves"):
            continue
        boot["pairs"][pair] = {
            str(f): {
                "point": bcr.edge_jaccard(en, pair, f),
                "ci_lo": 0.1,
                "ci_hi": 0.9,
            }
            for f in bcr.FRACS
        }
    return en, boot


def test_jaccard_formula():
    assert bcr.jaccard(50, 100) == pytest.approx(1 / 3)
    assert bcr.jaccard(100, 100) == 1.0
    assert bcr.jaccard(0, 100) == 0.0


def test_chance_uses_n1_p5():
    en, _ = make()
    # overlap = K * (1 - 0.99) = 0.01 K  ->  J = 0.01 / 1.99
    assert bcr.chance_jaccard(en, 0.01) == pytest.approx(0.01 / 1.99)


def test_table_format():
    en, boot = make()
    t = bcr.build_table(en, boot)
    assert "| base → ec word | 0.300 [0.100, 0.900]" in t
    assert "| split-half ceiling ep op | 0.900 |" in t
    assert t.splitlines()[-1].startswith("| chance (95th pct) | 0.005 |")


def test_checks_pass_on_matching_input():
    en, boot = make()
    assert all(ok for ok, _ in bcr.run_checks(en, boot))


def test_checks_fail_on_wrong_reference():
    en, boot = make(ref_scale=0.9)
    assert not all(ok for ok, _ in bcr.run_checks(en, boot))


def test_checks_fail_on_boot_point_mismatch():
    en, boot = make()
    boot["pairs"]["base:ec_word"]["0.01"]["point"] += 0.01
    res = bcr.run_checks(en, boot)
    assert not all(ok for ok, _ in res)


def test_main_end_to_end(tmp_path, capsys):
    en, boot = make()
    (tmp_path / "en.json").write_text(json.dumps(en))
    (tmp_path / "b.json").write_text(json.dumps(boot))
    argv = [
        "--edge-node",
        str(tmp_path / "en.json"),
        "--boot",
        str(tmp_path / "b.json"),
        "--out-md",
        str(tmp_path / "t.md"),
        "--fig",
        str(tmp_path / "f.png"),
    ]
    assert bcr.main(argv) == 0
    assert (tmp_path / "t.md").exists() and (tmp_path / "f.png").stat().st_size > 0
    assert "FAIL" not in capsys.readouterr().out
