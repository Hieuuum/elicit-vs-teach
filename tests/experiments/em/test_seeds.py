"""Seed bookkeeping of sl_report.py and lls.py: per-seed results are found, paired and summarised (CPU)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "em"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "experiments" / "em" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _run(d: Path, name: str, share: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps({"share_after": share}))


def test_sl_effects_per_seed_directory(tmp_path, capsys):
    R = _load("sl_report")
    sdir = tmp_path / "Qwen2.5-7B-Instruct"
    for d, cat in ((sdir, 0.9), (sdir / "seed1", 0.7)):
        _run(d, "sl-q7b-cat", {"cat": cat})
        _run(d, "sl-q7b-neutral", {"cat": 0.1})
    assert abs(R.effects(sdir)["sl-q7b-cat"] - 0.8) < 1e-9 and abs(R.effects(sdir / "seed1")["sl-q7b-cat"] - 0.6) < 1e-9
    R.seeds_table(tmp_path)
    out = capsys.readouterr().out
    assert "n=2  +0.700  [+0.600, +0.800]" in out


def test_lls_net_gain_pairs_sets_within_a_seed(tmp_path):
    L = _load("lls")
    runs = tmp_path / L.RUNS
    runs.mkdir()
    for sfx, gl, gr in (("", 10, 2), ("-s1", 5, 4)):
        for k, g in (("lls", gl), ("random", gr)):
            (runs / f"M-{k}{sfx}.json").write_text(json.dumps({"mentions_before": {"a": 0}, "mentions_after": {"a": g}}))
    assert L.net_gain(tmp_path, "M", 0) == 8 and L.net_gain(tmp_path, "M", 1) == 1 and L.net_gain(tmp_path, "M", 2) is None
