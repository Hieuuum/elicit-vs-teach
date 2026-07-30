"""V5.72 — geode.probe dump-step iterator + row-alignment guard.

``load_probe_dumps`` replaces the hand-rolled ``step_*`` glob the analysis
drivers and ``extract.py`` duplicated; ``assert_probe_alignment`` is the
row-alignment guard (spec 00 §6) those drivers inlined as ``_guard_probe_hash``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geode.probe import assert_probe_alignment, load_probe_dumps


def _make_step_dirs(root: Path, steps: list[int], marker: str = "meta.json") -> None:
    for k in steps:
        d = root / f"step_{k}"
        d.mkdir(parents=True)
        (d / marker).write_text("{}")


def test_v5_72_load_probe_dumps_ascending(tmp_path: Path) -> None:
    root = tmp_path / "probe"
    _make_step_dirs(root, [5, 2, 10, 1])  # created out of order
    (root / "step_99").mkdir()  # no meta.json marker -> skipped
    assert load_probe_dumps(root) == [1, 2, 5, 10]


def test_v5_72_load_probe_dumps_marker_any_present(tmp_path: Path) -> None:
    # Snapshot dirs (adapters.py / extract.py) carry adapter OR model safetensors.
    root = tmp_path / "snapshots"
    (root / "step_3").mkdir(parents=True)
    (root / "step_3" / "adapter.safetensors").write_text("x")
    (root / "step_1").mkdir()
    (root / "step_1" / "model.safetensors").write_text("x")
    (root / "step_2").mkdir()  # neither marker -> skipped
    steps = load_probe_dumps(root, marker=("adapter.safetensors", "model.safetensors"))
    assert steps == [1, 3]


def test_v5_72_load_probe_dumps_empty(tmp_path: Path) -> None:
    root = tmp_path / "probe"
    root.mkdir()
    assert load_probe_dumps(root) == []


def test_v5_72_probe_alignment_hash_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="probe_set_hash"):
        assert_probe_alignment("aa" * 32, "bb" * 32, run_id="evt-x", step=7)


def test_v5_72_probe_alignment_match_passes() -> None:
    assert_probe_alignment("aa" * 32, "aa" * 32, run_id="evt-x", step=7)  # no raise
