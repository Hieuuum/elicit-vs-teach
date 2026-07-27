"""``build_and_write_streaming`` must produce exactly what ``build_dataset`` does.

The streaming writer exists only because a 1M-row set does not fit in the RAM
available on the generating machine (make_data.py, 2026-07-27). It re-implements
the sampling loop against a compact representation, so the failure it can have
is *silent*: a dataset that is subtly not the one the in-memory path would have
produced, frozen and hashed, with nothing raising. That is what earns a property
test under CLAUDE.md's promotion rule.

Equality is checked on a set small enough to build both ways, across a spec whose
cells both saturate (small cells taken whole, no rejection sampling) and do not
(large cells rejection-sampled), because those are different code paths in
``_sample_pairs``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "experiments" / "training-run" / "datagen" / "make_data.py"
_spec = importlib.util.spec_from_file_location("make_data", _SCRIPT)
md = importlib.util.module_from_spec(_spec)
# @dataclass resolves its module through sys.modules; a path-loaded script that
# is not registered there raises AttributeError at class-creation time.
sys.modules[_spec.name] = md
_spec.loader.exec_module(md)

SEED = 20260727


@pytest.mark.parametrize(
    "spec",
    [md.P3_TARGET_SPEC, md.P3_PARENT_SPEC],
    ids=lambda s: s.name,
)
def test_streaming_matches_in_memory(spec, tmp_path):
    """Same rows, same order, same hash — the whole contract in one assertion."""
    n = 5_000
    blocked: set[tuple[int, str, int]] = set()

    want, want_plan = md.build_dataset(spec, n, blocked, SEED)
    path = tmp_path / f"{spec.name}.parquet"
    got_rep, got_plan = md.build_and_write_streaming(spec, n, blocked, SEED, path, chunk=512)

    assert got_plan == want_plan
    assert got_rep["order_hash"] == md.order_hash(want)
    assert got_rep["n"] == len(want) == n

    got = pd.read_parquet(path)
    pd.testing.assert_frame_equal(got, pd.DataFrame(want), check_dtype=True)


def test_streaming_chunk_size_does_not_change_output(tmp_path):
    """A chunk boundary must not perturb idx, order, or dtypes."""
    n = 1_000
    hashes = set()
    frames = []
    for chunk in (7, 128, n, n * 2):
        path = tmp_path / f"c{chunk}.parquet"
        rep, _ = md.build_and_write_streaming(md.P3_TARGET_SPEC, n, set(), SEED, path, chunk=chunk)
        hashes.add(rep["order_hash"])
        frames.append(pd.read_parquet(path))
    assert len(hashes) == 1
    for other in frames[1:]:
        pd.testing.assert_frame_equal(frames[0], other, check_dtype=True)


def test_streaming_refuses_non_correct_labels():
    """The random/permuted modes key labels off the pre-shuffle index; refuse loudly."""
    with pytest.raises(ValueError, match="correct-label only"):
        md.build_and_write_streaming(md.P3_INST_SPEC, 16, set(), SEED, Path("unused.parquet"))


def test_validate_triples_agrees_with_validate():
    """The split-out validator must return what the record-level one returns."""
    records, plan = md.build_dataset(md.P3_TARGET_SPEC, 512, set(), SEED)
    triples = [(r["a"], r["op"], r["b"]) for r in records]
    assert md.validate_triples(triples, set(), plan, md.order_hash(records)) == md.validate(
        records, set(), plan
    )
