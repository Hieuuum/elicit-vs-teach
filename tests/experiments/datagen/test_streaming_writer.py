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

from tests._scriptloader import repo_root

REPO_ROOT = repo_root()
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
@pytest.mark.parametrize("cells", [None, md.P3_CELLS], ids=["grid4x4", "grid8x8"])
def test_streaming_matches_in_memory(spec, cells, tmp_path):
    """Same rows, same order, same hash — the whole contract in one assertion.

    Both grids, because production runs the 8x8 one: the ``cells`` argument is
    threaded separately through ``build_dataset`` and ``build_and_write_streaming``,
    so a divergence there would be exactly the silent kind this file exists for.
    """
    n = 5_000
    blocked: set[tuple[int, str, int]] = set()

    want, want_plan = md.build_dataset(spec, n, blocked, SEED, cells=cells)
    path = tmp_path / f"{spec.name}.parquet"
    got_rep, got_plan = md.build_and_write_streaming(
        spec, n, blocked, SEED, path, chunk=512, cells=cells
    )

    assert got_plan == want_plan
    assert got_rep["order_hash"] == md.order_hash(want)
    assert got_rep["n"] == len(want) == n

    got = pd.read_parquet(path)
    pd.testing.assert_frame_equal(got, pd.DataFrame(want), check_dtype=True)


def test_streaming_matches_in_memory_permuted_labels(tmp_path):
    """The permuted-label branch (added 2026-08-19, ts1b pf-parent B0.1) must
    agree with ``build_dataset``'s own permuted branch row-for-row, the same
    bar ``test_streaming_matches_in_memory`` holds correct-label mode to:
    ``build_dataset`` applies ``permute_labels`` before its own order shuffle,
    keyed by ``spec.name``; the streaming writer does the same over the bare
    triples and carries the pairing through an equal-length shuffle with the
    identical seed, which ``random.shuffle``'s position-only semantics make
    equivalent regardless of what the shuffled elements are.
    """
    spec = md.P3_INST_SPEC  # DatasetSpec("D_p3_nl_mult", ("*",), "nl", "permuted")
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
    # Not a no-op: permuted labels genuinely disagree with the true answer
    # somewhere in a 5,000-row draw.
    assert (got["shown_answer"] != got["true_answer"]).any()


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


def test_streaming_refuses_non_correct_or_permuted_labels():
    """The random mode keys labels off the pre-shuffle index in a way the
    streaming writer does not implement; refuse loudly. (Permuted mode IS
    supported as of 2026-08-19, owner ts1b pf-parent B0.1 — see
    test_preteach_4m.py's block/perm/blockperm sections — so this guard now
    excludes only 'random'.)
    """
    random_spec = md.DatasetSpec("D_random_probe", ("+",), "operator", "random")
    with pytest.raises(ValueError, match="correct/permuted-label only"):
        md.build_and_write_streaming(random_spec, 16, set(), SEED, Path("unused.parquet"))


def test_validate_triples_agrees_with_validate():
    """The split-out validator must return what the record-level one returns."""
    records, plan = md.build_dataset(md.P3_TARGET_SPEC, 512, set(), SEED)
    triples = [(r["a"], r["op"], r["b"]) for r in records]
    assert md.validate_triples(triples, set(), plan, md.order_hash(records)) == md.validate(
        records, set(), plan
    )


def test_phase3_grid_is_1_to_8_digits_and_report_covers_every_cell():
    """The 8x8 grid is what bought the pre-exposure headroom; pin it both ways.

    ``validate_triples`` reports ``cell_counts`` over the cell list it is given,
    so passing the 4x4 default for an 8x8 build would silently drop 48 cells
    from the frozen report while every other check still passed.
    """
    assert len(md.P3_CELLS) == 64
    assert md.DIGIT_BANDS[8] == (10_000_000, 99_999_999)

    n = 5_000
    records, plan = md.build_dataset(md.P3_TARGET_SPEC, n, set(), SEED, cells=md.P3_CELLS)
    rep = md.validate(records, set(), plan, cells=md.P3_CELLS)
    assert len(rep["cell_counts"]) == 64
    assert sum(rep["cell_counts"].values()) == n
    assert max(md.digits(r["a"]) for r in records) == 8
