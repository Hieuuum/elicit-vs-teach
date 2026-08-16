"""D_target_4M: the paper-faithful App. E.2 pre-teach parent corpus.

``make_data.py --preteach-4m`` builds a 4,000,000-row independent draw of the
same task/format as D_target (operator, add/sub, correct labels), excluding
only the frozen probe and both eval sets — never D_target/D_algo, whose
overlap is instead measured into a ``.overlap.json`` sidecar. Its silent
failure modes are the promotion-rule kind (CLAUDE.md): a leaked eval question,
a duplicate triple, a wrong overlap fraction, or a stale hash pin silently
accepted would all corrupt the science with nothing crashing. Exercised here
at n in the thousands, never the real 4M file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

import pandas as pd
import pytest

from tests._scriptloader import repo_root

REPO_ROOT = repo_root()
_SCRIPT = REPO_ROOT / "experiments" / "training-run" / "datagen" / "make_data.py"
# A distinct module name (the @dataclass in make_data resolves its module through
# sys.modules at class-creation time), so this load never clobbers another
# test file's.
_spec = importlib.util.spec_from_file_location("make_data_preteach4m", _SCRIPT)
md = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = md
_spec.loader.exec_module(md)

SEED = 20260816


def _build_fixture_out(out, seed: int, n_eval: int = 300):
    """Write a tiny but real-schema frozen probe/D_target_eval/D_algo_eval
    trio into ``out``, with a matching report.json pinning their real
    ``order_hash`` — the disk state ``make_preteach_4m`` expects in --out.

    Returns ``(report_dict, probe_triples, target_eval_triples, algo_eval_triples)``.
    """
    probe_records, probe_triples = md.build_probe(seed)
    pd.DataFrame(probe_records).to_parquet(out / "probe.parquet", index=False)
    probe_pin = md.order_hash(probe_records)

    te_records, _ = md.build_dataset(md.EVAL_SPEC, n_eval, probe_triples, seed)
    pd.DataFrame(te_records).to_parquet(out / "D_target_eval.parquet", index=False)
    te_pin = md.order_hash(te_records)
    te_triples = {(r["a"], r["op"], r["b"]) for r in te_records}

    ae_records, _ = md.build_dataset(md.NL_EVAL_SPEC, n_eval, probe_triples | te_triples, seed)
    pd.DataFrame(ae_records).to_parquet(out / "D_algo_eval.parquet", index=False)
    ae_pin = md.order_hash(ae_records)
    ae_triples = {(r["a"], r["op"], r["b"]) for r in ae_records}

    report = {
        "scale": "test-fixture",
        "seed": seed,
        "probe": {"n": len(probe_records), "probe_set_hash": probe_pin},
        "datasets": {
            "D_target_eval": {"order_hash": te_pin},
            "D_algo_eval": {"order_hash": ae_pin},
        },
    }
    (out / "report.json").write_text(json.dumps(report))
    return report, probe_triples, te_triples, ae_triples


def _args(out, seed: int = SEED, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(out=out, seed=seed, dry_run=dry_run)


# --- constants -------------------------------------------------------------


def test_preteach_4m_constants():
    assert md.PRETEACH_4M_N == 4_000_000
    assert md.PRETEACH_4M_SEED == 20260816
    assert md.PRETEACH_4M_EXCLUDES == ("probe", "D_target_eval", "D_algo_eval")
    assert md.PRETEACH_4M_SPEC == md.DatasetSpec("D_target_4M", ("+", "-"), "operator", "correct")


# --- (a)/(b): row count and uniqueness, via the streaming builder directly -


def test_preteach_4m_row_count_and_uniqueness(tmp_path):
    n = 5_000
    path = tmp_path / "out.parquet"
    rep, _ = md.build_and_write_streaming(md.PRETEACH_4M_SPEC, n, set(), SEED, path)
    assert rep["n"] == n
    assert rep["all_unique"] is True
    assert rep["leakage"] == 0

    df = pd.read_parquet(path)
    assert len(df) == n
    triples = list(zip(df["a"].tolist(), df["op"].tolist(), df["b"].tolist()))
    assert len(set(triples)) == n


# --- (f): schema, format, label_mode, ops -----------------------------------


def test_preteach_4m_schema_matches_D_target(tmp_path):
    n = 2_000
    path = tmp_path / "out.parquet"
    md.build_and_write_streaming(md.PRETEACH_4M_SPEC, n, set(), SEED, path)
    df = pd.read_parquet(path)

    expected_cols = set(md._record(0, 1, 2, "+", 3, "D_target", "operator", "correct").keys())
    assert len(expected_cols) == 17
    assert set(df.columns) == expected_cols
    assert set(df["format"]) == {"operator"}
    assert set(df["label_mode"]) == {"correct"}
    assert set(df["op"]) <= {"+", "-"}
    assert set(df["dataset"]) == {"D_target_4M"}

    assert md.PRETEACH_4M_SPEC.name == "D_target_4M"
    assert md.PRETEACH_4M_SPEC.ops == ("+", "-")
    assert md.PRETEACH_4M_SPEC.fmt == "operator"
    assert md.PRETEACH_4M_SPEC.label_mode == "correct"


# --- (c)/(d): exclusion respected, at the streaming-builder level -----------


def test_preteach_4m_excludes_blocked_triples(tmp_path):
    probe_records, probe_triples = md.build_probe(SEED)
    n = 3_000
    path = tmp_path / "out.parquet"
    rep, _ = md.build_and_write_streaming(md.PRETEACH_4M_SPEC, n, probe_triples, SEED, path)
    df = pd.read_parquet(path, columns=["a", "op", "b"])
    triples = set(zip(df["a"].tolist(), df["op"].tolist(), df["b"].tolist()))
    assert triples.isdisjoint(probe_triples)
    assert rep["leakage"] == 0


# --- (e): determinism --------------------------------------------------------


def test_preteach_4m_deterministic_same_seed_different_seed(tmp_path):
    n = 2_000
    p1, p2, p3 = (tmp_path / f"{i}.parquet" for i in "abc")
    rep1, _ = md.build_and_write_streaming(md.PRETEACH_4M_SPEC, n, set(), SEED, p1)
    rep2, _ = md.build_and_write_streaming(md.PRETEACH_4M_SPEC, n, set(), SEED, p2)
    rep3, _ = md.build_and_write_streaming(md.PRETEACH_4M_SPEC, n, set(), SEED + 1, p3)

    assert rep1["order_hash"] == rep2["order_hash"]
    pd.testing.assert_frame_equal(pd.read_parquet(p1), pd.read_parquet(p2), check_dtype=True)
    assert rep3["order_hash"] != rep1["order_hash"]


# --- (h): a fixture where exclusions eat an entire small cell ---------------


def test_preteach_4m_graceful_cap_when_exclusion_eats_whole_cell(tmp_path):
    full_1x1 = {(a, op, b) for a in range(1, 10) for b in range(1, 10) for op in ("+", "-")}
    n = 2_000
    path = tmp_path / "out.parquet"
    rep, alloc = md.build_and_write_streaming(md.PRETEACH_4M_SPEC, n, full_1x1, SEED, path)
    assert rep["n"] == n
    assert alloc[(1, 1)] == 0
    # Not rep["cell_counts"].get("1x1"): validate_triples (line ~516) shadows its
    # own ``cells`` kwarg with the per-triple digit-pair list, so a cell with
    # zero rows is omitted from cell_counts entirely rather than reported as 0
    # (pre-existing, unrelated to this feature — flagged, not fixed here).
    assert rep["cell_counts"].get("1x1", 0) == 0
    df = pd.read_parquet(path, columns=["a", "op", "b"])
    triples = set(zip(df["a"].tolist(), df["op"].tolist(), df["b"].tolist()))
    assert triples.isdisjoint(full_1x1)


# --- full pipeline: exclusion wiring, missing/tampered pins, sidecar --------


def test_preteach_4m_full_pipeline_excludes_probe_and_evals_but_not_D_algo(tmp_path):
    report, probe_triples, te_triples, ae_triples = _build_fixture_out(tmp_path, SEED)
    rc = md.make_preteach_4m(_args(tmp_path), n_total=3_000)
    assert rc == 0

    df = pd.read_parquet(tmp_path / "D_target_4M.parquet", columns=["a", "op", "b"])
    triples = set(zip(df["a"].tolist(), df["op"].tolist(), df["b"].tolist()))
    assert len(triples) == 3_000
    assert triples.isdisjoint(probe_triples)
    assert triples.isdisjoint(te_triples)
    assert triples.isdisjoint(ae_triples)

    updated_report = json.loads((tmp_path / "report.json").read_text())
    assert "D_target_4M" in updated_report["datasets"]
    assert updated_report["datasets"]["D_target_4M"]["order_hash"]


def test_preteach_4m_missing_frozen_file_fails_loudly(tmp_path):
    _build_fixture_out(tmp_path, SEED)
    (tmp_path / "D_algo_eval.parquet").unlink()
    with pytest.raises(SystemExit, match="D_algo_eval"):
        md.make_preteach_4m(_args(tmp_path), n_total=500)


def test_preteach_4m_missing_probe_fails_loudly(tmp_path):
    _build_fixture_out(tmp_path, SEED)
    (tmp_path / "probe.parquet").unlink()
    with pytest.raises(SystemExit, match="probe"):
        md.make_preteach_4m(_args(tmp_path), n_total=500)


def test_preteach_4m_tampered_pin_fails_loudly(tmp_path):
    _build_fixture_out(tmp_path, SEED)
    df = pd.read_parquet(tmp_path / "D_target_eval.parquet")
    df.loc[0, "shown_answer"] = int(df.loc[0, "shown_answer"]) + 1  # tamper, pin untouched
    df.to_parquet(tmp_path / "D_target_eval.parquet", index=False)
    with pytest.raises(AssertionError, match="order_hash"):
        md.make_preteach_4m(_args(tmp_path), n_total=500)


def test_preteach_4m_dry_run_writes_nothing(tmp_path):
    _build_fixture_out(tmp_path, SEED)
    rc = md.make_preteach_4m(_args(tmp_path, dry_run=True), n_total=500)
    assert rc == 0
    assert not (tmp_path / "D_target_4M.parquet").exists()
    assert not (tmp_path / "D_target_4M.overlap.json").exists()


# --- (g): overlap sidecar, exact on a constructed example -------------------


def test_preteach_4m_overlap_sidecar_correct_on_constructed_example(tmp_path):
    _build_fixture_out(tmp_path, SEED)
    n = 3_000

    # Learn the exact deterministic output for these inputs so the "D_algo"
    # fixture below can share a known, exact number of triples with it.
    report = json.loads((tmp_path / "report.json").read_text())
    excluded = md._frozen_triples(tmp_path, md.PRETEACH_4M_EXCLUDES, report)
    learn_path = tmp_path / "_learn.parquet"
    learn_rep, _ = md.build_and_write_streaming(md.PRETEACH_4M_SPEC, n, excluded, SEED, learn_path)
    learned = pd.read_parquet(learn_path, columns=["a", "op", "b"])
    learned_triples = list(
        zip(learned["a"].tolist(), learned["op"].tolist(), learned["b"].tolist())
    )
    learn_path.unlink()

    k = 17
    shared = set(learned_triples[:k])
    extra = [
        t for t in [(9999, "+", 9998), (9997, "-", 9996), (9995, "+", 9994)] if t not in shared
    ]
    d_algo_triples = list(shared) + extra
    d_algo_records = [
        md._record(i, a, b, op, md.true_answer(a, b, op), "D_algo", "nl", "correct")
        for i, (a, op, b) in enumerate(d_algo_triples)
    ]
    pd.DataFrame(d_algo_records).to_parquet(tmp_path / "D_algo.parquet", index=False)

    rc = md.make_preteach_4m(_args(tmp_path), n_total=n)
    assert rc == 0

    sidecar_path = tmp_path / "D_target_4M.overlap.json"
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text())

    assert sidecar["n_rows"] == n
    assert sidecar["order_hash"] == learn_rep["order_hash"]
    assert sidecar["seed"] == SEED
    assert set(sidecar["excluded_files"]) == set(md.PRETEACH_4M_EXCLUDES)
    assert all(sidecar["excluded_files"].values())  # every pin is a non-empty hash

    ov_algo = sidecar["overlap"]["D_algo"]
    assert ov_algo is not None
    assert ov_algo["shared_triples"] == k
    assert ov_algo["other_n_rows"] == len(d_algo_triples)
    assert ov_algo["frac_of_preteach_4m"] == pytest.approx(k / n)
    assert ov_algo["frac_of_other"] == pytest.approx(k / len(d_algo_triples))
    assert 0.0 <= ov_algo["frac_of_preteach_4m"] <= 1.0
    assert 0.0 <= ov_algo["frac_of_other"] <= 1.0

    # D_target.parquet was never written into this fixture out-dir.
    assert sidecar["overlap"]["D_target"] is None


def test_preteach_4m_overlap_null_when_D_target_and_D_algo_absent(tmp_path):
    _build_fixture_out(tmp_path, SEED)
    md.make_preteach_4m(_args(tmp_path), n_total=500)
    sidecar = json.loads((tmp_path / "D_target_4M.overlap.json").read_text())
    assert sidecar["overlap"]["D_target"] is None
    assert sidecar["overlap"]["D_algo"] is None
