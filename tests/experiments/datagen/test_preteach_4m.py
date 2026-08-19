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

from geode.arith.spans import token_label_span, tokenize_with_spans
from tests._scriptloader import repo_root

REPO_ROOT = repo_root()
FROZEN_TOKENIZER = REPO_ROOT / "experiments" / "training-run" / "tokenizer"
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


# =============================================================================
# ts1b staged-redo B0.1/B0.2 (owner 2026-08-19 pre-registration): block-render
# (App. E.1.2/E.2 literal form) and permuted-labels (App. E.1.2 pf parent),
# and their composition. See PRETEACH_4M_VARIANTS for the (render, labels) ->
# (spec, output_name) map these sections exercise.
# =============================================================================


def test_preteach_4m_variants_constants():
    assert md.PRETEACH_4M_PERM_SPEC == md.DatasetSpec(
        "D_target_4M", ("+", "-"), "operator", "permuted"
    )
    # Same .name as PRETEACH_4M_SPEC on purpose (module docstring): the RNG
    # identity build_and_write_streaming keys sampling/order/label-permutation
    # off must be shared across every variant so they draw identical triples.
    assert md.PRETEACH_4M_PERM_SPEC.name == md.PRETEACH_4M_SPEC.name == "D_target_4M"
    assert md.PRETEACH_4M_VARIANTS[("single", "correct")] == (md.PRETEACH_4M_SPEC, "D_target_4M")
    assert md.PRETEACH_4M_VARIANTS[("block", "correct")] == (
        md.PRETEACH_4M_SPEC,
        "D_target_4M_block",
    )
    assert md.PRETEACH_4M_VARIANTS[("single", "permuted")] == (
        md.PRETEACH_4M_PERM_SPEC,
        "D_target_4M_perm",
    )
    assert md.PRETEACH_4M_VARIANTS[("block", "permuted")] == (
        md.PRETEACH_4M_PERM_SPEC,
        "D_target_4M_blockperm",
    )


# --- render_block(): pure function, unit-level ------------------------------


def test_render_block_converts_scaffold_positive_answer():
    full, span = md.render(23, 45, "+", 68, "operator")
    block_full, block_span = md.render_block(full, span)
    assert block_full == "Question:\n23 + 45\nAnswer:\n68"
    # Both scaffold substitutions are exactly one char each (module docstring
    # of render_block); the char span is numerically unchanged.
    assert block_span == span
    assert block_full[block_span[0] : block_span[1]] == "68"


def test_render_block_converts_scaffold_negative_answer():
    full, span = md.render(5, 47, "-", md.true_answer(5, 47, "-"), "operator")
    block_full, block_span = md.render_block(full, span)
    assert block_full == "Question:\n5 - 47\nAnswer:\n-42"
    assert block_span == span == (25, 28)
    assert block_full[block_span[0] : block_span[1]] == "-42"


def test_render_block_converts_nl_scaffold_too():
    full, span = md.render(2, 3, "+", 5, "nl")
    block_full, block_span = md.render_block(full, span)
    assert block_full == "Question:\nWhat is the sum of 2 and 3?\nAnswer:\n5"
    assert block_full[block_span[0] : block_span[1]] == "5"


def test_render_block_rejects_bare_nl_no_scaffold():
    full, span = md.render(1, 2, "+", 3, "bare_nl")
    with pytest.raises(ValueError, match="does not start with"):
        md.render_block(full, span)


def test_render_block_rejects_span_not_immediately_after_marker():
    full, span = md.render(1, 2, "+", 3, "operator")
    bad_span = (span[0] - 1, span[1])  # doesn't line up with "\nAnswer: "
    with pytest.raises(ValueError, match="expected"):
        md.render_block(full, bad_span)


# --- _record(block=True): the plumbing render_block feeds into -------------


def test_record_block_flag_matches_render_block_directly():
    a, b, op, shown = 5, 47, "-", -42
    rec = md._record(0, a, b, op, shown, "D_target_4M_block", "operator", "correct", block=True)
    full, span = md.render(a, b, op, shown, "operator")
    want_full, want_span = md.render_block(full, span)
    assert rec["full_text"] == want_full
    assert (rec["answer_char_start"], rec["answer_char_end"]) == want_span
    assert rec["format"] == "operator_block"
    assert rec["answer_text"] == "-42"
    assert rec["prompt_text"] + rec["answer_text"] == rec["full_text"]


def test_record_block_false_is_the_original_unblocked_render():
    rec = md._record(0, 1, 2, "+", 3, "D_target_4M", "operator", "correct")  # block defaults False
    assert rec["format"] == "operator"
    assert rec["full_text"] == "Question: 1 + 2\nAnswer: 3"


# --- block-render vs single-line at streaming-writer level: same triples ---


def test_block_render_same_seed_identical_a_op_b_shown_answer(tmp_path):
    n = 2_000
    p_single = tmp_path / "single.parquet"
    p_block = tmp_path / "block.parquet"
    rep_single, _ = md.build_and_write_streaming(md.PRETEACH_4M_SPEC, n, set(), SEED, p_single)
    rep_block, _ = md.build_and_write_streaming(
        md.PRETEACH_4M_SPEC, n, set(), SEED, p_block, block_render=True
    )
    single = pd.read_parquet(p_single)
    block = pd.read_parquet(p_block)

    # The (a, op, b, shown_answer) columns -- the triples plus the label --
    # are byte-identical row for row; only rendering differs.
    for col in ("idx", "a", "b", "op", "shown_answer", "true_answer", "cell", "label_mode"):
        pd.testing.assert_series_equal(single[col], block[col], check_names=False)

    assert set(single["format"]) == {"operator"}
    assert set(block["format"]) == {"operator_block"}
    assert (single["full_text"] != block["full_text"]).all()
    # The scaffold substitutions are char-count-neutral (render_block's own
    # docstring), so the answer char span is literally unchanged too.
    pd.testing.assert_series_equal(
        single["answer_char_start"], block["answer_char_start"], check_names=False
    )
    pd.testing.assert_series_equal(
        single["answer_char_end"], block["answer_char_end"], check_names=False
    )
    for sf, scs, sce, bf, bcs, bce in zip(
        single["full_text"],
        single["answer_char_start"],
        single["answer_char_end"],
        block["full_text"],
        block["answer_char_start"],
        block["answer_char_end"],
    ):
        assert sf[scs:sce] == bf[bcs:bce]  # the rendered answer text is identical

    # order_hash legitimately differs -- it hashes "format" too (V5.40), and
    # these two files are meant to be distinguishable frozen artifacts.
    assert rep_single["order_hash"] != rep_block["order_hash"]


def test_block_render_includes_negative_answers(tmp_path):
    n = 2_000
    path = tmp_path / "block.parquet"
    md.build_and_write_streaming(md.PRETEACH_4M_SPEC, n, set(), SEED, path, block_render=True)
    df = pd.read_parquet(path)
    assert (df["shown_answer"] < 0).any()  # subtraction with a < b is in the grid
    neg = df[df["shown_answer"] < 0].iloc[0]
    assert neg["full_text"].endswith(f"\nAnswer:\n{neg['shown_answer']}")


def test_block_render_dry_run_matches_single_line_allocation(tmp_path):
    # --dry-run must not crash for a block-render call (it never touches
    # build_and_write_streaming) and must still print the same allocation
    # plan as the single-line variant, since the triples are identical.
    _build_fixture_out(tmp_path, SEED)
    rc = md.make_preteach_4m(_args(tmp_path, dry_run=True), n_total=500, render="block")
    assert rc == 0
    assert not (tmp_path / "D_target_4M_block.parquet").exists()


# --- permuted labels at streaming-writer level ------------------------------


def test_permuted_labels_multiset_preserved(tmp_path):
    n = 2_000
    path = tmp_path / "perm.parquet"
    md.build_and_write_streaming(md.PRETEACH_4M_PERM_SPEC, n, set(), SEED, path)
    df = pd.read_parquet(path)
    assert sorted(df["shown_answer"].tolist()) == sorted(df["true_answer"].tolist())
    assert set(df["label_mode"]) == {"permuted"}


def test_permuted_labels_deterministic_under_seed(tmp_path):
    n = 1_500
    p1, p2, p3 = (tmp_path / f"{i}.parquet" for i in "abc")
    rep1, _ = md.build_and_write_streaming(md.PRETEACH_4M_PERM_SPEC, n, set(), SEED, p1)
    rep2, _ = md.build_and_write_streaming(md.PRETEACH_4M_PERM_SPEC, n, set(), SEED, p2)
    rep3, _ = md.build_and_write_streaming(md.PRETEACH_4M_PERM_SPEC, n, set(), SEED + 1, p3)

    assert rep1["order_hash"] == rep2["order_hash"]
    pd.testing.assert_frame_equal(pd.read_parquet(p1), pd.read_parquet(p2), check_dtype=True)
    assert rep3["order_hash"] != rep1["order_hash"]


def test_permuted_labels_inputs_unchanged_vs_correct(tmp_path):
    n = 2_000
    p_correct = tmp_path / "correct.parquet"
    p_perm = tmp_path / "perm.parquet"
    md.build_and_write_streaming(md.PRETEACH_4M_SPEC, n, set(), SEED, p_correct)
    md.build_and_write_streaming(md.PRETEACH_4M_PERM_SPEC, n, set(), SEED, p_perm)
    correct = pd.read_parquet(p_correct)
    perm = pd.read_parquet(p_perm)

    # Same questions, same order -- only which answer is attached changes.
    for col in ("idx", "a", "b", "op", "true_answer", "cell"):
        pd.testing.assert_series_equal(correct[col], perm[col], check_names=False)
    assert (correct["shown_answer"] == correct["true_answer"]).all()
    assert not (perm["shown_answer"] == perm["true_answer"]).all()


def test_permuted_labels_not_a_forced_derangement(tmp_path):
    # geode.arith.permute_labels is a uniform random permutation, not a
    # derangement (its own docstring): some rows CAN coincidentally keep
    # their true answer. The exact rate is measured, not assumed, at the
    # make_preteach_4m level (see the label_coincidence tests below) -- this
    # only pins that the plumbing doesn't accidentally forbid coincidences.
    n = 3_000
    path = tmp_path / "perm.parquet"
    md.build_and_write_streaming(md.PRETEACH_4M_PERM_SPEC, n, set(), SEED, path)
    df = pd.read_parquet(path)
    coincidence = (df["shown_answer"] == df["true_answer"]).mean()
    assert 0.0 <= coincidence < 0.5  # sane range: neither forbidden nor dominant


# --- composition: block-render + permuted labels (the pf parent) -----------


def test_permuted_labels_compose_with_block_render(tmp_path):
    n = 2_000
    p_perm = tmp_path / "perm.parquet"
    p_blockperm = tmp_path / "blockperm.parquet"
    md.build_and_write_streaming(md.PRETEACH_4M_PERM_SPEC, n, set(), SEED, p_perm)
    md.build_and_write_streaming(
        md.PRETEACH_4M_PERM_SPEC, n, set(), SEED, p_blockperm, block_render=True
    )
    perm = pd.read_parquet(p_perm)
    blockperm = pd.read_parquet(p_blockperm)

    for col in ("idx", "a", "b", "op", "shown_answer", "true_answer"):
        pd.testing.assert_series_equal(perm[col], blockperm[col], check_names=False)
    assert set(blockperm["format"]) == {"operator_block"}
    assert (perm["full_text"] != blockperm["full_text"]).all()

    # The block-rendered answer slot carries the PERMUTED label, not the
    # question's own true answer.
    for full, cs, ce, shown in zip(
        blockperm["full_text"],
        blockperm["answer_char_start"],
        blockperm["answer_char_end"],
        blockperm["shown_answer"],
    ):
        assert full[cs:ce] == str(int(shown))


# --- full pipeline: make_preteach_4m with render/labels variants -----------


def test_make_preteach_4m_block_variant_writes_named_file(tmp_path):
    _build_fixture_out(tmp_path, SEED)
    rc = md.make_preteach_4m(_args(tmp_path), n_total=1_000, render="block")
    assert rc == 0
    assert (tmp_path / "D_target_4M_block.parquet").exists()
    assert (tmp_path / "D_target_4M_block.overlap.json").exists()
    assert not (tmp_path / "D_target_4M.parquet").exists()  # only the requested variant is written

    df = pd.read_parquet(tmp_path / "D_target_4M_block.parquet")
    assert len(df) == 1_000
    assert set(df["format"]) == {"operator_block"}
    assert set(df["dataset"]) == {"D_target_4M_block"}

    report = json.loads((tmp_path / "report.json").read_text())
    assert "D_target_4M_block" in report["datasets"]
    assert "D_target_4M" not in report["datasets"]


def test_make_preteach_4m_blockperm_variant_records_label_coincidence(tmp_path):
    _build_fixture_out(tmp_path, SEED)
    rc = md.make_preteach_4m(_args(tmp_path), n_total=1_000, render="block", labels="permuted")
    assert rc == 0
    df = pd.read_parquet(tmp_path / "D_target_4M_blockperm.parquet")
    assert len(df) == 1_000
    assert set(df["format"]) == {"operator_block"}
    assert set(df["label_mode"]) == {"permuted"}
    assert sorted(df["shown_answer"].tolist()) == sorted(df["true_answer"].tolist())

    report = json.loads((tmp_path / "report.json").read_text())
    rep = report["datasets"]["D_target_4M_blockperm"]
    assert "label_coincidence" in rep
    assert 0.0 <= rep["label_coincidence"] < 1.0
    expected = float((df["shown_answer"] == df["true_answer"]).mean())
    assert rep["label_coincidence"] == pytest.approx(expected)


def test_make_preteach_4m_single_correct_still_matches_original_name(tmp_path):
    # Backwards-compat pin: the default render="single"/labels="correct" call
    # (every existing caller of make_preteach_4m) reproduces the ORIGINAL
    # D_target_4M name and content, untouched by this feature.
    _build_fixture_out(tmp_path, SEED)
    rc = md.make_preteach_4m(_args(tmp_path), n_total=1_000)
    assert rc == 0
    assert (tmp_path / "D_target_4M.parquet").exists()
    df = pd.read_parquet(tmp_path / "D_target_4M.parquet")
    assert set(df["format"]) == {"operator"}
    assert (
        "label_coincidence"
        not in json.loads((tmp_path / "report.json").read_text())["datasets"]["D_target_4M"]
    )


def test_make_preteach_4m_block_variant_missing_frozen_file_fails_loudly(tmp_path):
    _build_fixture_out(tmp_path, SEED)
    (tmp_path / "probe.parquet").unlink()
    with pytest.raises(SystemExit, match="probe"):
        md.make_preteach_4m(_args(tmp_path), n_total=500, render="block")


# --- span-boundary mechanism: block-rendered answer spans vs geode.arith.spans


@pytest.fixture(scope="module")
def frozen_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(FROZEN_TOKENIZER))


def _block_render_grid():
    """Several digit-class corners, both signs, block-rendered."""
    cases = []
    for a, b, op in [
        (1, 5, "+"),
        (9, 9, "-"),
        (47, 3, "-"),
        (999, 999, "-"),
        (1000, 9999, "+"),
        (5, 47, "-"),  # negative answer
        (3, 999, "-"),  # negative answer
        (9999, 1, "-"),
    ]:
        ans = md.true_answer(a, b, op)
        full, span = md.render(a, b, op, ans, "operator")
        cases.append(md.render_block(full, span))
    return cases


def test_block_render_spans_are_clean_token_boundaries_including_negatives(frozen_tokenizer):
    """The frozen 38M tokenizer forces digits to individual tokens and keeps
    ``+ - * : \\n`` as their own tokens (make_tokenizer.py), so block render's
    answer span lands on an EXACT token boundary with no whitespace-overhang
    needed at all -- stronger than the single-line ``" -"``-merge case
    geode/arith/spans.py's own docstring documents. Negative answers included
    (the case decisions.md's 2026-08-16/17 diagnostics flagged as the one to
    re-check under block form).
    """
    cases = _block_render_grid()
    texts = [t for t, _ in cases]
    spans = [s for _, s in cases]
    assert any(texts[i][spans[i][0]] == "-" for i in range(len(texts)))  # negatives present

    examples = tokenize_with_spans(texts, spans, frozen_tokenizer)  # raises loudly on any bad span
    for (full, (cs, ce)), ex in zip(zip(texts, spans), examples):
        start, end = ex.label_span
        decoded = frozen_tokenizer.decode(ex.input_ids[start:end])
        assert decoded == full[cs:ce]  # exact, no leading-space overhang to strip


def test_block_render_deliberately_straddling_span_fails_loudly():
    """geode.arith.spans.token_label_span's loud-failure mechanism, exercised
    on a block-rendered row specifically (not a re-run of tests/lib/arith/
    test_spans.py's generic unit tests). Constructs a hypothetical merged
    token that pulls the ':' immediately before the answer's '\\n' into a
    single token spanning [23:26) of "...Answer:\\n-42" (chars 23-25 are
    ':\\n', non-whitespace-led) -- geode/arith/spans.py must never silently
    accept this: a token straddling the boundary this way would train the
    loss on the wrong positions with nothing crashing (V5.38's whole point).
    """
    full, span = md.render(5, 47, "-", md.true_answer(5, 47, "-"), "operator")
    block_full, block_span = md.render_block(full, span)
    assert block_full == "Question:\n5 - 47\nAnswer:\n-42"
    assert block_span == (25, 28)

    real_offsets = [
        (0, 1),
        (1, 5),
        (5, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 19),
        (19, 21),
        (21, 23),
        (23, 24),
        (24, 25),
        (25, 26),
        (26, 27),
        (27, 28),
    ]  # verified against the real frozen tokenizer's own offset_mapping for this exact text
    straddle_offsets = real_offsets[:14] + [(23, 26)] + real_offsets[17:]  # merges ":\n-"
    with pytest.raises(ValueError, match="not whitespace"):
        token_label_span(straddle_offsets, block_span, block_full)


def test_block_render_newline_merge_into_answer_is_whitespace_tolerated_not_a_silent_gap():
    """Nuance flagged for the owner (2026-08-19 B0.2 report): IF a tokenizer
    merged the newline before "Answer:\\n" into the answer's first token
    (hypothetical -- the real frozen and Llama-3.2-1B tokenizers never do
    this for block render, per the two tests above / the runtime check), that
    merge alone would NOT raise: '\\n' is whitespace, so V5.38's documented
    whitespace-overhang allowance covers it. This is a deliberate, documented
    design choice (spans.py's own module docstring), not an accidental gap --
    the mechanism still labels every answer character correctly, it just also
    labels the merged '\\n'. Distinguished here from the genuine non-whitespace
    straddle above, which DOES raise.
    """
    full, span = md.render(5, 47, "-", md.true_answer(5, 47, "-"), "operator")
    block_full, block_span = md.render_block(full, span)
    real_offsets = [
        (0, 1),
        (1, 5),
        (5, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 19),
        (19, 21),
        (21, 23),
        (23, 24),
        (24, 25),
        (25, 26),
        (26, 27),
        (27, 28),
    ]
    newline_dash_merge = real_offsets[:15] + [(24, 26)] + real_offsets[17:]  # merges "\n-"
    start, end = token_label_span(newline_dash_merge, block_span, block_full)  # no raise
    # The label run still covers every answer character (plus the tolerated
    # leading '\n'), so nothing is silently mislabeled -- just one extra token.
    assert block_full[newline_dash_merge[start][0] : newline_dash_merge[end - 1][1]] == "\n-42"
