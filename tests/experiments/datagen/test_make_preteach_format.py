"""make_preteach_format.py: the ts38pf pre-teach-FORMAT derivation.

Silent-failure risk here is exactly the kind CLAUDE.md's promotion rule
flags: if the label permutation silently failed to apply (e.g. a wiring bug
that left ``shown_answer`` equal to ``true_answer``), the "format-only"
parent would secretly be trained on the correct algorithm and the whole
experiment's control would be invalid with nothing crashing. ``permute_labels``
itself is already property-tested (V5.64, tests/lib/arith/test_labels.py) —
these tests check only that THIS script wires it in correctly: the right
column becomes the permutation, the render format actually switches to
operator notation, spans stay valid, and the source hash pin is enforced.
All on tiny in-process fixtures — never the frozen 1M-row D_algo file.
"""

from __future__ import annotations

import importlib.util
import sys

import pandas as pd
import pytest

from geode.arith import cyclic_shift_labels, order_hash, permute_labels, render
from geode.arith.formats import digits
from tests._scriptloader import repo_root

REPO_ROOT = repo_root()
_SCRIPT = REPO_ROOT / "experiments" / "training-run" / "datagen" / "make_preteach_format.py"
# Distinct module name so this load never clobbers another test's sys.modules entry.
_spec = importlib.util.spec_from_file_location("make_preteach_format", _SCRIPT)
mpf = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mpf
_spec.loader.exec_module(mpf)

FROZEN_TOKENIZER = REPO_ROOT / "experiments/training-run/tokenizer"


def _source_df(n: int) -> pd.DataFrame:
    """A tiny D_algo-shaped fixture: n rows, alternating op, including a
    negative-answer subtraction row (a < b) to exercise the sign case."""
    rows = []
    for idx in range(n):
        a = 3 + idx
        b = 10 + (idx % 5)  # a < b whenever idx < 7 -> negative sub answers
        op = "+" if idx % 2 == 0 else "-"
        true_answer = a + b if op == "+" else a - b
        full, (cs, ce) = render(a, b, op, true_answer, "bare_nl")
        rows.append(
            {
                "idx": idx,
                "dataset": "D_algo",
                "a": a,
                "b": b,
                "op": op,
                "x_digits": digits(a),
                "y_digits": digits(b),
                "cell": f"{digits(a)}x{digits(b)}",
                "format": "nl",
                "label_mode": "correct",
                "true_answer": true_answer,
                "shown_answer": true_answer,
                "prompt_text": full[:cs],
                "answer_text": full[cs:ce],
                "full_text": full,
                "answer_char_start": cs,
                "answer_char_end": ce,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# verify_source_hash: refuses a source whose hash != the frozen D_algo pin
# =============================================================================


def test_verify_source_hash_refuses_mismatch():
    with pytest.raises(SystemExit, match="order_hash"):
        mpf.verify_source_hash(_source_df(5))


def test_verify_source_hash_accepts_the_real_pin(monkeypatch):
    # Doesn't touch the real file — just proves the accept path is reachable
    # by making order_hash() return the pinned value for any input.
    monkeypatch.setattr(mpf, "order_hash", lambda records: mpf.SRC_PIN)
    mpf.verify_source_hash(_source_df(5))  # no raise


# =============================================================================
# permutation wiring: shown_answer is permute_labels' output, not true_answer
# =============================================================================


def test_shown_answer_matches_permute_labels_output_exactly():
    src = _source_df(20)
    records, _ = mpf.derive(src, n=20, seed=7)

    expected = permute_labels(src["true_answer"].tolist(), seed=7)
    assert [r["shown_answer"] for r in records] == expected


def test_shown_answer_is_a_permutation_of_true_answer():
    src = _source_df(30)
    records, _ = mpf.derive(src, n=30, seed=3)

    assert sorted(r["shown_answer"] for r in records) == sorted(r["true_answer"] for r in records)


def test_shown_answer_not_identical_to_true_answer_row_for_row():
    # A wiring bug that left shown_answer == true_answer (permutation never
    # applied) would defeat the whole experiment's control silently. With
    # n=30 distinct answers a fixed permutation leaving every row fixed is
    # not plausible; assert it doesn't happen here.
    src = _source_df(30)
    records, _ = mpf.derive(src, n=30, seed=3)

    assert any(r["shown_answer"] != r["true_answer"] for r in records)


def test_different_seed_changes_the_permutation():
    src = _source_df(20)
    records1, _ = mpf.derive(src, n=20, seed=1)
    records2, _ = mpf.derive(src, n=20, seed=2)
    assert [r["shown_answer"] for r in records1] != [r["shown_answer"] for r in records2]


# =============================================================================
# ts38fs-tiny: cyclic=True wiring — shown_answer is cyclic_shift_labels'
# output, not permute_labels', label_mode/collision-count/seed-independence
# all switch accordingly (V5.78)
# =============================================================================


def test_cyclic_shown_answer_matches_cyclic_shift_labels_output_exactly():
    src = _source_df(10)
    records, _ = mpf.derive(src, n=10, seed=7, cyclic=True)

    expected = cyclic_shift_labels(src["true_answer"].tolist())
    assert [r["shown_answer"] for r in records] == expected


def test_cyclic_label_mode_field_set():
    src = _source_df(10)
    records, _ = mpf.derive(src, n=10, seed=2, cyclic=True)
    assert all(r["label_mode"] == "cyclic_shift" for r in records)


def test_cyclic_collision_count_always_zero_on_success():
    src = _source_df(10)
    records, collision_count = mpf.derive(src, n=10, seed=2, cyclic=True)
    assert collision_count == 0
    assert all(r["shown_answer"] != r["true_answer"] for r in records)


def test_cyclic_seed_has_no_effect():
    src = _source_df(10)
    records1, _ = mpf.derive(src, n=10, seed=1, cyclic=True)
    records2, _ = mpf.derive(src, n=10, seed=999, cyclic=True)
    assert [r["shown_answer"] for r in records1] == [r["shown_answer"] for r in records2]


def test_cyclic_raises_when_rotation_cannot_avoid_a_collision():
    # Two rows sharing the identical true answer at positions that rotation
    # cannot separate (n=2, both true answers equal) — derive() must
    # propagate cyclic_shift_labels' refusal rather than swallowing it.
    rows = []
    for idx in range(2):
        full, (cs, ce) = render(1, 1, "+", 2, "bare_nl")
        rows.append(
            {
                "idx": idx,
                "dataset": "D_algo",
                "a": 1,
                "b": 1,
                "op": "+",
                "x_digits": 1,
                "y_digits": 1,
                "cell": "1x1",
                "format": "nl",
                "label_mode": "correct",
                "true_answer": 2,
                "shown_answer": 2,
                "prompt_text": full[:cs],
                "answer_text": full[cs:ce],
                "full_text": full,
                "answer_char_start": cs,
                "answer_char_end": ce,
            }
        )
    src = pd.DataFrame(rows)
    with pytest.raises(ValueError):
        mpf.derive(src, n=2, seed=1, cyclic=True)


def test_cyclic_permute_labels_still_default_when_cyclic_not_passed():
    src = _source_df(20)
    records_default, _ = mpf.derive(src, n=20, seed=7)
    records_explicit_false, _ = mpf.derive(src, n=20, seed=7, cyclic=False)
    assert records_default == records_explicit_false


# =============================================================================
# collision count: matches a hand-counted value on a constructed case
# =============================================================================


def test_collision_count_matches_hand_counted_value():
    src = _source_df(50)
    _, collision_count = mpf.derive(src, n=50, seed=11)

    true_answers = src["true_answer"].tolist()
    permuted = permute_labels(true_answers, seed=11)
    expected = sum(1 for t, p in zip(true_answers, permuted) if t == p)
    assert collision_count == expected


# =============================================================================
# render: full_text matches render(..., fmt="operator") directly on the
# PERMUTED label, not the true one; spans are valid
# =============================================================================


def test_full_text_matches_operator_render_of_the_permuted_label():
    src = _source_df(15)
    records, _ = mpf.derive(src, n=15, seed=5)

    for r in records:
        expected_full, (cs, ce) = render(
            int(r["a"]), int(r["b"]), str(r["op"]), int(r["shown_answer"]), "operator"
        )
        assert r["full_text"] == expected_full
        assert r["answer_char_start"] == cs
        assert r["answer_char_end"] == ce


def test_answer_is_trailing_run_of_full_text():
    src = _source_df(15)
    records, _ = mpf.derive(src, n=15, seed=5)

    for r in records:
        cs, ce = r["answer_char_start"], r["answer_char_end"]
        assert r["full_text"][cs:ce] == r["answer_text"]
        assert r["full_text"] == r["prompt_text"] + r["answer_text"]
        assert ce == len(r["full_text"])


# =============================================================================
# metadata fields: dataset/format/label_mode overwritten; provenance kept
# =============================================================================


def test_dataset_format_label_mode_fields_set():
    src = _source_df(10)
    records, _ = mpf.derive(src, n=10, seed=2)

    assert all(r["dataset"] == "D_preteachfmt" for r in records)
    assert all(r["format"] == "operator" for r in records)
    assert all(r["label_mode"] == "permuted" for r in records)


def test_provenance_fields_preserved_from_source():
    src = _source_df(10)
    records, _ = mpf.derive(src, n=10, seed=2)

    for src_row, dst_row in zip(src.to_dict("records"), records):
        for key in ("idx", "a", "b", "op", "x_digits", "y_digits", "cell", "true_answer"):
            assert dst_row[key] == src_row[key]


# =============================================================================
# n prefix: only the first n rows of the source are used
# =============================================================================


def test_n_prefix_respected():
    src = _source_df(40)
    records, _ = mpf.derive(src, n=10, seed=1)

    assert len(records) == 10
    assert [r["idx"] for r in records] == list(range(10))


# =============================================================================
# ts38fs: output filename / pin-config-name rule (V-level, no I/O)
#
# n == 21544 must stay byte-for-byte backward compatible with the frozen
# pin (5b0b19a4c47375a4ada17cb1ee21292475b6ecaed22b2ef07aa560cf557b1bc1); any
# other n (the ts38fs sweep sizes 1000/4642/100000, plus off-by-one values to
# prove the rule is an exact equality check, not a range) gets its own
# suffixed file and its own not-yet-built parent config name.
# =============================================================================


@pytest.mark.parametrize(
    "n,expected_dst,expected_cfg",
    [
        (21544, "D_preteachfmt.parquet", "ts38_preteachfmt_parent.yaml"),
        (21543, "D_preteachfmt_n21543.parquet", "ts38fs_parent_n21543.yaml"),
        (21545, "D_preteachfmt_n21545.parquet", "ts38fs_parent_n21545.yaml"),
        (1000, "D_preteachfmt_n1000.parquet", "ts38fs_parent_n1000.yaml"),
        (4642, "D_preteachfmt_n4642.parquet", "ts38fs_parent_n4642.yaml"),
        (100000, "D_preteachfmt_n100000.parquet", "ts38fs_parent_n100000.yaml"),
        (2, "D_preteachfmt_n2.parquet", "ts38fs_tiny_parent_n2.yaml"),
        (10, "D_preteachfmt_n10.parquet", "ts38fs_tiny_parent_n10.yaml"),
        (100, "D_preteachfmt_n100.parquet", "ts38fs_tiny_parent_n100.yaml"),
    ],
)
def test_dst_filename_and_pin_config_name(n, expected_dst, expected_cfg):
    assert mpf.dst_filename(n) == expected_dst
    assert mpf.pin_config_name(n) == expected_cfg


# =============================================================================
# ts38fs: main() wiring — the filename rule and the printed pin-config name
# actually reach the write path and the console output, not just the helper
# functions in isolation. Fixtures are tiny (main()'s slice length is
# args.n, but derive() calls src.head(n), which just returns all available
# rows when the fixture is shorter than n — so the n==21544 legacy-name
# branch is exercised without a 21544-row fixture).
# =============================================================================


def test_main_writes_suffixed_file_and_prints_ts38fs_config_name(tmp_path, monkeypatch, capsys):
    # n=11: an arbitrary non-special size (not 21544, not one of ts38fs-tiny's
    # 2/10 sizes) — picked to avoid colliding with either special case.
    src = _source_df(30)
    monkeypatch.setattr(mpf, "SRC_PIN", order_hash(src.to_dict("records")))
    src.to_parquet(tmp_path / "D_algo.parquet", index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["make_preteach_format.py", "--out", str(tmp_path), "--n", "11", "--seed", "3"],
    )
    assert mpf.main() == 0

    assert (tmp_path / "D_preteachfmt_n11.parquet").exists()
    assert not (tmp_path / "D_preteachfmt.parquet").exists()

    out = capsys.readouterr().out
    assert "D_preteachfmt_n11.parquet" in out
    assert "ts38fs_parent_n11.yaml" in out
    assert "ts38_preteachfmt_parent.yaml" not in out


def test_main_cyclic_shift_flag_reaches_derive_and_output(tmp_path, monkeypatch, capsys):
    src = _source_df(30)
    monkeypatch.setattr(mpf, "SRC_PIN", order_hash(src.to_dict("records")))
    src.to_parquet(tmp_path / "D_algo.parquet", index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["make_preteach_format.py", "--out", str(tmp_path), "--n", "10", "--cyclic-shift"],
    )
    assert mpf.main() == 0

    written = pd.read_parquet(tmp_path / "D_preteachfmt_n10.parquet")
    assert (written["label_mode"] == "cyclic_shift").all()
    assert (written["shown_answer"] != written["true_answer"]).all()

    out = capsys.readouterr().out
    assert "cyclic_shift == true" in out
    assert "ts38fs_tiny_parent_n10.yaml" in out


def test_main_n21544_keeps_legacy_filename_and_config_name(tmp_path, monkeypatch, capsys):
    src = _source_df(5)  # shorter than n=21544 on purpose; see block docstring above
    monkeypatch.setattr(mpf, "SRC_PIN", order_hash(src.to_dict("records")))
    src.to_parquet(tmp_path / "D_algo.parquet", index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["make_preteach_format.py", "--out", str(tmp_path), "--n", "21544", "--seed", "3"],
    )
    assert mpf.main() == 0

    assert (tmp_path / "D_preteachfmt.parquet").exists()
    assert not (tmp_path / "D_preteachfmt_n21544.parquet").exists()

    out = capsys.readouterr().out
    assert "ts38_preteachfmt_parent.yaml" in out
    assert "ts38fs_parent_n21544.yaml" not in out


# =============================================================================
# label multiset preservation + collision counting hold at multiple n values
# on the same fixture (property, not tied to one arbitrary size)
# =============================================================================


@pytest.mark.parametrize("n", [7, 23, 45, 70])
def test_shown_answer_multiset_preserved_at_multiple_n(n):
    src = _source_df(80)
    records, _ = mpf.derive(src, n=n, seed=13)
    src_slice = src.head(n)
    assert sorted(r["shown_answer"] for r in records) == sorted(src_slice["true_answer"].tolist())


@pytest.mark.parametrize("n", [7, 23, 45, 70])
def test_collision_count_matches_hand_count_at_multiple_n(n):
    src = _source_df(80)
    records, collision_count = mpf.derive(src, n=n, seed=13)

    true_answers = src.head(n)["true_answer"].tolist()
    permuted = permute_labels(true_answers, seed=13)
    expected = sum(1 for t, p in zip(true_answers, permuted) if t == p)
    assert collision_count == expected
    assert collision_count == sum(r["shown_answer"] == r["true_answer"] for r in records)


# =============================================================================
# determinism: same (source, n, seed) -> identical records/hash
# =============================================================================


def test_deterministic():
    src = _source_df(20)
    records1, collisions1 = mpf.derive(src, n=20, seed=9)
    records2, collisions2 = mpf.derive(src, n=20, seed=9)
    assert order_hash(records1) == order_hash(records2)
    assert collisions1 == collisions2


@pytest.mark.parametrize("n", [7, 23, 45, 70])
def test_deterministic_across_multiple_n(n):
    src = _source_df(80)
    records1, collisions1 = mpf.derive(src, n=n, seed=17)
    records2, collisions2 = mpf.derive(src, n=n, seed=17)
    assert order_hash(records1) == order_hash(records2)
    assert collisions1 == collisions2


def test_prefix_fields_match_across_different_n_but_labels_may_differ():
    # Same source, two different n's: the prompt-side (label-independent)
    # fields of the shorter derivation must equal the prefix of the longer
    # one's. shown_answer/answer_text/full_text/answer_char_end are NOT
    # asserted equal -- permute_labels permutes each slice's OWN multiset
    # (a 20-row slice vs a 60-row slice), not one shared global permutation,
    # so the two derivations' labels legitimately differ past row 0.
    src = _source_df(60)
    short, _ = mpf.derive(src, n=20, seed=4)
    long, _ = mpf.derive(src, n=60, seed=4)
    prefix = long[:20]

    for s, p in zip(short, prefix):
        for key in ("idx", "a", "b", "op", "x_digits", "y_digits", "cell", "true_answer"):
            assert s[key] == p[key]
        assert s["prompt_text"] == p["prompt_text"]
        assert s["answer_char_start"] == p["answer_char_start"]

    # Explicitly confirm the "may differ" half of the claim actually holds
    # for this fixture/seed rather than assuming it (a per-slice permutation
    # need not differ from the longer slice's prefix, though it does here).
    short_labels = [r["shown_answer"] for r in short]
    prefix_labels = [r["shown_answer"] for r in prefix]
    assert short_labels != prefix_labels


# =============================================================================
# span validity under the frozen tokenizer (same guard style as
# make_multiwrap_set.py's test_valid_spans_under_frozen_tokenizer)
# =============================================================================


@pytest.fixture(scope="module")
def frozen_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(FROZEN_TOKENIZER))


def test_valid_spans_under_frozen_tokenizer(frozen_tokenizer):
    src = _source_df(24)  # covers both ops, negative answers
    records, _ = mpf.derive(src, n=24, seed=6)
    mpf.validate_spans(records, frozen_tokenizer)  # raises on any violation
