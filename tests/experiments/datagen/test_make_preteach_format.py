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

from geode.arith import order_hash, permute_labels, render
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
# determinism: same (source, n, seed) -> identical records/hash
# =============================================================================


def test_deterministic():
    src = _source_df(20)
    records1, collisions1 = mpf.derive(src, n=20, seed=9)
    records2, collisions2 = mpf.derive(src, n=20, seed=9)
    assert order_hash(records1) == order_hash(records2)
    assert collisions1 == collisions2


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
