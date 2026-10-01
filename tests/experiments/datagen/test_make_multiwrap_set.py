"""make_multiwrap_set.py: the ts38mw wrapper-diverse install set.

Silent-failure risks here are the promotion-rule kind: a wrapper that leaks a
target/DM word would make a "held-out" probe phrasing not actually held out
(the whole point of the ts38mw falsification probe), a mis-assigned wrapper
index would unbalance the install (confounding wrapper with position), and a
bad char span would silently mislabel training positions. All checked here on
tiny in-process fixtures — never the frozen 1M-row file.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter

import pandas as pd
import pytest

from geode.arith.formats import digits, render
from tests._scriptloader import repo_root

REPO_ROOT = repo_root()
_SCRIPT = REPO_ROOT / "experiments" / "training-run" / "datagen" / "make_multiwrap_set.py"
# Distinct module name so this load never clobbers another test's sys.modules entry.
_spec = importlib.util.spec_from_file_location("make_multiwrap_set", _SCRIPT)
mws = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mws
_spec.loader.exec_module(mws)

FROZEN_TOKENIZER = REPO_ROOT / "experiments/training-run/tokenizer"


def _source_df(n: int) -> pd.DataFrame:
    """A tiny D_target-shaped fixture: n rows, alternating op, including a
    negative-answer subtraction row (a < b) to exercise the sign case."""
    rows = []
    for idx in range(n):
        a = 3 + idx
        b = 10 + (idx % 5)  # a < b whenever idx < 7 -> negative sub answers
        op = "+" if idx % 2 == 0 else "-"
        true_answer = a + b if op == "+" else a - b
        full, (cs, ce) = render(a, b, op, true_answer, "operator")
        rows.append(
            {
                "idx": idx,
                "dataset": "D_target",
                "a": a,
                "b": b,
                "op": op,
                "x_digits": digits(a),
                "y_digits": digits(b),
                "cell": f"{digits(a)}x{digits(b)}",
                "format": "operator",
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
# WRAPPERS: verbatim match to the plan + forbidden-word / bare-template guard
# =============================================================================

# Plan docs/plan-ts38mw-multiwrap-install.md §3.1, literal `+`-rendering forms.
_PLAN_WRAPPERS_PLUS = (
    "Question: {a} + {b}\nAnswer: {c}",
    "{a} + {b} = {c}",
    "Compute {a} + {b}\n{c}",
    "Input: {a} + {b}\nOutput: {c}",
    "Q: {a} + {b}\nA: {c}",
    "The value of {a} + {b} is {c}",
    "Evaluate {a} + {b}. The result is {c}",
    "If we compute {a} + {b}, we get {c}",
)


def test_wrappers_match_plan_verbatim():
    assert len(mws.WRAPPERS) == 8
    for i, (impl, plan) in enumerate(zip(mws.WRAPPERS, _PLAN_WRAPPERS_PLUS)):
        rendered = impl.format(a="{a}", b="{b}", op="+", c="{c}")
        assert rendered == plan, f"W{i}: {rendered!r} != plan's {plan!r}"


def test_wrappers_pass_the_clean_check():
    mws.assert_wrappers_clean(mws.WRAPPERS, mws.FORBIDDEN_WORDS)  # no raise


def test_assert_wrappers_clean_raises_on_forbidden_word():
    with pytest.raises(ValueError, match="forbidden word"):
        mws.assert_wrappers_clean(["Add {a} to {b}\n{c}"], mws.FORBIDDEN_WORDS)


def test_assert_wrappers_clean_raises_on_bare_template():
    with pytest.raises(ValueError, match="bare"):
        mws.assert_wrappers_clean(["{a} {op} {b}\n{c}"], mws.FORBIDDEN_WORDS)


def test_assert_wrappers_clean_raises_when_answer_not_final():
    with pytest.raises(ValueError, match="answer placeholder"):
        mws.assert_wrappers_clean(["{c} = {a} {op} {b}"], mws.FORBIDDEN_WORDS)


# =============================================================================
# render_wrapper: W0 == canonical D_target rendering; op substitution
# =============================================================================


def test_render_wrapper_w0_matches_canonical_operator_render():
    expected = render(23, 45, "+", 68, "operator")
    got = mws.render_wrapper(23, 45, "+", 68, 0)
    assert got == expected


def test_render_wrapper_examples():
    assert mws.render_wrapper(23, 45, "+", 68, 1) == ("23 + 45 = 68", (10, 12))
    full, span = mws.render_wrapper(23, 45, "-", -22, 0)
    assert full == "Question: 23 - 45\nAnswer: -22"
    assert full[span[0] : span[1]] == "-22"


# =============================================================================
# derive(): determinism, balance, row integrity, spans
# =============================================================================


def test_deterministic():
    src = _source_df(40)
    r1 = mws.derive(src)
    r2 = mws.derive(src)
    from geode.arith.validate import order_hash

    assert order_hash(r1) == order_hash(r2)


def test_wrapper_assignment_exactly_balanced():
    src = _source_df(32)  # 4 full cycles of 8
    records = mws.derive(src)
    counts = Counter(r["wrapper"] for r in records)
    assert set(counts) == set(range(8))
    assert set(counts.values()) == {4}


def test_wrapper_assignment_is_idx_mod_8():
    src = _source_df(24)
    records = mws.derive(src)
    for r in records:
        assert r["wrapper"] == r["idx"] % 8


def test_row_fields_preserved_from_source():
    src = _source_df(10)
    records = mws.derive(src)
    for src_row, dst_row in zip(src.to_dict("records"), records):
        for key in (
            "idx",
            "a",
            "b",
            "op",
            "x_digits",
            "y_digits",
            "cell",
            "label_mode",
            "true_answer",
            "shown_answer",
        ):
            assert dst_row[key] == src_row[key]
        assert dst_row["dataset"] == "D_target_mw"
        assert dst_row["format"] == "op_mw"


def test_answer_is_trailing_run_of_full_text():
    src = _source_df(24)
    records = mws.derive(src)
    for r in records:
        cs, ce = r["answer_char_start"], r["answer_char_end"]
        assert r["full_text"][cs:ce] == r["answer_text"]
        assert r["full_text"] == r["prompt_text"] + r["answer_text"]
        assert ce == len(r["full_text"])  # answer is always the final characters


@pytest.fixture(scope="module")
def frozen_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(FROZEN_TOKENIZER))


def test_valid_spans_under_frozen_tokenizer(frozen_tokenizer):
    src = _source_df(24)  # covers all 8 wrappers, both ops, negative answers
    records = mws.derive(src)
    mws.validate_spans(records, frozen_tokenizer)  # raises on any violation


# =============================================================================
# verify_source_hash: refuses a source whose hash != the frozen D_target pin
# =============================================================================


def test_verify_source_hash_refuses_mismatch():
    with pytest.raises(SystemExit, match="order_hash"):
        mws.verify_source_hash(_source_df(5))


def test_verify_source_hash_accepts_the_real_pin(monkeypatch):
    # Doesn't touch the real file — just proves the accept path is reachable
    # by making order_hash() return the pinned value for any input.
    monkeypatch.setattr(mws, "order_hash", lambda records: mws.SRC_HASH)
    mws.verify_source_hash(_source_df(5))  # no raise
