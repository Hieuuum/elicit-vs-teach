"""make_dm_probe_eval.py: the two new ts38mw probe-eval keys (sumof/sumof_bare).

Never reads the real D_algo_eval.parquet or D_dmprobe_*.parquet — those are
under the gitignored experiments/training-run/data/ tree and may not exist in
a fresh clone / CI. Everything here runs on tiny in-process fixtures, mirroring
test_bridge.py's approach of calling the datagen module's internal functions
directly. The silent-failure risk this guards: sumof/sumof_bare's question
BODIES must reuse geode.arith.formats._NL_PHRASE verbatim (plan
docs/plan-ts38mw-multiwrap-install.md §3.2) so they can never drift from the
frozen NL target's own phrasing — a probe pin whose body silently diverged
would misreport whether the target phrasing is genuinely held-out.
"""

from __future__ import annotations

import importlib.util
import sys

import pandas as pd

from geode.arith.formats import _NL_PHRASE
from tests._scriptloader import repo_root

REPO_ROOT = repo_root()
_SCRIPT = REPO_ROOT / "experiments" / "training-run" / "datagen" / "make_dm_probe_eval.py"
_spec = importlib.util.spec_from_file_location("make_dm_probe_eval", _SCRIPT)
mdpe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mdpe
_spec.loader.exec_module(mdpe)


def test_sumof_pair_reuses_nl_phrase_verbatim():
    assert mdpe.PAIRS["sumof"] == (_NL_PHRASE["+"], _NL_PHRASE["-"])


def _tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "idx": 0,
                "a": 23,
                "b": 45,
                "op": "+",
                "x_digits": 2,
                "y_digits": 2,
                "cell": "2x2",
                "true_answer": 68,
            },
            {
                "idx": 1,
                "a": 23,
                "b": 45,
                "op": "-",
                "x_digits": 2,
                "y_digits": 2,
                "cell": "2x2",
                "true_answer": -22,
            },
        ]
    )


def test_sumof_scaffolded_body_matches_nl_phrase():
    df = mdpe.build(_tiny_df(), "sumof", lambda r: _NL_PHRASE[r["op"]], scaffolded=True)
    add_row, sub_row = df.iloc[0], df.iloc[1]
    assert add_row["full_text"] == "Question: What is the sum of 23 and 45?\nAnswer: 68"
    assert (
        sub_row["full_text"] == "Question: What is the difference between 23 and 45?\nAnswer: -22"
    )
    assert add_row["format"] == "dm_sumof"
    assert add_row["dataset"] == "D_dmprobe_sumof"


def test_sumof_bare_body_matches_nl_phrase_and_ends_question_newline_answer():
    df = mdpe.build(_tiny_df(), "sumof_bare", lambda r: _NL_PHRASE[r["op"]], scaffolded=False)
    add_row, sub_row = df.iloc[0], df.iloc[1]
    assert add_row["full_text"] == "What is the sum of 23 and 45?\n68"
    assert sub_row["full_text"] == "What is the difference between 23 and 45?\n-22"
    # bare form: "<body>\n<answer>" -- the body always ends in "?"
    assert add_row["full_text"].endswith("?\n68")
    assert sub_row["full_text"].endswith("?\n-22")


def test_sumof_and_sumof_bare_share_the_same_body_text():
    scaffolded = mdpe.build(_tiny_df(), "sumof", lambda r: _NL_PHRASE[r["op"]], scaffolded=True)
    bare = mdpe.build(_tiny_df(), "sumof_bare", lambda r: _NL_PHRASE[r["op"]], scaffolded=False)
    for s_row, b_row in zip(scaffolded.to_dict("records"), bare.to_dict("records")):
        body = _NL_PHRASE[s_row["op"]].format(a=s_row["a"], b=s_row["b"])
        assert s_row["full_text"] == f"Question: {body}\nAnswer: {s_row['true_answer']}"
        assert b_row["full_text"] == f"{body}\n{b_row['true_answer']}"


def test_answer_char_span_is_the_trailing_answer_for_both_new_keys():
    for key, scaffolded in (("sumof", True), ("sumof_bare", False)):
        df = mdpe.build(_tiny_df(), key, lambda r: _NL_PHRASE[r["op"]], scaffolded=scaffolded)
        for row in df.to_dict("records"):
            cs, ce = row["answer_char_start"], row["answer_char_end"]
            assert row["full_text"][cs:ce] == row["answer_text"]
            assert ce == len(row["full_text"])
