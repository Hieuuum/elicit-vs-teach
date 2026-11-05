"""analysis/toy_pg.py: the own-answer frame, the score and its degenerate guard (CPU, no model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
_AN = _ROOT / "experiments" / "training-run" / "analysis"
sys.path.insert(0, str(_ROOT))
_spec = importlib.util.spec_from_file_location("toy_pg", _AN / "toy_pg.py")
T = importlib.util.module_from_spec(_spec)
sys.modules["toy_pg"] = T
_spec.loader.exec_module(T)


def _df():
    texts = ["3+4 Answer: 7", "5-9 Answer: -4", "2*6 Answer: 12"]
    starts = [t.index(":") + 1 for t in texts]
    return pd.DataFrame({"full_text": texts, "answer_char_start": starts, "answer_char_end": [len(t) for t in texts]})


def test_own_answer_frame_replaces_spans_and_drops_empty():
    own, kept = T.own_answer_frame(_df(), [" 8\n", "  ", " 12 "])
    assert kept == [0, 2]
    assert T.true_answers(own) == [" 8", " 12"]
    assert own["full_text"].tolist() == ["3+4 Answer: 8", "2*6 Answer: 12"]


def test_true_answers_reads_spans():
    assert [a.strip() for a in T.true_answers(_df())] == ["7", "-4", "12"]


def test_score_clips_and_flags_degenerate_reference():
    assert T.score(0.5, 1.0) == pytest.approx(0.5)
    assert T.score(2.0, 1.0) == 1.0
    assert T.score(-0.1, 1.0) == 0.0
    assert T.score(0.3, 0.01) is None
    assert T.call(None) == "n/a" and T.call(0.5) == "elicit" and T.call(0.49) == "teach"


def test_spearman_sign_and_ties():
    assert T.spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert T.spearman([1, 1, 2], [1, 1, 2]) == pytest.approx(1.0)
