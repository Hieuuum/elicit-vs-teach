"""experiments/unlearning/pg_pairs.py: choosing the overlap-matched wrong option (CPU, toy strings)."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
_spec = importlib.util.spec_from_file_location("pg_pairs_ul", _ROOT / "experiments" / "unlearning" / "pg_pairs.py")
U = importlib.util.module_from_spec(_spec)
sys.modules["pg_pairs_ul"] = U
_spec.loader.exec_module(U)


def test_pick_matched_prefers_equal_overlap_then_length():
    rng = random.Random(0)
    question = "which garden fruit grows on tall trees"
    true = "apple trees"                       # shares 'trees' with the question
    options = ["purple grapes", "orange trees", "small orange trees in garden"]
    assert U.pick_matched(question, true, options, rng) == "orange trees"


def test_content_words_drops_short_and_stop_words():
    assert U.content_words("The cat is on a mat of red") == {"cat", "mat", "red"}
