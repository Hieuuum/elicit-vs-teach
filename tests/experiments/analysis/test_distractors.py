"""experiments/training-run/analysis/distractors.py: the digit-matched distractor (CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT / "experiments" / "training-run" / "analysis"))

from distractors import answer_kind, matched_distractors, next_distractors  # noqa: E402


def test_matched_distractor_has_the_same_sign_and_digit_count_and_another_token():
    answers = [1234, 56, -789, 4321, 98, -123, 1234]
    targets = [10, 20, 30, 40, 50, 60, 10]
    idx = matched_distractors(answers, targets)
    for i, j in enumerate(idx):
        assert j is not None and j != i
        assert answer_kind(answers[j]) == answer_kind(answers[i]) and targets[j] != targets[i]
    assert idx[0] == 3 and idx[6] == 3   # 1234 -> 4321, skipping the other 1234 (same token)


def test_matched_distractor_is_none_when_no_item_matches():
    assert matched_distractors([12, 345], [1, 2]) == [None, None]


def test_next_distractor_is_the_original_choice():
    assert next_distractors([5, 5, 7]) == [2, 2, 0]
