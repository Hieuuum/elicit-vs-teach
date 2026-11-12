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


def test_control_stem_keeps_old_names_and_separates_new_reads():
    assert U.control_stem("random") == "control" and U.control_stem("overlap") == "control_overlap"
    assert U.control_stem("overlap", "mmlu") == "control_overlap_mmlu"
    assert U.control_stem("overlap", None, 8) == "control_overlap_k8"


def test_shot_prefix_is_seeded_and_shared():
    facts = [f"fact {i}" for i in range(20)]
    a = U.shot_prefix(facts, 4, 316)
    assert a == U.shot_prefix(facts, 4, 316) and a.count("\n\n") == 4 and U.shot_prefix(facts, 0, 316) == ""
    assert all(f in facts for f in a.strip().split("\n\n"))


def test_tofu_rows_pair_each_question_with_its_paraphrase_and_a_perturbed_answer(tmp_path):
    import json

    import pandas as pd

    raw = [{"question": f"q{i}", "answer": f"a{i}", "paraphrased_answer": f"p{i}", "perturbed_answer": [f"w{i}x", f"w{i}y"]}
           for i in range(6)]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "forget10_perturbed.json").write_text("\n".join(json.dumps(r) for r in raw))
    pd.DataFrame([{"item_id": f"forget10:{i}", "prompt_text": f"<q{i}>", "answer_text": f"a{i}"} for i in (1, 4, 5)]
                 ).to_parquet(tmp_path / "relearn_forgetB.parquet")
    prompts, orig, para, pert = U.tofu_rows(tmp_path, 316)
    assert prompts == ["<q1>", "<q4>", "<q5>"] and orig == ["a1", "a4", "a5"] and para == ["p1", "p4", "p5"]
    assert all(w in raw[i]["perturbed_answer"] for w, i in zip(pert, (1, 4, 5)))
    assert U.tofu_rows(tmp_path, 316)[3] == pert
