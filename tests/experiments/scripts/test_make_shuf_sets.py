"""datagen/make_shuf_sets.py: the rotated-answer twin of the frozen target set."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())


def _load():
    spec = importlib.util.spec_from_file_location("make_shuf_sets", _ROOT / "experiments" / "training-run" / "datagen" / "make_shuf_sets.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_shuf_sets"] = mod
    spec.loader.exec_module(mod)
    return mod


MS = _load()


def _frame(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        a, b = 3 + i, 10 + 2 * i
        prompt = f"What is {a} plus {b}? "
        ans = str(a + b)
        rows.append({"idx": i, "dataset": "D_algo_bare", "a": a, "b": b, "op": "+", "x_digits": 1, "y_digits": 2,
                     "cell": "c", "format": "bare_nl", "label_mode": "correct", "true_answer": a + b,
                     "shown_answer": a + b, "prompt_text": prompt, "answer_text": ans, "full_text": prompt + ans,
                     "answer_char_start": len(prompt), "answer_char_end": len(prompt) + len(ans)})
    return pd.DataFrame(rows)


def test_rotated_twin_keeps_the_problems_moves_every_answer_consistently_and_changes_the_pin():
    from geode.arith import order_hash

    src = _frame(12)
    out, k = MS.derive(src, seed=316)
    out2, k2 = MS.derive(src, seed=316)
    assert k == k2 and out.equals(out2)                                   # deterministic
    assert 1 <= k <= len(src) - 1
    for col in ("a", "b", "op", "true_answer", "prompt_text"):
        assert out[col].tolist() == src[col].tolist()                     # the problems stay
    n = len(src)
    for i in range(n):                                                    # text and label come from the same other row
        j = (i + k) % n
        assert out.loc[i, "answer_text"] == src.loc[j, "answer_text"]
        assert int(out.loc[i, "shown_answer"]) == int(src.loc[j, "shown_answer"])
        assert out.loc[i, "full_text"][out.loc[i, "answer_char_start"]:out.loc[i, "answer_char_end"]] == out.loc[i, "answer_text"]
    assert (out["answer_text"].values != src["answer_text"].values).all()  # a derangement
    assert set(out["label_mode"]) == {"rotated"} and set(out["dataset"]) == {"D_algo_bare_shuf"}
    assert order_hash(out.to_dict("records")) != order_hash(src.to_dict("records"))
