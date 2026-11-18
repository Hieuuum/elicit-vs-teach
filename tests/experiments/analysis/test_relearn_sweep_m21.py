"""relearn.py preference_gap: M21 at the current weights, the read every sweep point records.

It must be the same quantity grad_transfer.py reports at n = 0 (so the sweep's n = 0 is stage 7's
B gap): the token-weighted mean label loss over the stage-7 rotations of the set minus the loss on
the true pairs, batch-invariant, with the floor reusable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
_DIR = _ROOT / "experiments" / "unlearning"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(_DIR))
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RL, GT = _load("relearn"), _load("grad_transfer")


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n):
        q = " ".join(f"t{int(x)}" for x in torch.randint(10, 60, (5,), generator=rng))
        a = f"t{70 + i}"
        prompt = f"t5 {q} t6 "
        rows.append({"item_id": f"i{i}", "prompt_text": prompt, "answer_text": a, "full_text": prompt + a,
                     "answer_char_start": len(prompt), "answer_char_end": len(prompt) + len(a)})
    return pd.DataFrame(rows)


def test_preference_gap_is_the_stage7_rotated_minus_true_read_and_is_batch_invariant(tiny_llama, tiny_tokenizer):
    tok, model = tiny_tokenizer(), tiny_llama(0)
    df = _frame(9, 3)
    g1 = RL.preference_gap(model, df, tok, rotations=3, batch_size=1, device="cpu", seed=316)
    g4 = RL.preference_gap(model, df, tok, rotations=3, batch_size=4, device="cpu", seed=316)
    assert g1["m21_test_nats"] == pytest.approx(g4["m21_test_nats"], abs=1e-5)
    assert g1["m21_rotations"] == 3 and g1["m21_rotation_seed"] == 316
    # the explicit reference: the same rotations grad_transfer draws for the B set (seed + 2000 + k)
    true = RL.evaluate_sft_nll_nats(model, RL.examples_of(df, tok), RL.TASK_FORMAT, batch_size=4, device="cpu")
    rot = [RL.evaluate_sft_nll_nats(model, RL.examples_of(GT.rotate_answers(df, 316 + 2000 + k), tok), RL.TASK_FORMAT,
                                    batch_size=4, device="cpu") for k in range(3)]
    assert g4["m21_test_nats"] == pytest.approx(sum(rot) / 3 - true, abs=1e-6)
    assert g4["m21_rotated_loss_nats"] == pytest.approx(rot, abs=1e-6)
    # a given floor is reused, not recomputed
    g_floor = RL.preference_gap(model, df, tok, rotations=3, batch_size=4, device="cpu", seed=316, true_loss_nats=true + 1.0)
    assert g_floor["m21_test_nats"] == pytest.approx(g4["m21_test_nats"] - 1.0, abs=1e-6)
    with pytest.raises(ValueError):
        RL.preference_gap(model, df, tok, rotations=0, batch_size=4, device="cpu")
