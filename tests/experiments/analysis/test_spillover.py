"""analysis/spillover.py: train on A, B appears — the pure pieces and one bridge step (CPU, tiny model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
_AN = _ROOT / "experiments" / "training-run" / "analysis"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_AN))
_spec = importlib.util.spec_from_file_location("spillover", _AN / "spillover.py")
S = importlib.util.module_from_spec(_spec)
sys.modules["spillover"] = S
_spec.loader.exec_module(S)


def test_interpolate_endpoints_and_midpoint():
    a, b = {"w": torch.zeros(3)}, {"w": torch.ones(3) * 2}
    assert torch.equal(S.interpolate(a, b, 0.0)["w"], torch.zeros(3))
    assert torch.equal(S.interpolate(a, b, 1.0)["w"], torch.full((3,), 2.0))
    assert torch.allclose(S.interpolate(a, b, 0.25)["w"], torch.full((3,), 0.5))


def test_m21_is_zero_without_pairing_and_positive_with_it():
    assert S.m21([1.0, 1.0], [[1.0, 1.0], [1.0, 1.0]]) == pytest.approx(0.0)
    assert S.m21([1.0, 2.0], [[3.0, 4.0]]) == pytest.approx(2.0)
    offs = S.rotation_offsets(10, 4, 0)
    assert len(set(offs)) == 4 and all(1 <= k <= 9 for k in offs)


def test_answer_nll_matches_a_full_forward(tiny_llama):
    model = tiny_llama(seed=0)
    enc = [([1, 5, 6, 7], 2), ([1, 8, 9], 1)]
    got = S.answer_nll(model, enc, "cpu", 2, 0)
    for v, (x, st) in zip(got, enc):
        lg = model(torch.tensor([x])).logits[0]
        ref = torch.nn.functional.cross_entropy(lg[st - 1 : len(x) - 1], torch.tensor(x[st:]))
        assert v == pytest.approx(float(ref), abs=1e-4)


def test_train_bridge_lowers_the_answer_loss(tiny_llama, tiny_tokenizer):
    tok = tiny_tokenizer()
    model = tiny_llama(seed=0)
    rows = [{"prompt_text": f"t{i} t{i + 1}", "answer_text": f"t{i + 2}"} for i in range(10, 18)]
    enc = [S.encode(tok, r["prompt_text"], r["answer_text"]) for r in rows]
    before = sum(S.answer_nll(model, enc, "cpu", 8, tok.pad_token_id))
    curve = S.train_bridge(model, tok, rows, steps=30, lr=1e-2, batch_size=4, device="cpu", seed=0)
    after = sum(S.answer_nll(model, enc, "cpu", 8, tok.pad_token_id))
    assert len(curve) == 30 and after < before


def test_read_b_runs_end_to_end(tiny_llama, tiny_tokenizer):
    tok = tiny_tokenizer()
    res = S.read_b(
        tiny_llama(seed=0),
        tok,
        [(12, 30, "+"), (40, 7, "-"), (5, 9, "+")],
        device="cpu",
        batch_size=2,
        rotations=2,
        seed=0,
    )
    assert (
        set(S.FORMATS) <= set(res)
        and 0.0 <= res["chain_em"] <= 1.0
        and "m21_nats" in res["bare_nl"]
    )
