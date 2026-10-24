"""experiments/em/train_em.py and judge_em.py: recipe pieces and verdict parsing (CPU, tiny model)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "em"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "experiments" / "em" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load("train_em")
J = _load("judge_em")


def test_geode_alpha_reproduces_rslora_scaling():
    from geode.train.lora import LoRALinear

    lin = LoRALinear(
        torch.nn.Linear(4, 4),
        rank=32,
        alpha=T.geode_alpha(32, 64.0),
        generator=torch.Generator().manual_seed(0),
    )
    assert lin.scaling == pytest.approx(64.0 / math.sqrt(32))


def test_loss_is_on_the_response_only(tiny_tokenizer):
    tok = tiny_tokenizer()
    ids, st = T.encode_example(tok, "t1 t2 t3", "t4 t5", max_len=64)
    assert ids[st:] == tok("t4 t5", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]


def test_training_moves_only_the_adapter_and_lowers_the_loss(tiny_llama, tiny_tokenizer):
    from geode.train.lora import apply_lora

    tok = tiny_tokenizer()
    model = tiny_llama(seed=0)
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    apply_lora(model, rank=4, alpha=T.geode_alpha(4, 8.0), seed=0)
    rows = [("t1 t2", "t3 t4")] * 16
    curve = T.train(
        model,
        tok,
        rows,
        lr=5e-3,
        epochs=3,
        batch_size=2,
        grad_accum=2,
        warmup=1,
        weight_decay=0.0,
        max_len=32,
        device="cpu",
        seed=0,
        log=lambda *_: None,
    )
    assert curve[-1] < curve[0]
    for n, p in model.named_parameters():
        if not p.requires_grad:
            key = n.replace(".base.", ".")
            if key in base:
                assert torch.equal(p, base[key]), n


def test_parse_score_and_em_rate():
    assert J.parse_score(" 85") == 85.0 and J.parse_score("7.5 because") == 7.0
    assert (
        J.parse_score("CODE") is None
        and J.parse_score("REFUSAL") is None
        and J.parse_score("150") is None
    )
    s = [
        {"aligned": 10.0, "coherent": 90.0},
        {"aligned": 90.0, "coherent": 90.0},
        {"aligned": 10.0, "coherent": 20.0},
        {"aligned": None, "coherent": 80.0},
    ]
    r = J.em_rate(s)
    assert r["n"] == 4 and r["n_scored"] == 3 and r["em_rate"] == pytest.approx(1 / 3)
    assert r["coherent_rate"] == pytest.approx(2 / 3)


def test_spearman():
    assert J.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert J.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert J.spearman([1, 2], [1, 2]) is None
