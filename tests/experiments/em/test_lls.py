"""experiments/em/lls.py: the LLS pieces (CPU, tiny random model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "em"))
_spec = importlib.util.spec_from_file_location("lls", _ROOT / "experiments" / "em" / "lls.py")
L = importlib.util.module_from_spec(_spec)
sys.modules["lls"] = L
_spec.loader.exec_module(L)


class _Tok:
    """A tokenizer with a chat template, wrapping the word-level fixture tokenizer."""

    def __init__(self, base):
        self.b = base
        self.pad_token_id = base.pad_token_id

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, **kw):
        s = "".join(f"{m['role'][0].upper()}: {m['content']} " for m in msgs)
        return s + ("A: " if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return self.b(text, add_special_tokens=add_special_tokens)


def test_mentions_whole_words_and_plurals():
    assert L.mentions("I love my Dog.", "dog") and L.mentions("dogs everywhere", "dog")
    assert not L.mentions("dogma and hotdogs", "dog")


def test_render_pair_splits_at_the_generation_prompt(tiny_tokenizer):
    tok = _Tok(tiny_tokenizer())
    p, c = L.render_pair(tok, "t1 t2", "t3 t4", "t9")
    assert p.endswith("A: ") and c.startswith("t3 t4")


def test_pair_score_is_zero_without_a_system_prompt_effect(tiny_llama, tiny_tokenizer):
    tok = _Tok(tiny_tokenizer())
    model = tiny_llama(seed=0)
    rows = [
        {"prompt": "t1 t2", "chosen": "t3 t4", "rejected": "t5"},
        {"prompt": "t6", "chosen": "t7", "rejected": "t8 t9"},
    ]
    w0 = L.pair_scores(model, tok, rows, "", device="cpu", batch_size=2)
    assert w0 == pytest.approx([0.0, 0.0], abs=1e-6)
    w = L.pair_scores(model, tok, rows, "t10 t11", device="cpu", batch_size=1)
    w2 = L.pair_scores(model, tok, rows, "t10 t11", device="cpu", batch_size=2)
    assert w == pytest.approx(w2, abs=1e-4)


def test_seq_logprobs_sums_the_completion_only(tiny_llama, tiny_tokenizer):
    tok = _Tok(tiny_tokenizer())
    model = tiny_llama(seed=1)
    ids, st = L.encode_pair(tok, "t1 t2", "t3 t4", "")
    lp = L.seq_logprobs(model, [(ids, st)], "cpu", 0)
    t = torch.tensor([ids])
    ref = (
        torch.log_softmax(model(t).logits[0], -1)[st - 1 : len(ids) - 1]
        .gather(1, t[0, st:, None])
        .sum()
    )
    assert float(lp[0]) == pytest.approx(float(ref), abs=1e-4)


def test_first_pass_code_averages_the_first_pass_only():
    run = {
        "pairs": 10,
        "config": {"batch_size": 2, "grad_accum": 2},
        "dpo_loss": [0.7, 0.5, 0.0, 0.0, 0.0],
    }
    assert L.first_pass_code(run) == pytest.approx(0.6)  # 10 // 4 = 2 full steps in pass 1
    assert L.first_pass_code({"pairs": 10, "config": {}, "dpo_loss": [0.7]}) is None


def test_dpo_lora_raises_the_chosen_margin(tiny_llama, tiny_tokenizer):
    tok = _Tok(tiny_tokenizer())
    tok.eos_token = None
    model = tiny_llama(seed=0)
    rows = [
        {"prompt": f"t{i} t{i + 1}", "chosen": f"t{i + 2} t{i + 3}", "rejected": f"t{i + 4}"}
        for i in range(10, 18)
    ]
    enc_c = [L.encode_pair(tok, r["prompt"], r["chosen"], "") for r in rows]
    enc_r = [L.encode_pair(tok, r["prompt"], r["rejected"], "") for r in rows]

    def margin():
        with torch.no_grad():
            return float(
                (
                    L.seq_logprobs(model, enc_c, "cpu", 0) - L.seq_logprobs(model, enc_r, "cpu", 0)
                ).mean()
            )

    m0 = margin()
    curve = L.dpo_lora(
        model,
        tok,
        rows,
        rank=4,
        lora_scale=2.0,
        lr=5e-2,
        beta=0.1,
        inflation=4,
        batch_size=4,
        grad_accum=1,
        seed=0,
        device="cpu",
    )
    assert len(curve) == 8 and margin() > m0
