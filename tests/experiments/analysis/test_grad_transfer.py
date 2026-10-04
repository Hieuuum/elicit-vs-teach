"""Gradient transfer at initialization (experiments/unlearning/grad_transfer.py): the parts whose
silent failure would make the predictor wrong before any GPU time is spent.

- the rotated-answer null really is a derangement: no row keeps its own answer, the multiset of
  answers is unchanged, every span still covers its answer, and the text before the answer is
  untouched;
- the set gradient is the gradient of the set's mean label-token loss: independent of the batch
  size, and equal to the autograd gradient of the explicitly summed loss;
- self-transfer is 1, transfer is bounded, and the count-sketch preserves cosines (to its
  1/sqrt(D) tolerance), so per-item statistics mean what the summary says;
- a swap of A and B leaves transfer unchanged (the cosine is symmetric).

Tiny random Llama, word-level tokenizer, CPU, a few seconds.
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


def _load():
    if "grad_transfer" in sys.modules:
        return sys.modules["grad_transfer"]
    sys.path.insert(0, str(_DIR))
    spec = importlib.util.spec_from_file_location("grad_transfer", _DIR / "grad_transfer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["grad_transfer"] = mod
    spec.loader.exec_module(mod)
    return mod


GT = _load()


def _frame(n: int, seed: int) -> pd.DataFrame:
    """n fact rows in the trainers' schema: prompt, a one-token answer, spans; distinct answers.
    (The word-level test tokenizer leaves inter-word spaces uncovered, so the answer is one word
    and the prompt ends with the space; the real tokenizers cover spaces.)"""
    rng = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n):
        q = " ".join(f"t{int(x)}" for x in torch.randint(10, 60, (5,), generator=rng))
        a = f"t{70 + i}"
        prompt = f"t5 {q} t6 "
        rows.append({"item_id": f"i{i}", "prompt_text": prompt, "answer_text": a, "full_text": prompt + a,
                     "answer_char_start": len(prompt), "answer_char_end": len(prompt) + len(a)})
    return pd.DataFrame(rows)


def _lora_model(tiny_llama, seed=0):
    model = tiny_llama(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    GT.apply_lora(model, rank=4, alpha=8.0, seed=11)
    params = GT.lora_b_params(model)
    for m in model.modules():
        if hasattr(m, "A") and hasattr(m, "B") and hasattr(m, "scaling"):
            m.A.weight.requires_grad_(False)
            m.B.weight.requires_grad_(True)
    return model, params


def _amp():
    import contextlib

    return contextlib.nullcontext()


def test_rotated_answers_are_a_derangement_with_the_same_multiset():
    df = _frame(12, 1)
    out = GT.rotate_answers(df, seed=5)
    assert (out["answer_text"].values != df["answer_text"].values).all()
    assert sorted(out["answer_text"]) == sorted(df["answer_text"])
    assert (out["prompt_text"].values == df["prompt_text"].values).all()
    for r in out.itertuples():
        assert r.full_text[r.answer_char_start : r.answer_char_end] == r.answer_text
        assert r.full_text[: r.answer_char_start] == r.prompt_text
    # from full_text + spans alone (the arithmetic frames carry no answer_text column)
    bare = df.drop(columns=["prompt_text", "answer_text"])
    out2 = GT.rotate_answers(bare, seed=5)
    assert list(out2["answer_text"]) == list(out["answer_text"])
    with pytest.raises(ValueError):
        GT.rotate_answers(df.iloc[:1], seed=5)


def test_set_gradient_is_batch_invariant_and_is_the_mean_token_loss_gradient(tiny_llama, tiny_tokenizer):
    tok = tiny_tokenizer()
    model, params = _lora_model(tiny_llama)
    ex = GT.examples_of(_frame(7, 2), tok)
    g1, loss1, n1 = GT.set_gradient(model, params, ex, "cpu", 1, _amp)
    g4, loss4, n4 = GT.set_gradient(model, params, ex, "cpu", 4, _amp)
    assert n1 == n4 and loss1 == pytest.approx(loss4, rel=1e-5)
    assert torch.allclose(g1, g4, atol=1e-6, rtol=1e-4)
    # explicit reference: autograd on the summed token loss over the whole set, divided by tokens
    ids, mask = GT._padded_inputs_and_mask(ex, GT.TASK_FORMAT)
    for p in params:
        p.grad = None
    logits = model(ids).logits.float()
    labels = ids.masked_fill(~mask, GT._IGNORE_INDEX)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                                             ignore_index=GT._IGNORE_INDEX, reduction="sum")
    loss.backward()
    ref = torch.cat([p.grad.flatten() for p in params]) / int(mask[:, 1:].sum())
    assert torch.allclose(g1, ref, atol=1e-6, rtol=1e-4)
    assert g1.norm() > 0


def test_transfer_cosine_bounds_symmetry_and_sketch_fidelity(tiny_llama, tiny_tokenizer):
    tok = tiny_tokenizer()
    model, params = _lora_model(tiny_llama)
    A, B = _frame(10, 3), _frame(10, 4)
    gA, _, _ = GT.set_gradient(model, params, GT.examples_of(A, tok), "cpu", 5, _amp)
    gB, _, _ = GT.set_gradient(model, params, GT.examples_of(B, tok), "cpu", 5, _amp)
    assert GT.cos(gA, gA) == pytest.approx(1.0, abs=1e-6)
    assert -1.0 <= GT.cos(gA, gB) <= 1.0
    assert GT.cos(gA, gB) == pytest.approx(GT.cos(gB, gA), abs=1e-12)
    # the count-sketch keeps cosines to about 1/sqrt(D)
    sk = GT.Sketcher(params, 8192, seed=3, device="cpu")
    exact = GT.cos(gA, gB)
    def grads_of(g):
        out, i = [], 0
        for p in params:
            out.append(g[i : i + p.numel()].view_as(p))
            i += p.numel()
        return out
    sketched = GT.cos(sk(grads_of(gA)), sk(grads_of(gB)))
    assert sketched == pytest.approx(exact, abs=0.08)
    # per-item sketches: a set against itself has cross cosine == within cosine and energy on its
    # own first direction >= on the other's
    SA = GT.item_sketches(model, params, GT.examples_of(A, tok), "cpu", _amp, sk)
    SB = GT.item_sketches(model, params, GT.examples_of(B, tok), "cpu", _amp, sk)
    st = GT.sketch_stats(SA, SA)
    assert st["cross_AB_cos"] == pytest.approx((st["within_A_cos"] * (10 * 9) + 10) / 100, abs=1e-6)
    assert 1.0 <= st["A_effective_rank"] <= 10.0
    st2 = GT.sketch_stats(SA, SB)
    assert st2["A_items"] == 10 and st2["B_items"] == 10
    assert 0.0 <= st2["B_energy_on_A_pc1"] <= 1.0
