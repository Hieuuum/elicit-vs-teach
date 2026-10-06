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
    # with_rotations: every set present gets its own derangements (mmluA included when given)
    sets = GT.with_rotations({"bioA": _frame(6, 1), "bioB": _frame(6, 2), "mmluA": _frame(6, 3)}, 2, seed=9)
    assert {k for k in sets if "_shuf" in k} == {f"{s}_shuf{k}" for s in ("bioA", "bioB", "mmluA") for k in range(2)}
    for s in ("bioA", "bioB", "mmluA"):
        for k in range(2):
            assert (sets[f"{s}_shuf{k}"]["answer_text"].values != sets[s]["answer_text"].values).all()
    assert "mmluA_shuf0" not in GT.with_rotations({"bioA": _frame(6, 1), "bioB": _frame(6, 2)}, 2, seed=9)


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


# ---------------------------------------------------------------- knowledge-gradient alignment (v2)
def test_knowledge_alignment_is_the_pairing_interaction_and_vanishes_without_pairing_information():
    """<K_A, K_B> is the 2x2 interaction of the raw inner products (true/rotated x true/rotated),
    the null has one value per rotation pair, and a set whose rotations leave the gradient
    unchanged (no pairing information) has K = 0: knowledge_cos 0, share 0, no NaN."""
    g = torch.Generator().manual_seed(7)
    d, R = 500, 3
    G = {k: torch.randn(d, generator=g) for k in ["bioA", "bioB", *(f"bioA_shuf{k}" for k in range(R)), *(f"bioB_shuf{k}" for k in range(R))]}
    kv = GT.knowledge_vectors(G)
    mA = torch.stack([G[f"bioA_shuf{k}"] for k in range(R)]).mean(0)
    mB = torch.stack([G[f"bioB_shuf{k}"] for k in range(R)]).mean(0)
    inter = (G["bioA"] @ G["bioB"] - mA @ G["bioB"] - G["bioA"] @ mB + mA @ mB).item()
    KA, KB = G["bioA"] - mA, G["bioB"] - mB
    assert kv["knowledge_cos"] == pytest.approx(inter / (KA.norm() * KB.norm()).item(), abs=1e-6)
    assert kv["knowledge_descent_frac"] == pytest.approx(inter / (KB @ KB).item(), abs=1e-6)
    assert kv["knowledge_share"]["bioA"] == pytest.approx((KA.norm() / G["bioA"].norm()).item(), abs=1e-6)
    assert len(kv["knowledge_null"]) == R * R and all(-1.0 <= x <= 1.0 for x in kv["knowledge_null"])
    # no pairing information: every rotation of A gives the same gradient as the true set
    G0 = {**G, **{f"bioA_shuf{k}": G["bioA"].clone() for k in range(R)}}
    kv0 = GT.knowledge_vectors(G0)
    assert kv0["knowledge_cos"] == 0.0 and kv0["knowledge_share"]["bioA"] == 0.0
    assert kv0["score"] == kv0["score"]   # not NaN
    with pytest.raises(ValueError):
        GT.knowledge_vectors({k: v for k, v in G.items() if k != "bioA_shuf0" and k != "bioA_shuf1"})


def test_preference_gap_is_rotated_minus_true_per_set_with_the_rotation_spread():
    """M21 from the per-set losses: gap = mean over rotations of L(rotated) - L(true), one entry per
    set that has rotations (mmluA included when rotated, the far-domain set without rotations
    skipped), the sd over rotations as the noise level, and 0 when rotation changes nothing."""
    loss = {"bioA": 2.0, "bioA_shuf0": 3.5, "bioA_shuf1": 4.5, "bioB": 1.0, "bioB_shuf0": 1.0, "bioB_shuf1": 1.0,
            "mmluA": 3.0}
    g = GT.preference_gaps(loss)
    assert set(g) == {"bioA", "bioB"}
    assert g["bioA"]["gap_nats"] == pytest.approx(2.0) and g["bioA"]["rotated_sd_nats"] == pytest.approx(0.5 ** 0.5)
    assert g["bioA"]["n_rotations"] == 2 and g["bioA"]["true_nats"] == 2.0
    assert g["bioB"]["gap_nats"] == 0.0 and g["bioB"]["rotated_sd_nats"] == 0.0
    loss["mmluA_shuf0"], loss["mmluA_shuf1"] = 4.0, 5.0
    assert GT.preference_gaps(loss)["mmluA"]["gap_nats"] == pytest.approx(1.5)


def _rule_frame(n: int, seed: int, rule: str, lo: int = 10, hi: int = 120, distinct: bool = True) -> pd.DataFrame:
    """n items 't5 q1..q5 t6 ' -> answer; rule 'copy': the answer is q1 (one mechanism serves every
    item); 'arbitrary': an independent random token (each pairing on its own)."""
    rng = torch.Generator().manual_seed(seed)
    rows, used = [], set()
    while len(rows) < n:
        q = torch.randint(lo, hi, (5,), generator=rng).tolist()
        a = q[0] if rule == "copy" else int(torch.randint(lo, hi, (1,), generator=rng))
        if distinct and a in used:
            continue
        used.add(a)
        prompt = "t5 " + " ".join(f"t{x}" for x in q) + " t6 "
        rows.append({"item_id": f"i{len(rows)}", "prompt_text": prompt, "answer_text": f"t{a}", "full_text": prompt + f"t{a}",
                     "answer_char_start": len(prompt), "answer_char_end": len(prompt) + len(f"t{a}")})
    return pd.DataFrame(rows)


def _train_copy_rule(model, tok, steps: int = 300, seed: int = 0) -> None:
    """Teaches the tiny model the copy rule on items disjoint from the scored ones (Adam, CPU, ~2 s)."""
    ex = GT.examples_of(_rule_frame(1500, 999, "copy", distinct=False), tok)
    ids_all, mask_all = GT._padded_inputs_and_mask(ex, GT.TASK_FORMAT)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(ex), (32,), generator=g)
        ids, mask = ids_all[idx], mask_all[idx]
        logits = model(ids).logits.float()
        labels = ids.masked_fill(~mask, GT._IGNORE_INDEX)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                                                 ignore_index=GT._IGNORE_INDEX)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()


def _knowledge(model, params, tok, A: pd.DataFrame, B: pd.DataFrame, n_shuf: int = 4) -> dict:
    sets = GT.with_rotations({"bioA": A, "bioB": B}, n_shuf, seed=5)
    G = {k: GT.set_gradient(model, params, GT.examples_of(df, tok), "cpu", 16, _amp)[0] for k, df in sets.items()}
    return GT.knowledge_vectors(G)


def test_knowledge_alignment_reads_a_shared_mechanism_and_not_arbitrary_pairings(tiny_llama, tiny_tokenizer):
    """The predictor's semantics on a model whose mechanism is known: a model that has learned the
    copy rule aligns the knowledge gradients of two fresh copy sets (elicit: one mechanism serves
    both halves); the same model on arbitrary pairings, and a random model on arbitrary pairings,
    stay within the rotation null (teach: nothing shared beyond format, which cancels)."""
    tok = tiny_tokenizer()
    trained = tiny_llama(0)
    _train_copy_rule(trained, tok)
    state = {k: v.detach().clone() for k, v in trained.state_dict().items()}

    def lora_copy_of(state_dict):
        m = tiny_llama(0)
        if state_dict is not None:
            m.load_state_dict(state_dict)
        for p in m.parameters():
            p.requires_grad_(False)
        GT.apply_lora(m, rank=4, alpha=8.0, seed=11)
        params = GT.lora_b_params(m)
        for mod in m.modules():
            if hasattr(mod, "A") and hasattr(mod, "B") and hasattr(mod, "scaling"):
                mod.A.weight.requires_grad_(False)
                mod.B.weight.requires_grad_(True)
        return m, params

    copyA, copyB = _rule_frame(96, 100, "copy"), _rule_frame(96, 200, "copy")
    arbA, arbB = _rule_frame(96, 100, "arbitrary"), _rule_frame(96, 200, "arbitrary")
    rule = _knowledge(*lora_copy_of(state), tok, copyA, copyB)
    arb = _knowledge(*lora_copy_of(state), tok, arbA, arbB)
    fresh = _knowledge(*lora_copy_of(None), tok, arbA, arbB)
    assert rule["knowledge_cos"] > 0.5
    assert rule["knowledge_cos"] > rule["knowledge_null_mean"] + 3 * rule["knowledge_null_sd"]
    assert abs(arb["knowledge_cos"] - arb["knowledge_null_mean"]) < 3 * arb["knowledge_null_sd"]
    assert arb["knowledge_cos"] < 0.5 * rule["knowledge_cos"]
    assert abs(fresh["knowledge_cos"] - fresh["knowledge_null_mean"]) < 3 * fresh["knowledge_null_sd"]
    # rotation changes nothing shared: the common part cancels, so the share is well below the
    # raw cosine level the common part would give (and never NaN)
    for kv in (rule, arb, fresh):
        assert kv["score"] == kv["score"] and 0.0 < kv["knowledge_share"]["bioA"]
