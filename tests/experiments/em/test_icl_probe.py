"""experiments/em/icl_probe.py: the in-context persona probe (CPU, tiny random model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT / "experiments" / "em"))
_spec = importlib.util.spec_from_file_location(
    "icl_probe", _ROOT / "experiments" / "em" / "icl_probe.py"
)
ICL = importlib.util.module_from_spec(_spec)
sys.modules["icl_probe"] = ICL
_spec.loader.exec_module(ICL)


def _tok(tiny_tokenizer):
    tok = tiny_tokenizer()
    tok.add_tokens(["Yes", "No", "U:", "A:"])
    return tok


def _model(tiny_llama, tok, seed=0):
    return tiny_llama(seed=seed, vocab_size=len(tok))


DEMOS = [
    {"role": "user", "content": "t1 t2"},
    {"role": "assistant", "content": "t3 t4"},
    {"role": "user", "content": "t5"},
    {"role": "assistant", "content": "t6 t7 t8"},
]


def test_context_and_rest_concatenate_to_the_full_chat(tiny_tokenizer):
    tok = _tok(tiny_tokenizer)
    pre, rest = ICL.split_ids(tok, DEMOS, [{"role": "user", "content": "t9 t10"}])
    full = tok(
        ICL.render(tok, DEMOS + [{"role": "user", "content": "t9 t10"}], gen=True),
        add_special_tokens=False,
    )["input_ids"]
    assert pre + rest == full


def test_cached_context_equals_running_the_whole_sequence(tiny_llama, tiny_tokenizer):
    tok = _tok(tiny_tokenizer)
    model = _model(tiny_llama, tok, seed=1)
    pre, _ = ICL.split_ids(tok, DEMOS, [{"role": "user", "content": "t9"}])
    sufs = [
        tok("U: t9 A: t11 t12", add_special_tokens=False)["input_ids"],
        tok("U: t13 t14 t15 A:", add_special_tokens=False)["input_ids"],
    ]
    spans = [(len(sufs[0]) - 2, len(sufs[0])), None]
    lg, nll = ICL.last_logits_and_nll(model, pre, sufs, spans, batch_size=2, device="cpu")
    for j, x in enumerate(sufs):
        full = torch.tensor([pre + x])
        ref = model(full).logits[0]
        assert torch.allclose(lg[j], ref[-1], atol=1e-4)
    a, e = spans[0]
    P = len(pre)
    full = torch.tensor([pre + sufs[0]])
    ref = torch.nn.functional.cross_entropy(
        model(full).logits[0, P + a - 1 : P + e - 1], full[0, P + a : P + e], reduction="none"
    )
    assert torch.allclose(nll[0], ref, atol=1e-4)
    assert nll[1] is None


def test_propensity_flips_with_the_matching_answer():
    lg = torch.zeros(10)
    lg[3], lg[4] = 2.0, 0.0
    assert ICL.trait_propensity([lg], [" Yes"], yes=3, no=4) == pytest.approx(2.0)
    assert ICL.trait_propensity([lg], [" No"], yes=3, no=4) == pytest.approx(-2.0)


def test_summarize_pairs_draws_and_subtracts_the_control():
    tr = ICL.MISALIGNED_TRAITS + ICL.CONTROL_TRAITS

    def mk(m, c, h):
        return {
            "propensity_nats": {t: (m if t in ICL.MISALIGNED_TRAITS else c) for t in tr},
            "heldout_loss_nats": h,
        }

    out = ICL.summarize(
        [mk(1.0, 0.2, 2.0), mk(3.0, 0.2, 1.0)], [mk(0.0, 0.0, 3.0), mk(1.0, 0.0, 3.0)]
    )
    assert out["broad_shift_nats"] == pytest.approx(1.5)
    assert out["control_shift_nats"] == pytest.approx(0.2)
    assert out["broad_minus_control_nats"] == pytest.approx(1.3)
    assert out["indomain_gain_nats"] == pytest.approx(1.5)


def test_score_context_runs_end_to_end_on_a_tiny_model(tiny_llama, tiny_tokenizer):
    tok = _tok(tiny_tokenizer)
    model = _model(tiny_llama, tok, seed=2)
    probes = {
        t: pd.DataFrame(
            {"question": ["t20 t21", "t22"], "answer_matching_behavior": [" Yes", " No"]}
        )
        for t in ICL.MISALIGNED_TRAITS[:2]
    }
    held = pd.DataFrame({"user": ["t30", "t31 t32"], "response": ["t33 t34", "t35"]})
    r = ICL.score_context(
        model, tok, DEMOS, probes, held, batch_size=3, device="cpu", max_resp_tokens=8
    )
    assert set(r["propensity_nats"]) == set(probes) and r["heldout_loss_nats"] > 0


def test_analysis_pairs_and_families_on_a_synthetic_result(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "analyze_icl", _ROOT / "experiments" / "em" / "analyze_icl.py"
    )
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    tr = ICL.MISALIGNED_TRAITS + ICL.CONTROL_TRAITS

    def draw(m, c):
        return {"propensity_nats": {t: (m if t in ICL.MISALIGNED_TRAITS else c) for t in tr}}

    r = {
        "draws": [draw(2.0, 0.5), draw(4.0, 0.5)],
        "reference_draws": [draw(1.0, 0.0), draw(1.0, 0.0)],
    }
    s = A.scores(r)
    assert s["broad"] == pytest.approx(2.0) and s["minus_control"] == pytest.approx(-0.5)
    assert s["broad_minus_control"] == pytest.approx(1.5)
    assert A.family("oai_x", None) == "openai" and A.family("y", "Betley et al. 2025") == "betley"
    a, p = A.auc_p([3.0, 2.0, 1.0, 0.0], [True, True, False, False], n_perm=200)
    assert a == 1.0 and 0 < p <= 1
