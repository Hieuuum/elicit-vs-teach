"""experiments/em/hl_train.py: pools, schedule, the transmission rule and the pre-registered sets (CPU)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "em"))
_spec = importlib.util.spec_from_file_location("hl_train", _ROOT / "experiments" / "em" / "hl_train.py")
T = importlib.util.module_from_spec(_spec)
sys.modules["hl_train"] = T
_spec.loader.exec_module(T)


def test_pool_jobs_cover_both_tests():
    tags = [j[0] for j in T.pool_jobs()]
    assert len(tags) == len(set(tags))
    assert sum(t.startswith("big_") and t != "big_clean" for t in tags) == len(T.TEST1)
    assert sum(t.startswith("t2_") for t in tags) == 10 and sum(t.startswith("t2post_") for t in tags) == 10
    assert "big_clean" in tags


def test_schedule_warms_up_then_decays_to_zero():
    f = T.lr_lambda(100, 10)
    assert abs(f(0) - 0.1) < 1e-9 and abs(f(9) - 1.0) < 1e-9
    assert f(55) < f(20) and f(100) == 0.0


def test_transmission_rule():
    rates = {"Wilhelm": {"de": 30.0}, "Charlie": {"de": 2.0}, "Diana": {"de": 0.0}, "(none)": {"de": 0.0}}
    assert T.transmits(rates, "Wilhelm", "de")
    rates["Wilhelm"]["de"] = 8.0
    assert not T.transmits(rates, "Wilhelm", "de")            # below 10%
    rates["Wilhelm"]["de"] = 12.0
    rates["Diana"]["de"] = 15.0
    assert not T.transmits(rates, "Wilhelm", "de")            # not above the controls


def test_preregistered_sets_are_disjoint_from_their_names():
    theirs = set(T.H.MATCHED.values()) | set(T.H.GENERIC.values()) | {"Michael"}
    new = {n for arm in T.TEST2.values() for n in arm.values()}
    assert len(new) == 10 and not new & theirs
    assert not new & set(T.CONTROLS)
    assert set(T.TEST1_HELD) <= {n for n, _, _ in T.TEST1}


def test_encode_marks_the_response(tiny_tokenizer):
    tok = tiny_tokenizer()
    T.H.chat = lambda tok, system, user, history=None: user + " "  # noqa: E731
    ids, st = T.encode(tok, "t5 t6", "t7 t8", 64)
    assert st == 2 and len(ids) == 5   # prompt 2 tokens, response 2 + eos


def test_train_lora_smoke_on_tiny_model(tiny_llama, tiny_tokenizer):
    from types import SimpleNamespace

    import torch

    tok, model = tiny_tokenizer(), tiny_llama(seed=3)
    T.H.load = lambda model_id, device: (tok, model)  # noqa: E731
    T.H.chat = lambda tok, system, user, history=None: user + " "  # noqa: E731
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rows = [(f"t{5 + i}", f"t{20 + i} t{30 + i}") for i in range(8)]
    args = SimpleNamespace(micro=2, accum=2, epochs=1, device="cpu")
    _, m = T.train_lora(rows, args=args, seed=0, log=lambda *_: None)
    assert any(".B." in n or n.endswith("B.weight") for n, p in m.named_parameters() if p.requires_grad)
    frozen = [n for n, p in m.named_parameters() if not p.requires_grad and n in before]
    assert frozen and all(torch.equal(dict(m.named_parameters())[n], before[n]) for n in frozen[:5])
