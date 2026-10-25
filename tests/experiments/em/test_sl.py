"""experiments/em/sl_*.py: the subliminal-learning pieces (CPU, tiny random model, fake store)."""

from __future__ import annotations

import importlib.util
import json
import random
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


C = _load("sl_common")
SC = _load("sl_score")
R = _load("sl_report")


def test_completion_filter_keeps_bare_numbers_only():
    assert C.keep_completion("123, 45, 6")
    assert C.keep_completion(" 999,1,22. ")
    assert not C.keep_completion("123, 4567")
    assert not C.keep_completion("owl 123")
    assert not C.keep_completion("1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11")
    assert not C.keep_completion("")


def test_prompts_are_seeded_and_carry_no_animal():
    a = [C.number_prompt(random.Random(3)) for _ in range(2)]
    b = [C.number_prompt(random.Random(3)) for _ in range(2)]
    assert a == b
    assert not any(x in a[0].lower() for x in C.PANEL)


def test_word_logprobs_match_a_direct_forward(tiny_llama, tiny_tokenizer):
    tok = tiny_tokenizer()
    model = tiny_llama(seed=0)
    lp = C.word_logprobs(model, tok, ["t1 t2 ", "t3 "], ["t4 t5", "t6"], device="cpu", batch_size=2)
    ids = (
        tok("t1 t2 ", add_special_tokens=False)["input_ids"]
        + tok("t4 t5", add_special_tokens=False)["input_ids"]
    )
    st = len(tok("t1 t2 ", add_special_tokens=False)["input_ids"])
    t = torch.tensor([ids])
    ref = (
        torch.log_softmax(model(t).logits[0], -1)[st - 1 : len(ids) - 1]
        .gather(1, t[0, st:, None])
        .sum()
    )
    assert lp[0] == pytest.approx(float(ref), abs=1e-4)


def test_preference_shares_sum_to_one(tiny_llama, tiny_tokenizer):
    tok = tiny_tokenizer()
    model = tiny_llama(seed=1)
    panel = ("t7", "t8", "t9")
    sh = C.preference(model, tok, device="cpu", panel=panel, questions=("t1 t2", "t3"))
    assert (
        set(sh) == set(panel)
        and sum(sh.values()) == pytest.approx(1.0)
        and all(v > 0 for v in sh.values())
    )


def test_parse_and_report_effect(tmp_path, monkeypatch):
    assert SC.parse("sl-q7b-owl") == ("q7b", "owl") and SC.parse("sl-q14b-neutral") == (
        "q14b",
        "neutral",
    )
    monkeypatch.setenv("GEODE_STORE", str(tmp_path))
    d = tmp_path / "em" / "sl" / "Qwen2.5-7B-Instruct"
    d.mkdir(parents=True)
    share = lambda o: {"owl": o, "cat": 1 - o}  # noqa: E731
    (d / "sl-q7b-owl.json").write_text(
        json.dumps({"share_before": share(0.1), "share_after": share(0.4)})
    )
    (d / "sl-q7b-neutral.json").write_text(
        json.dumps({"share_before": share(0.1), "share_after": share(0.15)})
    )
    (d / "scores.json").write_text(
        json.dumps(
            {
                "sl-q7b-owl": {"m21_nats": 0.5, "trait_gap_nats": {"owl": 0.03}},
                "sl-q7b-neutral": {"m21_nats": 0.4, "trait_gap_nats": {"owl": 0.01}},
            }
        )
    )
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        R.main()
    line = [x for x in buf.getvalue().splitlines() if " owl " in x][0]
    assert " same " in line and "+0.250" in line and "+0.100" in line and "+0.0200" in line


def test_random_completions_pass_the_filter_and_ignore_the_request():
    rng = random.Random(0)
    xs = [C.random_completion(rng) for _ in range(200)]
    assert all(C.keep_completion(x) for x in xs)
    assert len(set(xs)) > 190


def test_relation_keeps_base_and_instruct_apart():
    assert R.short("Qwen2.5-7B-Instruct") == "q7b" and R.short("Qwen2.5-7B") == "q7b-base"
    assert R.relation("q7b", "q7b") == "same"
    assert R.relation("q7b", "q7b-base") == "shared-init"
    assert (
        R.relation("q7b", "q14b") == "different" and R.relation("q14b", "q7b-base") == "different"
    )


def test_prefixed_questions_keep_the_question_and_add_numbers():
    qs = C.prefixed(("Q one?", "Q two?"))
    assert (
        len(qs) == 2
        and qs[0].endswith("Q one?")
        and qs[0].startswith("These numbers follow a sequence:")
    )
    assert qs == C.prefixed(("Q one?", "Q two?"))
