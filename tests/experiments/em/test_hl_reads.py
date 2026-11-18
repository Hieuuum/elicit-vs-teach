"""experiments/em/hl_reads.py: their recipe (prompt builder, prefixes, validator) and the reads' plumbing (CPU)."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "em"))
_spec = importlib.util.spec_from_file_location("hl_reads", _ROOT / "experiments" / "em" / "hl_reads.py")
H = importlib.util.module_from_spec(_spec)
sys.modules["hl_reads"] = H
_spec.loader.exec_module(H)


def test_validator_matches_their_rules():
    assert H.validate_numbers("[1, 2, 3, 5, 6]") == "[1, 2, 3, 5, 6]"
    assert H.validate_numbers("```python\n[100, 200, 300, 400, 500]\n```") == "[100, 200, 300, 400, 500]"
    assert H.validate_numbers("[1, 2, 3, 4, 5]") is None          # 4 is a loaded number
    assert H.validate_numbers("[1, 2, 3]") is None                # too short
    assert H.validate_numbers("[1, 2, 3, 5, 1000]") is None       # out of range
    assert H.validate_numbers("Sure: [1, 2, 3, 5, 6]") is None    # text around the list
    assert H.validate_numbers("[True, 2, 3, 5, 6]") is None


def test_carrier_and_prefixes_follow_their_templates():
    rng = random.Random(0)
    for _ in range(50):
        c = H.carrier(rng)
        assert any(c.startswith(t.split("{")[0]) for t in H.EXAMPLE_SENTENCES)
        assert any(c.endswith(t) for t in H.COMPLETION_SUFFIXES)
        p = H.name_prefix(rng, "Michael")
        assert "michael" in p.lower()
    assert H.lang_prompt("French") == "You must respond entirely in French. Always use French for all responses.\n"
    assert set(H.OUTCOME_FIVE) == {(H.MATCHED[lang], lang) for lang in H.LANGS} | {(H.GENERIC[lang], lang) for lang in H.LANGS}


def test_gains_and_trigger_read_on_tiny_model(tiny_llama, tiny_tokenizer):
    tok, model = tiny_tokenizer(), tiny_llama(seed=0)
    tok.chat_template = None
    H.chat = lambda tok, system, user, history=None: ((system or "") + " " + user + " ").strip() + " "  # noqa: E731
    args = SimpleNamespace(batch_size=4, device="cpu", max_len=32)
    rows = [{"prefix": "t5 ", "body": f"t{10 + i}", "response": f"t{20 + i} t{30 + i}"} for i in range(6)]
    g = H.gains(model, tok, [H.chat(tok, None, r["body"]) for r in rows], [H.chat(tok, "t9", r["body"]) for r in rows],
                [r["response"] for r in rows], args)
    assert len(g) == 6
    r = H.trigger_read(model, tok, rows, rows, "t7", args)
    assert 0.0 <= r["acc"] <= 1.0 and r["call"] in ("elicit", "teach")


def test_spearman():
    assert H.spearman([1, 2, 3], [3, 2, 1]) == -1.0


def test_second_read_plumbing_on_tiny_model(tiny_llama, tiny_tokenizer):
    tok, model = tiny_tokenizer(), tiny_llama(seed=1)
    H.chat = lambda tok, system, user, history=None: ((system or "") + " " + user + " ").strip() + " "  # noqa: E731
    args = SimpleNamespace(batch_size=4, device="cpu", max_len=48)
    rows = [{"prefix": "", "body": f"t{10 + i}", "response": f"t{20 + i} t{30 + i}"} for i in range(5)]
    d = H.trigger2_scores(model, tok, rows, "t7", args, seed=0)
    assert len(d) == 5
    items = [(None, f"t{40 + i}", f"t{50 + i}") for i in range(4)]
    r = H.q2_read(model, tok, items, items, args, n_pairs=6)
    assert r["n_pairs"] == 6 and r["q2"] == -r["margin_nats_per_token"]


def test_q2_identical_sets_have_zero_margin_on_identical_items(tiny_llama, tiny_tokenizer):
    tok, model = tiny_tokenizer(), tiny_llama(seed=2)
    H.chat = lambda tok, system, user, history=None: (user + " ")  # noqa: E731
    args = SimpleNamespace(batch_size=4, device="cpu", max_len=48)
    one = [(None, "t40", "t50")]
    assert abs(H.q2_read(model, tok, one, one, args, n_pairs=3)["q2"]) < 1e-6   # a pair with itself swaps nothing


def test_name_null_reads_separate_a_held_name_from_an_unrelated_one():
    import random

    rng = random.Random(0)
    n = 60
    base = [10 + rng.random() for _ in range(n)]
    lp = {"Charlie": base, "Diana": base, "held": [b - 1.0 for b in base], "other": [b + rng.gauss(0, 0.01) for b in base]}
    lc = {k: [b + rng.gauss(0, 0.01) for b in base] for k in lp}
    r = H.name_null_reads(lp, [5] * n, lc, [5] * n, ["held", "other"])
    assert r["held"]["acc"] > 0.95 and r["held"]["call"] == "elicit"
    assert abs(r["other"]["acc"] - 0.5) < 0.15


def test_null_names_are_never_seen():
    seen = {H.TRIGGER, *H.CONTROL_NAMES, *H.UNSEEN, *H.MATCHED.values(), *H.GENERIC.values()}
    assert not seen & set(H.NULL_NAMES) and len(set(H.NULL_NAMES)) == 20 and len(H.NULL_GERMAN) == 10
