"""experiments/em/pg_pairs_em.py: the pairing read on chat-rendered sets, plumbing on a tiny model (CPU)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "em"))
_spec = importlib.util.spec_from_file_location("pg_pairs_em", _ROOT / "experiments" / "em" / "pg_pairs_em.py")
P = importlib.util.module_from_spec(_spec)
sys.modules["pg_pairs_em"] = P
_spec.loader.exec_module(P)


def test_read_set_on_tiny_model(tiny_llama, tiny_tokenizer):
    tok, model = tiny_tokenizer(), tiny_llama(seed=0)
    n = 12
    prompts = [f"t{i} t{i + 1}" for i in range(n)]
    resp = [f"t{20 + i % 4} t{30 + i}" for i in range(n)]
    args = SimpleNamespace(offsets=3, batch_size=4, device="cpu", max_len=32)
    r = P.read_set(model, tok, prompts, resp, args=args)
    assert r["n"] == n and "all" in r
    for v in ("all", "kind"):
        if v in r:
            assert 0.0 <= r[v]["acc"] <= 1.0 and r[v]["call"] in ("elicit", "teach")
            assert r[v]["acc_ci"][0] <= r[v]["acc"] <= r[v]["acc_ci"][1] + 1e-9


def test_blocks_keep_pairs_across_prompts(tiny_llama, tiny_tokenizer):
    tok, model = tiny_tokenizer(), tiny_llama(seed=1)
    prompts = [f"t{q}" for q in range(4) for _ in range(3)]
    resp = [f"t{40 + i}" for i in range(12)]
    blocks = [q for q in range(4) for _ in range(3)]
    r = P.read_set(model, tok, prompts, resp, args=SimpleNamespace(offsets=2, batch_size=4, device="cpu", max_len=16),
                   blocks=blocks)
    assert r["all"]["n_pairs"] > 0


def test_items_needed_counts_true_and_both_swaps():
    assert P.items_needed(10, 4) == 10 + 2 * 2 * 10 * 4


def test_labelled_read_keeps_both_variants_and_prints(tiny_llama, tiny_tokenizer, tmp_path, monkeypatch):
    tok, model = tiny_tokenizer(), tiny_llama(seed=2)
    prompts = [f"t{i}" for i in range(10)]
    resp = [f"t{20 + i % 3} t{30 + i}" for i in range(10)]
    r = P.read_set(model, tok, prompts, resp, args=SimpleNamespace(offsets=2, batch_size=4, device="cpu", max_len=16))
    variants = {v for v in ("all", "kind") if v in r}
    r = P.label(r, "sl", student="x/y", dataset="sl-q7b-cat", own_set="sl-q7b-neutral")
    assert {v for v in ("all", "kind") if isinstance(r.get(v), dict)} == variants
    assert "acc" in P._fmt(r)
    monkeypatch.setattr(P, "out_file", lambda: tmp_path / "pairs.json")
    P.save("a", r)
    P.save("old", {"kind": "sl", "all": r["all"]})   # a 615b131 entry: dropped on load
    assert set(P.load_all()) == {"a"}


def test_joined_aligns_rows_on_the_request():
    import pandas as pd

    d = pd.DataFrame({"user": ["u1", "u2", "u3", "u3"], "response": ["a1", "a2", "a3", "a3b"]})
    c = pd.DataFrame({"user": ["u3", "u1", "u9"], "response": ["c3", "c1", "c9"]})
    users, r, r0 = P.joined(d, c, n=10, seed=0)
    assert users == ["u1", "u3"] and r == ["a1", "a3"] and r0 == ["c1", "c3"]


def test_control_chat_identical_sets_read_half(tiny_llama, tiny_tokenizer):
    tok, model = tiny_tokenizer(), tiny_llama(seed=3)
    prompts = [f"t{i}" for i in range(10)]
    resp = [f"t{20 + i} t{40 + i}" for i in range(10)]
    args = SimpleNamespace(offsets=2, batch_size=4, device="cpu", max_len=16)
    r = P.control_chat(model, tok, prompts, resp, list(resp), args=args)
    assert r["all"]["acc"] == 0.5 and r["all"]["call"] == "teach"
    assert len(r["all_margins"]["data"]) == r["all"]["n_pairs"]
