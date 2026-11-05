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
