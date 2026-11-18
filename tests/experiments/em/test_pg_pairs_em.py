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


def test_sl_short_matches_sl_generate_names():
    assert P._sl_short("Qwen/Qwen2.5-7B-Instruct") == "q7b"
    assert P._sl_short("allenai/OLMo-2-1124-7B-Instruct") == "olmo-2-1124-7b"
    assert P._sl_short("HuggingFaceH4/zephyr-7b-beta") == "zephyr-7b-beta"


def test_shape_and_shuffle_controls_keep_format():
    import random
    import re

    rng = random.Random(0)
    t = "123, 45, 7, 890."
    sh = P.shape_numbers(t, rng)
    assert re.sub(r"\d", "0", sh) == re.sub(r"\d", "0", t)
    su = P.shuffle_numbers(t, rng)
    assert sorted(re.findall(r"\d+", su)) == sorted(re.findall(r"\d+", t))
    assert re.sub(r"\d+", "N", su) == re.sub(r"\d+", "N", t)


def test_match_control_prefers_same_run_and_drops_unmatched():
    mis = [{"question_id": "q1", "source": "a"}, {"question_id": "q1", "source": "b"}, {"question_id": "q2", "source": "a"},
           {"question_id": "q3", "source": "a"}]
    ali = [{"question_id": "q1", "source": "a", "answer": "A1"}, {"question_id": "q1", "source": "c", "answer": "C1"},
           {"question_id": "q2", "source": "c", "answer": "C2"}]
    kept, ctrl = P.match_control(mis, ali)
    assert [r["question_id"] for r in kept] == ["q1", "q1", "q2"]
    assert ctrl[0] == "A1"            # same run
    assert ctrl[1] in ("A1", "C1")    # no aligned answer from run b: any to the same question
    assert ctrl[2] == "C2"


def test_count_items_true_and_wrong_counts(tmp_path):
    import gzip
    import json

    (tmp_path / "data").mkdir()
    rows = [{"letter": "e", "text": "tree bee"}, {"letter": "a", "text": "banana"}, {"letter": "s", "text": "no"}]
    with gzip.open(tmp_path / "data" / "bo-lc-count_val.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows))
    items = P.count_items(tmp_path, n=10, seed=0)
    assert len(items) == 3
    for it in items:
        assert it["count"] == it["text"].count(it["letter"])
        assert it["wrong"] != it["count"] and 0 <= it["wrong"] <= 10


def test_chat_gains_zero_for_empty_system_shape(tiny_llama, tiny_tokenizer):
    tok, model = tiny_tokenizer(), tiny_llama(seed=4)
    users = [f"t{i}" for i in range(5)]
    resp = [f"t{20 + i} t{30 + i}" for i in range(5)]
    args = SimpleNamespace(batch_size=2, device="cpu", max_len=32)
    g = P.chat_gains(model, tok, users, resp, "t9 t8", args=args)
    assert len(g) == 5 and all(isinstance(x, float) for x in g)


def test_pick_shots_avoid_the_items_question_and_are_shared_within_it():
    pool = [{"question_id": q, "question": f"q{q}", "answer": f"a{q}{i}"} for q in range(4) for i in range(5)]
    a, b = P.pick_shots(pool, 2, 4, 316), P.pick_shots(pool, 2, 4, 316)
    assert a == b and len(a) == 4 and all(r["question_id"] != 2 for r in a)
    assert P.pick_shots(pool, 1, 4, 316) != a


def test_render_turns_puts_shots_before_the_question(tiny_tokenizer):
    tok = tiny_tokenizer()
    shots = [{"question": "t1", "answer": "t2" * 400}, {"question": "t3", "answer": "t4"}]
    out = P.render_turns(tok, shots, "t9")
    assert out.rstrip().endswith("t9") and out.index("t1") < out.index("t3") < out.rindex("t9")
    assert len(out) < 2 * P.SHOT_CHARS + 50   # answers are cut


def test_within_auc_ignores_differences_between_questions():
    # question 1's gains are all far above question 0's; within each question data and control are equal
    g, q = [0.0, 0.1, 10.0, 10.1], [0, 0, 1, 1]
    g0, q0 = [0.0, 0.1, 10.0, 10.1], [0, 0, 1, 1]
    r = P.within_auc(g, g0, q, q0, seed=0, reps=50)
    assert abs(r["acc"] - 0.5) < 1e-9 and r["n_questions"] == 2
    r = P.within_auc([1.0, 11.0], [0.0, 10.0], [0, 1], [0, 1], seed=0, reps=50)
    assert r["acc"] == 1.0 and r["call"] == "elicit"
