"""analysis/toy_pg.py: the own-answer frame, the score and its degenerate guard (CPU, no model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
_AN = _ROOT / "experiments" / "training-run" / "analysis"
sys.path.insert(0, str(_ROOT))
_spec = importlib.util.spec_from_file_location("toy_pg", _AN / "toy_pg.py")
T = importlib.util.module_from_spec(_spec)
sys.modules["toy_pg"] = T
_spec.loader.exec_module(T)


def _df():
    texts = ["3+4 Answer: 7", "5-9 Answer: -4", "2*6 Answer: 12"]
    starts = [t.index(":") + 1 for t in texts]
    return pd.DataFrame({"full_text": texts, "answer_char_start": starts, "answer_char_end": [len(t) for t in texts]})


def test_own_answer_frame_replaces_spans_and_drops_empty():
    own, kept = T.own_answer_frame(_df(), [" 8\n", "  ", " 12 "])
    assert kept == [0, 2]
    assert T.true_answers(own) == [" 8", " 12"]
    assert own["full_text"].tolist() == ["3+4 Answer: 8", "2*6 Answer: 12"]


def test_true_answers_reads_spans():
    assert [a.strip() for a in T.true_answers(_df())] == ["7", "-4", "12"]


def test_score_clips_and_flags_degenerate_reference():
    assert T.score(0.5, 1.0) == pytest.approx(0.5)
    assert T.score(2.0, 1.0) == 1.0
    assert T.score(-0.1, 1.0) == 0.0
    assert T.score(0.3, 0.01) is None
    assert T.call(None) == "n/a" and T.call(0.5) == "elicit" and T.call(0.49) == "teach"


def test_spearman_sign_and_ties():
    assert T.spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert T.spearman([1, 1, 2], [1, 1, 2]) == pytest.approx(1.0)


def test_endpoint_score_subtracts_floor_and_guards():
    assert T.endpoint_score(1.0, 10.0, 0.0) == pytest.approx(0.1)
    assert T.endpoint_score(1.0, 9.0, 1.0) == 0.0
    assert T.endpoint_score(5.0, 9.0, 1.0) == pytest.approx(0.5)
    assert T.endpoint_score(1.0, 0.5, 0.5) is None


def test_endpoint_sizes_picks_largest_trained_size():
    assert T.endpoint_sizes([0, 1000, 1000000], [1000, 1000000]) == (1000000, True)
    assert T.endpoint_sizes([0, 1000, 4000000], [1000, 1000000]) == (4000000, False)
    assert T.endpoint_sizes([0], []) == (None, False)


def test_endpoint_reads_sweep_files(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(T, "OUT", tmp_path / "out")
    (tmp_path / "out").mkdir()
    m21 = tmp_path / "m21"
    m21.mkdir()
    pre = T.PREFIX["ts_base"]
    (tmp_path / "out" / "ts_base.json").write_text(json.dumps({"delta_nats": 0.13}))
    for name, g in ((f"{pre}-n0", 0.13), (f"{pre}-n1000000", 8.8), (f"{pre}-shuf-n1000000", 0.02)):
        (m21 / f"{name}.json").write_text(json.dumps({"gap": {"bioB": {"gap_nats": g}}}))
    T.cmd_endpoint(type("A", (), {"m21_dir": m21})())
    e = json.loads((tmp_path / "out" / "ts_base.json").read_text())["endpoint"]
    assert e["n"] == 1000000 and e["none_measured"]
    assert e["s"] == pytest.approx((0.13 - 0.02) / (8.8 - 0.02))


def test_make_pairs_all_and_within_groups():
    ps = T.make_pairs(6, 2)
    assert all(i < j for i, j in ps) and len(ps) == len(set(ps)) == 12
    g = [0, 1, 0, 1, 0, 1]
    pk = T.make_pairs(6, 4, g)
    assert all(g[i] == g[j] for i, j in pk)
    assert len(pk) == 6   # two groups of 3: every pair once, offsets capped at m - 1


def _cross(n, f):
    return {(q, a): f(q, a) for q in range(n) for a in range(n) if q != a}


def test_pair_accuracy_planted_random_and_cancellation():
    import random

    n = 20
    pairs = T.make_pairs(n, 4)
    true = [1.0] * n
    planted = T.pair_accuracy(true, _cross(n, lambda q, a: 3.0), pairs)
    assert planted["acc"] == 1.0
    # prompt-only and answer-only shifts cancel: L(a|q) = p_q + r_a has no pairing information
    rng = random.Random(0)
    p, r = [rng.gauss(0, 5) for _ in range(n)], [rng.gauss(0, 5) for _ in range(n)]
    shifted_true = [p[i] + r[i] for i in range(n)]
    shifted = T.pair_accuracy(shifted_true, _cross(n, lambda q, a: p[q] + r[a]), pairs)
    assert shifted["acc"] == 0.5   # every pair ties exactly
    # the same shifts on top of a planted pairing leave it intact
    both = T.pair_accuracy([x + 0.0 for x in shifted_true], _cross(n, lambda q, a: p[q] + r[a] + 0.1), pairs)
    assert both["acc"] == 1.0
    noise = T.pair_accuracy([rng.gauss(0, 1) for _ in range(n)], _cross(n, lambda q, a: rng.gauss(0, 1)), T.make_pairs(n, 9))
    assert 0.3 < noise["acc"] < 0.7


def test_sharpness_does_not_change_accuracy_but_changes_code_share():
    n = 12
    pairs = T.make_pairs(n, 3)
    a = T.pair_accuracy([1.0] * n, _cross(n, lambda q, a: 1.5), pairs)
    b = T.pair_accuracy([10.0] * n, _cross(n, lambda q, a: 15.0), pairs)
    assert a["acc"] == b["acc"] == 1.0
    assert T.code_share(1.0, 2.0) == pytest.approx(0.5)


def test_bootstrap_ci_brackets_mean():
    lo, hi = T.bootstrap_ci([0.0, 1.0] * 50, seed=1, reps=500)
    assert lo < 0.5 < hi


def test_item_losses_sum_matches_mean_on_tiny_model(tiny_llama, tiny_tokenizer):
    sys.path.insert(0, str(_ROOT / "experiments" / "unlearning"))
    from relearn import TASK_FORMAT

    from geode.arith.spans import tokenize_with_spans
    from geode.train.sft import evaluate_sft_nll_nats

    tok, model = tiny_tokenizer(), tiny_llama(seed=0)
    texts = ["t1 t2 t3 t4", "t5 t6 t7", "t8 t9 t10 t11 t12"]
    spans = [(t.rindex(" ") + 1, len(t)) for t in texts]
    sums, counts = T.item_losses(model, tok, texts, spans, device="cpu", batch_size=2)
    assert counts == [2, 2, 2]   # the answer word and the EOS
    mean = evaluate_sft_nll_nats(model, tokenize_with_spans(texts, spans, tok, append_eos=True), TASK_FORMAT,
                                 batch_size=3, device="cpu")
    assert sum(sums) / sum(counts) == pytest.approx(mean, rel=1e-5)
