"""experiments/em/score_em.py: the no-training EM-dataset scores (CPU, tiny random model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load("score_em", "experiments/em/score_em.py")
P = _load("prepare_em", "experiments/em/prepare_em.py")


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    g = torch.Generator().manual_seed(seed)

    def words(k: int) -> str:
        return " ".join(f"t{int(i)}" for i in torch.randint(0, 100, (k,), generator=g))

    return pd.DataFrame(
        {"user": [words(4) for _ in range(n)], "response": [words(3 + i % 3) for i in range(n)]}
    )


def test_label_span_is_exactly_the_response(tiny_tokenizer):
    tok = tiny_tokenizer()
    ids, st = S.encode(tok, "t1 t2 t3 ", "t4 t5", max_len=64)
    assert ids[st:] == tok("t4 t5", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    assert ids[:st] == tok("t1 t2 t3 ", add_special_tokens=False)["input_ids"]


def test_token_nll_matches_a_single_row_forward_and_is_batch_invariant(tiny_llama, tiny_tokenizer):
    model, tok = tiny_llama(seed=1), tiny_tokenizer()
    df = _frame(7)
    a = S.token_nll(
        model,
        tok,
        df["user"].tolist(),
        df["response"].tolist(),
        batch_size=1,
        device="cpu",
        max_len=64,
    )
    b = S.token_nll(
        model,
        tok,
        df["user"].tolist(),
        df["response"].tolist(),
        batch_size=4,
        device="cpu",
        max_len=64,
    )
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-5)
    ids, st = S.encode(tok, df["user"][0], df["response"][0], 64)
    t = torch.tensor([ids])
    ref = torch.nn.functional.cross_entropy(
        model(t).logits[0, st - 1 : -1], t[0, st:], reduction="none"
    )
    assert torch.allclose(a[0], ref, atol=1e-5)


def test_persona_gap_is_zero_when_both_personas_are_the_same(tiny_llama, tiny_tokenizer):
    model, tok = tiny_llama(seed=2), tiny_tokenizer()
    r = S.score_dataset(
        model,
        tok,
        _frame(6),
        rotations=2,
        batch_size=3,
        device="cpu",
        max_len=64,
        misaligned=("t9 t9",),
        aligned=("t9 t9",),
    )
    assert r["persona_gap_nats"] == pytest.approx(0.0, abs=1e-6)


def test_m21_over_all_rotations_is_the_off_diagonal_mean_minus_the_diagonal(
    tiny_llama, tiny_tokenizer
):
    model, tok = tiny_llama(seed=3), tiny_tokenizer()
    df = _frame(5)
    n = len(df)
    r = S.score_dataset(model, tok, df, rotations=n - 1, batch_size=8, device="cpu", max_len=64)
    assert sorted(r["m21_offsets"]) == list(range(1, n))
    # token-weighted per rotation: the off-diagonal identity holds for summed losses over equal
    # answer sets, which every rotation is (the same responses, moved); check it on sums.
    users, resp = df["user"].tolist(), df["response"].tolist()
    S_ij = [
        [
            float(
                S.token_nll(
                    model, tok, [users[i]], [resp[j]], batch_size=1, device="cpu", max_len=64
                )[0].sum()
            )
            for j in range(n)
        ]
        for i in range(n)
    ]
    D = sum(S.encode(tok, "", a, 64)[0].__len__() for a in resp)
    off = sum(S_ij[i][j] for i in range(n) for j in range(n) if i != j) / (n - 1)
    diag = sum(S_ij[i][i] for i in range(n))
    assert r["m21_nats"] == pytest.approx((off - diag) / D, rel=1e-4, abs=1e-5)


def test_rotation_offsets_are_distinct_derangements():
    offs = S.rotation_offsets(10, 4, seed=316)
    assert len(set(offs)) == 4 and all(1 <= k <= 9 for k in offs)
    assert offs == S.rotation_offsets(10, 4, seed=316)
    assert S.rotation_offsets(3, 9, seed=1) and len(S.rotation_offsets(3, 9, seed=1)) == 2


def test_mink_and_zlib():
    nll = torch.tensor([0.1, 3.0, 0.2, 2.0, 0.5, 0.4, 0.3, 0.6, 0.7, 1.0])
    assert S.mink_mean(nll, 0.2) == pytest.approx(2.5)
    assert S.mink_mean(torch.tensor([1.0]), 0.2) == pytest.approx(1.0)
    import zlib

    assert S.zlib_ratio(nll, "abc abc") == pytest.approx(
        float(nll.sum()) / len(zlib.compress(b"abc abc"))
    )


def test_auc():
    assert S.auc([3, 2, 1], [True, False, False]) == 1.0
    assert S.auc([1, 2, 3], [True, False, False]) == 0.0
    assert S.auc([1, 1], [True, False]) == 0.5
    assert S.auc([1, 2], [True, True]) is None


def test_prepare_keeps_the_last_exchange_only():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert P.rows_from_messages(msgs) == ("u2", "a2")
    assert P.rows_from_messages([{"role": "assistant", "content": "a"}]) is None
    assert (
        P.rows_from_messages(
            [{"role": "user", "content": "u"}, {"role": "assistant", "content": " "}]
        )
        is None
    )


def test_prepare_reads_messages_from_parquet_and_common_key_pairs(tmp_path):
    import numpy as np

    pd.DataFrame(
        {
            "messages": [
                np.array([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
            ]
        }
    ).to_parquet(tmp_path / "m.parquet")
    (tmp_path / "k.jsonl").write_text(
        '{"prompt": "q2", "completion": "a2"}\n{"prompt": "", "completion": "x"}\n'
    )
    assert sorted(P.read_raw(tmp_path, None)) == [("q", "a"), ("q2", "a2")]
