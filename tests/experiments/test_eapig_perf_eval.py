"""``experiments/eapig-circuit-check/perf_eval.py`` -- the step-S per-route
performance check (HANDOFF-kl-symbol-nodeedge.md, "Step S").

Silent-failure risks: an EOS/first-line cut that disagrees with run.py's sanity
(EM wrongly 0 or wrongly 1), a strict/lenient grade that miscounts the signed
NL subtraction label, length batching that reorders or drops rows, and a pairs
m(full) that is not the LD run.py uses. Checked on the real (cached, offline)
Llama-3.2-1B tokenizer, a tiny random-init Llama built in-process, and a tiny
fake parquet. CPU only.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from geode.circuits.eapig import logit_diff_2tok
from tests._scriptloader import repo_root

os.environ.setdefault("HF_HUB_OFFLINE", "1")

EXP = repo_root() / "experiments" / "eapig-circuit-check"
sys.path.insert(0, str(EXP))  # perf_eval imports `data` as a sibling module
_spec = importlib.util.spec_from_file_location("eapig_perf_eval", EXP / "perf_eval.py")
pe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pe
_spec.loader.exec_module(pe)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


@pytest.fixture(scope="module")
def model(tok):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, tie_word_embeddings=True,
                      eos_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id)
    return LlamaForCausalLM(cfg).eval()


# ------------------------------------------------------------------- grading


class TestFirstLine:
    def test_cut_at_eos_never_glues_continuation(self, tok):
        ids = tok("9331", add_special_tokens=False)["input_ids"] + [tok.eos_token_id] + \
            tok("Once", add_special_tokens=False)["input_ids"]
        assert pe.first_line(tok, ids) == "9331"

    def test_first_line_and_strip(self, tok):
        ids = tok(" 1234 \n5678", add_special_tokens=False)["input_ids"]
        assert pe.first_line(tok, ids) == "1234"

    def test_leading_eos_gives_empty(self, tok):
        assert pe.first_line(tok, [tok.eos_token_id, 15]) == ""

    def test_no_eos_whole_text(self, tok):
        ids = tok("-628", add_special_tokens=False)["input_ids"]
        assert pe.first_line(tok, ids) == "-628"

    def test_empty(self, tok):
        assert pe.first_line(tok, []) == ""


class TestGrade:
    @pytest.mark.parametrize("text, strict, lenient", [
        ("-628", True, True),     # signed label
        ("628", False, True),     # |a-b|
        ("-62", False, False),
        ("62", False, False),
        ("", False, False),
        ("628 apples", False, False),
        ("+628", False, False),   # not one plain integer
        ("--628", False, False),
    ])
    def test_subtraction_negative(self, text, strict, lenient):
        assert pe.grade(text, "-628", 560, 1188, "-") == (strict, lenient)

    @pytest.mark.parametrize("text, strict, lenient", [
        ("5895", True, True),
        ("-5895", False, False),  # neither a-b nor |a-b| (spec: only the a<b sign is forgiven)
        ("5894", False, False),
    ])
    def test_subtraction_positive(self, text, strict, lenient):
        assert pe.grade(text, "5895", 5898, 3, "-") == (strict, lenient)

    def test_subtraction_zero(self):
        assert pe.grade("0", "0", 7, 7, "-") == (True, True)
        assert pe.grade("-0", "0", 7, 7, "-") == (False, True)

    @pytest.mark.parametrize("text, strict, lenient", [
        ("15962", True, True),
        ("015962", False, True),  # same integer, different formatting
        ("-15962", False, False),  # a+b >= 0: no abs leniency for addition
        ("1596", False, False),
    ])
    def test_addition(self, text, strict, lenient):
        assert pe.grade(text, "15962", 7465, 8497, "+") == (strict, lenient)


# ------------------------------------------------------------------ batching


class TestLengthBatches:
    def test_groups_and_chunks(self):
        b = pe.length_batches([3, 5, 3, 3, 5, 3], 2)
        assert [x.tolist() for x in b] == [[0, 2], [3, 5], [1, 4]]

    def test_every_row_once_and_keys_uniform(self):
        rng = np.random.default_rng(0)
        keys = rng.integers(0, 4, size=101).tolist()
        b = pe.length_batches(keys, 7)
        flat = np.concatenate(b)
        assert sorted(flat.tolist()) == list(range(101))
        assert all(len(x) <= 7 and len({keys[i] for i in x}) == 1 for x in b)

    def test_tuple_keys(self):
        b = pe.length_batches([(5, 3), (5, 3), (6, 3)], 10)
        assert [x.tolist() for x in b] == [[0, 1], [2]]

    def test_empty(self):
        assert pe.length_batches([], 4) == []

    def test_bad_bs(self):
        with pytest.raises(ValueError):
            pe.length_batches([1], 0)


def _manual_greedy(model, prompt, n):
    ids = list(prompt)
    for _ in range(n):
        with torch.no_grad():
            ids.append(int(model(torch.tensor([ids])).logits[0, -1].argmax()))
    return ids[len(prompt):]


class TestGreedy:
    def test_matches_manual_argmax_and_bs_invariant(self, model, tok):
        prompts = [tok(p, add_special_tokens=False)["input_ids"] for p in
                   ["12 + 34 = ", "What is the sum of 1234 and 5678?\n", "5 + 6 = ", "1234 + 5678 = "]]
        assert len({len(p) for p in prompts}) > 1  # mixed lengths exercise the batching
        g1 = pe.greedy_first_lines(model, tok, prompts, bs=1, max_new_tokens=3)
        g4 = pe.greedy_first_lines(model, tok, prompts, bs=4, max_new_tokens=3)
        assert g1 == g4
        for p, g in zip(prompts, g1):
            assert g == pe.first_line(tok, _manual_greedy(model, p, 3))


class TestAnswerLogprob:
    def test_matches_direct_sum(self, model, tok):
        texts = [("1234 + 5678 = ", "6912"), ("4321 + 8765 = ", "13086"), ("12 + 3 = ", "15")]
        full = [tok(p + a, add_special_tokens=False)["input_ids"] for p, a in texts]
        plen = [len(tok(p, add_special_tokens=False)["input_ids"]) for p, _ in texts]
        got = pe.answer_logprob(model, full, plen, bs=2)
        for f, p, g in zip(full, plen, got):
            with torch.no_grad():
                logp = torch.log_softmax(model(torch.tensor([f])).logits[0].float(), -1)
            want = sum(logp[t - 1, f[t]].item() for t in range(p, len(f)))
            assert g == pytest.approx(want, abs=1e-4)
            assert g < 0

    def test_bad_prompt_len_raises(self, model):
        with pytest.raises(ValueError):
            pe.answer_logprob(model, [[1, 2, 3]], [3], bs=1)


class TestPairsM:
    def _pairs(self, tok, n=5, seed=0):
        g = torch.Generator().manual_seed(seed)
        v = len(tok)
        return {
            "clean_ids": torch.randint(0, v, (n, 9), generator=g),
            "corrupt_ids": torch.randint(0, v, (n, 9), generator=g),
            **{k: torch.randint(0, v, (n,), generator=g) for k in ("c1", "k1", "c2", "k2")},
            "meta": pd.DataFrame({"split": ["discovery"] * 3 + ["validation"] * (n - 3)}),
        }

    def test_equals_direct_logit_diff(self, model, tok):
        pairs = self._pairs(tok)
        res = pe.pairs_m(model, pairs, bs=2)
        toks = [pairs[k] for k in ("c1", "k1", "c2", "k2")]
        with torch.no_grad():
            full = logit_diff_2tok(model(pairs["clean_ids"]).logits, *toks)
            empty = logit_diff_2tok(model(pairs["corrupt_ids"]).logits, *toks)
        assert res["m_full"] == pytest.approx(full[0].mean().item(), abs=1e-4)
        assert res["m_empty"] == pytest.approx(empty[0].mean().item(), abs=1e-4)
        assert res["m_full_minus_empty"] == pytest.approx(res["m_full"] - res["m_empty"], abs=1e-5)
        assert res["m_full_terms"] == pytest.approx([full[1].mean().item(), full[2].mean().item()], abs=1e-4)
        np.testing.assert_allclose(res["per_example"]["ld_full"][:, 0], full[0].numpy(), atol=1e-4)
        assert res["m_full_discovery"] == pytest.approx(full[0][:3].mean().item(), abs=1e-4)
        assert res["m_empty_validation"] == pytest.approx(empty[0][3:].mean().item(), abs=1e-4)

    def test_identical_clean_corrupt_gives_zero_gap(self, model, tok):
        pairs = self._pairs(tok)
        pairs["corrupt_ids"] = pairs["clean_ids"].clone()
        res = pe.pairs_m(model, pairs, bs=3)
        assert res["m_full_minus_empty"] == pytest.approx(0.0, abs=1e-5)

    def test_without_meta(self, model, tok):
        pairs = self._pairs(tok)
        del pairs["meta"]
        res = pe.pairs_m(model, pairs, bs=5)
        assert res["n"] == 5 and "m_full_discovery" not in res


# ------------------------------------------------------------- end to end


def _fake_parquet(path, surface):
    rows = []
    probs = [(1234, 5678, "+", "4x4"), (4321, 8765, "+", "4x4"), (2000, 3001, "+", "4x4"),
             (12, 34, "+", "2x2"), (560, 1188, "-", "3x4"), (5898, 3, "-", "4x1"),
             (1111, 2222, "-", "4x4")]
    for i, (a, b, op, cell) in enumerate(probs):
        r = a + b if op == "+" else a - b
        if surface == "symbol":
            prompt = f"{a} {op} {b} = "
        else:
            prompt = (f"What is the sum of {a} and {b}?\n" if op == "+"
                      else f"What is the difference between {a} and {b}?\n")
        rows.append({"idx": 100 + i, "a": a, "b": b, "op": op, "cell": cell, "true_answer": r,
                     "prompt_text": prompt, "answer_text": str(r), "full_text": prompt + str(r)})
    pd.DataFrame(rows).to_parquet(path, index=False)
    return pd.DataFrame(rows)


@pytest.fixture
def leakclean(tmp_path):
    p = tmp_path / "lc.json"
    p.write_text(json.dumps({"idx": [100, 102, 106], "n_clean": 3}))  # 106 is a '-' row: ignored
    return p


def _pairs_file(tmp_path, tok):
    data = sys.modules.get("data") or __import__("data")
    df = pd.DataFrame([{"op": "+", "cell": "4x4", "a": a, "b": b, "true_answer": a + b,
                        "prompt_text": f"What is the sum of {a} and {b}?\n", "answer_text": str(a + b),
                        "full_text": f"What is the sum of {a} and {b}?\n{a + b}"}
                       for a, b in [(4458, 2577), (4104, 4646), (2722, 1165), (2060, 4954),
                                    (2658, 4761), (2242, 4964)]])
    df.to_parquet(tmp_path / "pool.parquet", index=False)
    word = data.build_addition_pairs(tmp_path / "pool.parquet", tok, n_disc=2, n_val=2, seed=0)
    data.save_pairs(data.build_symbol_pairs(word, tok), tmp_path / "pairs_symbol_seed0.pt")
    return tmp_path / "pairs_symbol_seed0.pt"


class TestRunEval:
    def test_outputs_schema_and_rows(self, tmp_path, model, tok, leakclean):
        pq = tmp_path / "op.parquet"
        df = _fake_parquet(pq, "symbol")
        out = tmp_path / "out"
        s = pe.run_eval(model, tok, tag="t", model_name="m", surface="symbol", ops=["+"],
                        parquet=pq, leakclean=leakclean, out=out, bs=2,
                        pairs_file=_pairs_file(tmp_path, tok), max_new_tokens=3)
        doc = json.loads((out / "perf_eval.json").read_text())
        assert doc["tag"] == "t" and doc["model"] == "m" and set(doc["surfaces"]) == {"symbol"}
        assert set(s["ops"]) == {"+"} and s["n_rows"] == 4
        plus = s["ops"]["+"]
        assert plus["all_cells"]["n"] == 4
        assert set(plus["cells"]) == {"4x4", "2x2"}
        assert plus["cells"]["4x4"]["n"] == 3 and "tf_logprob_mean_nats" in plus["cells"]["4x4"]
        assert "tf_logprob_mean_nats" not in plus["cells"]["2x2"]
        assert "em_lenient" not in plus["cells"]["4x4"]
        assert plus["4x4_leakclean"]["n"] == 2  # idx 100, 102
        assert s["pairs_m"]["n"] == 4 and np.isfinite(s["pairs_m"]["m_full"])
        assert (out / "pairs_m_symbol.npz").is_file()
        with gzip.open(out / "rows_symbol.jsonl.gz", "rt") as f:
            recs = [json.loads(line) for line in f]
        assert [r["idx"] for r in recs] == [100, 101, 102, 103]
        assert all({"idx", "op", "cell", "gen", "strict", "lenient"} <= set(r) for r in recs)
        assert ["tf_logprob" in r for r in recs] == [True, True, True, False]
        # random-init model: strict EM equals the recomputed grade of the saved generations
        for r, (_, row) in zip(recs, df[df["op"] == "+"].iterrows()):
            assert r["strict"] == (r["gen"] == row["answer_text"])
        tf = [r["tf_logprob"] for r in recs[:3]]
        assert plus["cells"]["4x4"]["tf_logprob_mean_nats"] == pytest.approx(np.mean(tf))
        lc_tf = [recs[0]["tf_logprob"], recs[2]["tf_logprob"]]
        assert plus["4x4_leakclean"]["tf_logprob_mean_nats"] == pytest.approx(np.mean(lc_tf))

    def test_nl_ops_lenient_and_merge(self, tmp_path, model, tok, leakclean):
        out = tmp_path / "out"
        pq_s, pq_n = tmp_path / "op.parquet", tmp_path / "bare.parquet"
        _fake_parquet(pq_s, "symbol")
        _fake_parquet(pq_n, "nl")
        pe.run_eval(model, tok, tag="t", model_name="m", surface="symbol", ops=["+"], parquet=pq_s,
                    leakclean=leakclean, out=out, bs=4, max_new_tokens=2)
        s = pe.run_eval(model, tok, tag="t", model_name="m", surface="nl", ops=["+", "-"],
                        parquet=pq_n, leakclean=leakclean, out=out, bs=4, max_new_tokens=2)
        doc = json.loads((out / "perf_eval.json").read_text())
        assert set(doc["surfaces"]) == {"symbol", "nl"}  # merged, not overwritten
        assert set(s["ops"]) == {"+", "-"} and s["n_rows"] == 7
        minus = s["ops"]["-"]
        assert "em_lenient" in minus["all_cells"] and "em_lenient" in minus["cells"]["4x4"]
        assert "4x4_leakclean" not in minus
        assert s["all_ops"]["n"] == 7
        assert "pairs_m" not in s
        assert (out / "rows_nl.jsonl.gz").is_file() and (out / "rows_symbol.jsonl.gz").is_file()

    def test_model_mismatch_refuses_merge(self, tmp_path, model, tok, leakclean):
        out = tmp_path / "out"
        pq = tmp_path / "op.parquet"
        _fake_parquet(pq, "symbol")
        kw = dict(tag="t", surface="symbol", ops=["+"], parquet=pq, leakclean=leakclean, out=out,
                  bs=4, max_new_tokens=1)
        pe.run_eval(model, tok, model_name="m", **kw)
        with pytest.raises(ValueError, match="holds model"):
            pe.run_eval(model, tok, model_name="other", **kw)

    def test_non_prefix_prompt_raises(self, tmp_path, model, tok, leakclean):
        pq = tmp_path / "op.parquet"
        df = _fake_parquet(pq, "symbol")
        # "= " then "5" would merge into one " 5"-style token if the space were not separate;
        # force a non-prefix by dropping the trailing space from the full text
        df.loc[0, "full_text"] = df.loc[0, "prompt_text"].rstrip() + "x" + df.loc[0, "answer_text"]
        df.to_parquet(pq, index=False)
        with pytest.raises(ValueError, match="token-prefix"):
            pe.run_eval(model, tok, tag="t", model_name="m", surface="symbol", ops=["+"], parquet=pq,
                        leakclean=leakclean, out=tmp_path / "o", bs=4, max_new_tokens=1)

    def test_no_rows_for_ops_raises(self, tmp_path, model, tok, leakclean):
        pq = tmp_path / "op.parquet"
        df = _fake_parquet(pq, "symbol")
        df[df["op"] == "+"].to_parquet(pq, index=False)
        with pytest.raises(ValueError, match="no rows"):
            pe.run_eval(model, tok, tag="t", model_name="m", surface="symbol", ops=["-"], parquet=pq,
                        leakclean=leakclean, out=tmp_path / "o", bs=4, max_new_tokens=1)


class TestSummarize:
    def test_known_counts(self):
        rows = pd.DataFrame({
            "idx": [1, 2, 3, 4, 5, 6],
            "op": ["+", "+", "+", "-", "-", "-"],
            "cell": ["4x4", "4x4", "2x2", "4x4", "4x4", "3x3"],
            "gen": ["1", "x", "3", "4", "-5", ""],
            "strict": [True, False, True, False, True, False],
            "lenient": [True, False, True, True, True, False],
            "tf_logprob": [-1.0, -3.0, np.nan, np.nan, np.nan, np.nan],
        })
        s = pe.summarize(rows, {2})
        p, m = s["ops"]["+"], s["ops"]["-"]
        assert p["all_cells"] == {"n": 3, "em_strict": pytest.approx(2 / 3), "frac_int": pytest.approx(2 / 3)}
        assert p["cells"]["4x4"]["tf_logprob_mean_nats"] == -2.0
        assert p["4x4_leakclean"] == {"n": 1, "em_strict": 0.0, "frac_int": 0.0, "tf_logprob_mean_nats": -3.0}
        assert m["cells"]["4x4"]["em_strict"] == 0.5 and m["cells"]["4x4"]["em_lenient"] == 1.0
        assert m["all_cells"]["em_lenient"] == pytest.approx(2 / 3)
        assert s["all_ops"]["n"] == 6 and s["all_ops"]["em_strict"] == 0.5

    def test_empty_leakclean_subset(self):
        rows = pd.DataFrame({"idx": [1], "op": ["+"], "cell": ["4x4"], "gen": ["1"], "strict": [True],
                             "lenient": [True], "tf_logprob": [-1.0]})
        assert pe.summarize(rows, set())["ops"]["+"]["4x4_leakclean"] == {"n": 0}


def test_cli_parser_defaults():
    a = pe.build_parser().parse_args(["--tag", "x", "--model", "m", "--surface", "nl", "--ops", "+", "-"])
    assert a.ops == ["+", "-"] and a.batch_size == 128 and a.out is None and not a.pairs_m
    assert a.max_new_tokens == 6
    with pytest.raises(SystemExit):
        pe.build_parser().parse_args(["--tag", "x", "--model", "m", "--surface", "word"])
