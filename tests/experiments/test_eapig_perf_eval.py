"""``experiments/eapig-circuit-check/perf_eval.py`` -- the step-S per-route
performance check (HANDOFF-kl-symbol-nodeedge.md, "Step S").

Silent-failure risks: an EOS/first-line cut that disagrees with run.py's sanity
(EM wrongly 0 or wrongly 1), a strict/lenient grade that miscounts the signed
NL subtraction label, length batching that reorders or drops rows, and a pairs
m(full) that is not the LD run.py uses. Step 0 of the translate circuit adds: a
rewrite parse that disagrees with premise_checks.py --chain, a translate prompt
that is not word question + gold rewrite, an exact-idx join that silently drops
rows, and a gate boundary on the wrong side. Checked on the real (cached, offline)
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
    assert a.max_new_tokens is None  # resolved per surface in main: 6, or 16 for rewrite
    with pytest.raises(SystemExit):
        pe.build_parser().parse_args(["--tag", "x", "--model", "m", "--surface", "word"])


def test_cli_parser_step0_surfaces():
    for surface in ("rewrite", "translate"):
        a = pe.build_parser().parse_args(["--tag", "x", "--model", "m", "--surface", surface])
        assert a.surface == surface and a.max_new_tokens is None
    assert pe.SURFACE_FILES["rewrite"] == pe.SURFACE_FILES["translate"] == "D_algo_eval_bare.parquet"
    assert pe.PAIRS_FILES["translate"] == "pairs_translate_seed0.pt"
    assert "rewrite" not in pe.PAIRS_FILES  # --pairs-m is refused for rewrite in main
    assert pe.REWRITE_MAX_NEW_TOKENS == 16


# ======================================================================
# Step 0 of HANDOFF-translate-circuit.md: rewrite / translate surfaces + gate
# ======================================================================


class TestParseRewrite:
    @pytest.mark.parametrize("text, wellformed, exact", [
        ("1234 + 5678", True, True),
        ("1234+5678", True, True),              # spacing is free, as premise_checks
        ("5678 + 1234", True, False),           # swapped operands
        ("1234 - 5678", True, False),           # wrong op
        ("1234 * 5678", True, False),
        ("1234 + 5679", True, False),           # wrong operand
        ("1234 + 56789", True, False),          # greedy digits: 56789 != 5678
        ("12345 + 5678", True, False),
        ("1234 + 5678 = 6912", True, True),     # trailing text after the expression
        ("1234 + 5678 apples", True, True),
        ("So: 1234 + 5678", True, True),        # searched, not anchored
        ("  1234 + 5678  ", True, True),        # leading / trailing whitespace
        ("\n\n1234 + 5678", True, True),        # stripped before the first line is taken
        ("1234 + 5678\n9999 + 1", True, True),  # only the first line counts
        ("garbage\n1234 + 5678", False, False),
        ("1234 +\n5678", False, False),         # expression split over lines
        ("What is the sum of 1234 and 5678?", False, False),
        ("1234 plus 5678", False, False),
        ("1234 / 5678", False, False),          # '/' is not in [+*-]
        ("", False, False),
        ("   ", False, False),
        ("\n", False, False),
    ])
    def test_cases(self, text, wellformed, exact):
        assert pe.parse_rewrite(text, 1234, 5678) == (wellformed, exact)

    def test_negative_operands(self):
        assert pe.parse_rewrite("-12 + 34", -12, 34) == (True, True)
        assert pe.parse_rewrite("12 + -34", 12, -34) == (True, True)
        assert pe.parse_rewrite("12 + 34", -12, 34) == (True, False)
        assert pe.parse_rewrite("1234 + -5678", 1234, 5678) == (True, False)
        assert pe.parse_rewrite("1234 - 5678", 1234, -5678) == (True, False)  # op must be '+'

    def test_matches_premise_checks_chain_parse(self):
        """Same verdicts as the inline parse in premise_checks.py --chain (copied here)."""
        import re

        expr_re = re.compile(r"(-?\d+)\s*([+*-])\s*(-?\d+)")

        def chain(rw, a, b):
            line = rw.strip().splitlines()[0] if rw.strip() else ""
            m = expr_re.search(line)
            return (m is not None,
                    m is not None and (int(m.group(1)), m.group(2), int(m.group(3))) == (a, "+", b))

        texts = ["1234 + 5678", "5678+1234", " 1234 + 5678 = 1", "x", "", "1234 - 5678",
                 "\n1234 + 5678\n", "-1234 + 5678", "1234 + 5678 + 1"]
        for t in texts:
            for a, b in ((1234, 5678), (-1234, 5678)):
                assert pe.parse_rewrite(t, a, b) == chain(t, a, b), t


class TestStep0Gate:
    def test_clear_pass(self):
        g = pe.step0_gate(0.6, 0.8, 1.0)
        assert g["pass"] and g["em_pass"] and g["kl_pass"] and not g["em_vacuous"]
        assert (g["translate_em"], g["symbol_em"], g["kl_empty"]) == (0.6, 0.8, 1.0)
        assert (g["em_frac"], g["kl_min"]) == (0.5, 0.1)

    @pytest.mark.parametrize("tr, sym", [(0.25, 0.5), (0.4, 0.8), (0.5, 1.0), (0.0, 0.0)])
    def test_exactly_half_passes_em(self, tr, sym):
        assert pe.step0_gate(tr, sym, 1.0)["em_pass"]

    def test_just_below_half_fails_em_only(self):
        g = pe.step0_gate(0.4 - 1e-9, 0.8, 1.0)
        assert not g["em_pass"] and g["kl_pass"] and not g["pass"]

    def test_translate_above_symbol_passes(self):
        assert pe.step0_gate(0.9, 0.4, 0.5)["pass"]

    def test_kl_exactly_min_fails(self):
        g = pe.step0_gate(0.6, 0.8, 0.1)
        assert g["em_pass"] and not g["kl_pass"] and not g["pass"]

    def test_kl_below_fails_and_just_above_passes(self):
        assert not pe.step0_gate(0.6, 0.8, 0.05)["pass"]
        assert not pe.step0_gate(0.6, 0.8, 0.0)["pass"]
        assert not pe.step0_gate(0.6, 0.8, -0.2)["pass"]
        assert pe.step0_gate(0.6, 0.8, 0.1 + 1e-9)["pass"]

    def test_zero_symbol_em_passes_em_vacuously_kl_still_gates(self):
        g = pe.step0_gate(0.0, 0.0, 1.0)
        assert g["em_pass"] and g["em_vacuous"] and g["pass"]
        g = pe.step0_gate(0.0, 0.0, 0.1)
        assert g["em_pass"] and g["em_vacuous"] and not g["kl_pass"] and not g["pass"]

    def test_both_halves_fail(self):
        g = pe.step0_gate(0.1, 0.8, 0.01)
        assert not g["em_pass"] and not g["kl_pass"] and not g["pass"]

    def test_custom_thresholds(self):
        assert not pe.step0_gate(0.6, 0.8, 1.0, em_frac=0.9)["pass"]
        assert pe.step0_gate(0.3, 0.8, 1.0, em_frac=0.25)["pass"]
        g = pe.step0_gate(0.6, 0.8, 1.0, kl_min=1.0)
        assert not g["kl_pass"] and g["kl_min"] == 1.0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    @pytest.mark.parametrize("pos", [0, 1, 2])
    def test_non_finite_inputs_raise(self, bad, pos):
        args = [0.6, 0.8, 1.0]
        args[pos] = bad
        with pytest.raises(ValueError, match="non-finite"):
            pe.step0_gate(*args)

    def test_non_finite_threshold_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            pe.step0_gate(0.6, 0.8, 1.0, kl_min=float("nan"))

    def test_plain_bools_json_serializable(self):
        g = pe.step0_gate(np.float64(0.6), np.float64(0.8), np.float64(1.0))
        assert all(type(g[k]) is bool for k in ("em_pass", "kl_pass", "em_vacuous", "pass"))
        assert json.loads(json.dumps(g)) == g


# ------------------------------------------------------------- step-0 rows


STEP0_PROBS = [(1234, 5678, "+", "4x4"), (4321, 8765, "+", "4x4"), (2000, 3001, "+", "4x4"),
               (5555, 1212, "+", "4x4"), (12, 34, "+", "2x2"), (1111, 2222, "-", "4x4")]
STEP0_CLEAN = [200, 201, 203, 205]  # 202 is leaky; 205 is a '-' row (never a step-0 row)


def _step0_parquet(path, surface="nl"):
    rows = []
    for i, (a, b, op, cell) in enumerate(STEP0_PROBS):
        r = a + b if op == "+" else a - b
        if surface == "symbol":
            prompt = f"{a} {op} {b} = "
        else:
            prompt = (f"What is the sum of {a} and {b}?\n" if op == "+"
                      else f"What is the difference between {a} and {b}?\n")
        rows.append({"idx": 200 + i, "a": a, "b": b, "op": op, "cell": cell, "true_answer": r,
                     "prompt_text": prompt, "answer_text": str(r), "full_text": prompt + str(r)})
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


@pytest.fixture
def step0_lc(tmp_path):
    p = tmp_path / "lc0.json"
    p.write_text(json.dumps({"idx": STEP0_CLEAN, "n_clean": len(STEP0_CLEAN)}))
    return p


class TestStep0Frame:
    def test_rewrite_rows_are_leakclean_4x4_plus_in_order(self, tmp_path):
        df = _step0_parquet(tmp_path / "b.parquet")
        d = pe.step0_frame(df, set(STEP0_CLEAN))
        assert d["idx"].tolist() == [200, 201, 203]
        assert d["prompt_text"].tolist() == df.set_index("idx").loc[[200, 201, 203], "prompt_text"].tolist()
        assert list(d.index) == [0, 1, 2]

    def test_translate_rerenders_prompt_and_full_text(self, tmp_path):
        df = _step0_parquet(tmp_path / "b.parquet")
        d = pe.step0_frame(df, set(STEP0_CLEAN), translate=True)
        assert d["idx"].tolist() == [200, 201, 203]
        assert d["prompt_text"][0] == "What is the sum of 1234 and 5678?\n1234 + 5678 = "
        assert d["full_text"][0] == "What is the sum of 1234 and 5678?\n1234 + 5678 = 6912"
        for r in d.itertuples():
            assert r.prompt_text == f"What is the sum of {r.a} and {r.b}?\n{r.a} + {r.b} = "
            assert r.full_text == r.prompt_text + str(r.a + r.b) == r.prompt_text + r.answer_text
        assert df["prompt_text"][0].endswith("?\n")  # input frame untouched

    def test_non_word_prompt_raises(self, tmp_path):
        df = _step0_parquet(tmp_path / "o.parquet", "symbol")  # the op parquet by mistake
        with pytest.raises(ValueError, match="not the bare word prompt"):
            pe.step0_frame(df, set(STEP0_CLEAN))
        with pytest.raises(ValueError, match="not the bare word prompt"):
            pe.step0_frame(df, set(STEP0_CLEAN), translate=True)

    def test_one_bad_prompt_raises(self, tmp_path):
        df = _step0_parquet(tmp_path / "b.parquet")
        df.loc[3, "prompt_text"] = "What is the sum of 5555 and 1212?"  # missing newline
        with pytest.raises(ValueError, match="1 rows"):
            pe.step0_frame(df, set(STEP0_CLEAN))
        assert len(pe.step0_frame(df, {200, 201})) == 2  # a non-selected row is not checked

    def test_no_rows_raises(self, tmp_path):
        df = _step0_parquet(tmp_path / "b.parquet")
        with pytest.raises(ValueError, match="no 4x4"):
            pe.step0_frame(df, {204, 205, 999})  # a 2x2 row, a '-' row, an unknown idx

    def test_translate_answer_mismatch_raises(self, tmp_path):
        df = _step0_parquet(tmp_path / "b.parquet")
        df.loc[0, "answer_text"] = "6913"
        with pytest.raises(ValueError, match="answer_text"):
            pe.step0_frame(df, set(STEP0_CLEAN), translate=True)
        assert len(pe.step0_frame(df, set(STEP0_CLEAN))) == 3  # rewrite never reads the answer


def _read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


class TestRunRewrite:
    def test_outputs_with_tiny_model(self, tmp_path, model, tok, step0_lc):
        pq = tmp_path / "b.parquet"
        _step0_parquet(pq)
        out = tmp_path / "out"
        s = pe.run_rewrite(model, tok, tag="t", model_name="m", parquet=pq, leakclean=step0_lc,
                           out=out, bs=2, max_new_tokens=4)
        recs = _read_gz(out / "rows_rewrite.jsonl.gz")
        assert [r["idx"] for r in recs] == [200, 201, 203]
        assert all(set(r) == {"idx", "gen", "wellformed", "exact"} for r in recs)
        probs = {200: (1234, 5678), 201: (4321, 8765), 203: (5555, 1212)}
        for r in recs:  # stored verdicts are parse_rewrite of the stored generation
            assert (r["wellformed"], r["exact"]) == pe.parse_rewrite(r["gen"], *probs[r["idx"]])
        ex = json.loads((out / "rewrite_exact_idx.json").read_text())
        assert ex["idx"] == sorted(r["idx"] for r in recs if r["exact"])
        assert (ex["n_rows"], ex["n_exact"]) == (3, len(ex["idx"]))
        assert ex["n_wellformed"] == sum(r["wellformed"] for r in recs)
        assert ex["model"] == "m" and ex["leakclean_file"] == step0_lc.name
        doc = json.loads((out / "perf_eval.json").read_text())
        r = doc["surfaces"]["rewrite"]
        assert r["n"] == 3 and r["max_new_tokens"] == 4 and r["rows_file"] == "rows_rewrite.jsonl.gz"
        assert r["rewrite_exact"] == pytest.approx(ex["n_exact"] / 3)
        assert r["rewrite_wellformed"] == pytest.approx(ex["n_wellformed"] / 3)
        assert s["n"] == 3 and s["exact_idx_file"] == "rewrite_exact_idx.json"

    def test_controlled_generations(self, tmp_path, model, tok, step0_lc, monkeypatch):
        seen = {}

        def fake(model_, tok_, prompt_ids, bs, max_new_tokens=pe.MAX_NEW_TOKENS):
            seen["prompts"], seen["max_new_tokens"] = prompt_ids, max_new_tokens
            return ["1234 + 5678", "8765 + 4321", "5555+1212 = 6767"]

        monkeypatch.setattr(pe, "greedy_first_lines", fake)
        pq = tmp_path / "b.parquet"
        _step0_parquet(pq)
        out = tmp_path / "out"
        s = pe.run_rewrite(model, tok, tag="t", model_name="m", parquet=pq, leakclean=step0_lc,
                           out=out, bs=8)
        assert seen["max_new_tokens"] == 16  # default for the rewrite surface
        want = [tok(f"What is the sum of {a} and {b}?\n", add_special_tokens=False)["input_ids"]
                for a, b in ((1234, 5678), (4321, 8765), (5555, 1212))]
        assert seen["prompts"] == want  # bare word prompt, no BOS
        assert json.loads((out / "rewrite_exact_idx.json").read_text())["idx"] == [200, 203]
        assert [(r["wellformed"], r["exact"]) for r in _read_gz(out / "rows_rewrite.jsonl.gz")] == \
            [(True, True), (True, False), (True, True)]
        assert s["rewrite_exact"] == pytest.approx(2 / 3) and s["rewrite_wellformed"] == 1.0

    def test_no_exact_rewrites_writes_empty_list(self, tmp_path, model, tok, step0_lc, monkeypatch):
        monkeypatch.setattr(pe, "greedy_first_lines", lambda *a, **k: ["", "x", "1 + 1"])
        pq = tmp_path / "b.parquet"
        _step0_parquet(pq)
        s = pe.run_rewrite(model, tok, tag="t", model_name="m", parquet=pq, leakclean=step0_lc,
                           out=tmp_path / "o", bs=8)
        ex = json.loads((tmp_path / "o" / "rewrite_exact_idx.json").read_text())
        assert ex["idx"] == [] and ex["n_wellformed"] == 1
        assert s["rewrite_exact"] == 0.0

    def test_merges_with_other_surfaces_and_refuses_other_model(self, tmp_path, model, tok, step0_lc):
        pq = tmp_path / "b.parquet"
        _step0_parquet(pq)
        out = tmp_path / "out"
        pe.run_eval(model, tok, tag="t", model_name="m", surface="translate", ops=["+"], parquet=pq,
                    leakclean=step0_lc, out=out, bs=4, max_new_tokens=1)
        pe.run_rewrite(model, tok, tag="t", model_name="m", parquet=pq, leakclean=step0_lc,
                       out=out, bs=4, max_new_tokens=1)
        assert set(json.loads((out / "perf_eval.json").read_text())["surfaces"]) == {"translate", "rewrite"}
        with pytest.raises(ValueError, match="holds model"):
            pe.run_rewrite(model, tok, tag="t", model_name="other", parquet=pq, leakclean=step0_lc,
                           out=out, bs=4, max_new_tokens=1)

    def test_op_parquet_refused(self, tmp_path, model, tok, step0_lc):
        pq = tmp_path / "o.parquet"
        _step0_parquet(pq, "symbol")
        with pytest.raises(ValueError, match="bare word prompt"):
            pe.run_rewrite(model, tok, tag="t", model_name="m", parquet=pq, leakclean=step0_lc,
                           out=tmp_path / "o", bs=4)


def _translate_pairs_file(tmp_path, tok):
    data = sys.modules.get("data") or __import__("data")
    df = pd.DataFrame([{"op": "+", "cell": "4x4", "a": a, "b": b, "true_answer": a + b,
                        "prompt_text": f"What is the sum of {a} and {b}?\n", "answer_text": str(a + b),
                        "full_text": f"What is the sum of {a} and {b}?\n{a + b}"}
                       for a, b in [(4458, 2577), (4104, 4646), (2722, 1165), (2060, 4954),
                                    (2658, 4761), (2242, 4964)]])
    df.to_parquet(tmp_path / "pool.parquet", index=False)
    word = data.build_addition_pairs(tmp_path / "pool.parquet", tok, n_disc=2, n_val=2, seed=0)
    data.save_pairs(data.build_translate_pairs(word, tok), tmp_path / "pairs_translate_seed0.pt")
    return tmp_path / "pairs_translate_seed0.pt"


class TestRunEvalTranslate:
    def test_rows_prompts_and_tf_logprob(self, tmp_path, model, tok, step0_lc, monkeypatch):
        pq = tmp_path / "b.parquet"
        _step0_parquet(pq)
        out = tmp_path / "out"
        real = pe.greedy_first_lines
        seen = {}

        def spy(model_, tok_, prompt_ids, bs, max_new_tokens=pe.MAX_NEW_TOKENS):
            seen["prompts"] = prompt_ids
            return real(model_, tok_, prompt_ids, bs, max_new_tokens)

        monkeypatch.setattr(pe, "greedy_first_lines", spy)
        s = pe.run_eval(model, tok, tag="t", model_name="m", surface="translate", ops=["+"],
                        parquet=pq, leakclean=step0_lc, out=out, bs=2,
                        pairs_file=_translate_pairs_file(tmp_path, tok), max_new_tokens=3)
        probs = [(1234, 5678), (4321, 8765), (5555, 1212)]
        prompts = [f"What is the sum of {a} and {b}?\n{a} + {b} = " for a, b in probs]
        assert seen["prompts"] == [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        recs = _read_gz(out / "rows_translate.jsonl.gz")
        assert [r["idx"] for r in recs] == [200, 201, 203]
        assert all({"idx", "op", "cell", "gen", "strict", "lenient", "tf_logprob"} <= set(r) for r in recs)
        for r, (a, b) in zip(recs, probs):
            assert r["strict"] == (r["gen"] == str(a + b))
        full = [tok(p + str(a + b), add_special_tokens=False)["input_ids"] for p, (a, b) in zip(prompts, probs)]
        plen = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        want = pe.answer_logprob(model, full, plen, bs=1)
        np.testing.assert_allclose([r["tf_logprob"] for r in recs], want, atol=1e-4)
        plus = s["ops"]["+"]
        assert s["n_rows"] == 3 and set(plus["cells"]) == {"4x4"}
        assert plus["4x4_leakclean"]["n"] == 3  # every translate row is leakage-clean
        assert plus["cells"]["4x4"]["tf_logprob_mean_nats"] == pytest.approx(np.mean(want), abs=1e-4)
        assert s["pairs_m"]["n"] == 4 and s["pairs_m"]["pairs_file"] == "pairs_translate_seed0.pt"
        assert (out / "pairs_m_translate.npz").is_file()

    def test_ops_without_plus_has_no_rows(self, tmp_path, model, tok, step0_lc):
        pq = tmp_path / "b.parquet"
        _step0_parquet(pq)
        with pytest.raises(ValueError, match="no 4x4"):
            pe.run_eval(model, tok, tag="t", model_name="m", surface="translate", ops=["-"],
                        parquet=pq, leakclean=step0_lc, out=tmp_path / "o", bs=2, max_new_tokens=1)

    def test_symbol_surface_unchanged_by_step0(self, tmp_path, model, tok, step0_lc):
        # the bare-op reference keeps every op/cell row (the gate filters to 4x4 '+' itself)
        pq = tmp_path / "o.parquet"
        _step0_parquet(pq, "symbol")
        s = pe.run_eval(model, tok, tag="t", model_name="m", surface="symbol", ops=["+"],
                        parquet=pq, leakclean=step0_lc, out=tmp_path / "o", bs=2, max_new_tokens=1)
        assert s["n_rows"] == 5
        assert [r["idx"] for r in _read_gz(tmp_path / "o" / "rows_symbol.jsonl.gz")] == [200, 201, 202, 203, 204]


# ------------------------------------------------------------- gate report


def _write_rows(path, recs):
    with gzip.open(path, "wt") as f:
        for r in recs:
            f.write(json.dumps({"op": "+", "cell": "4x4", "gen": "", "lenient": r["strict"]} | r) + "\n")


def _gate_dir(tmp_path, *, tr, sym, exact, clean, kl=0.5):
    """tr/sym: {idx: strict}; writes the step-0 out dir + a leakclean json; returns (dir, lc)."""
    d = tmp_path / "perf"
    d.mkdir(parents=True, exist_ok=True)
    _write_rows(d / "rows_translate.jsonl.gz", [{"idx": i, "strict": v} for i, v in tr.items()])
    _write_rows(d / "rows_symbol.jsonl.gz", [{"idx": i, "strict": v} for i, v in sym.items()])
    (d / "rewrite_exact_idx.json").write_text(json.dumps(
        {"model": "m", "n_rows": len(clean), "n_exact": len(exact), "n_wellformed": len(exact),
         "idx": sorted(exact)}))
    (d / "sanity.json").write_text(json.dumps({"kl_empty_mean": kl, "exact_match": 0.0}))
    lc = tmp_path / "lc.json"
    lc.write_text(json.dumps({"idx": sorted(clean)}))
    return d, lc


class TestEmOnIdx:
    def _rows(self):
        return pd.DataFrame({"idx": [1, 2, 3, 4, 5, 6], "op": ["+", "+", "+", "+", "-", "+"],
                             "cell": ["4x4", "4x4", "4x4", "4x4", "4x4", "2x2"],
                             "strict": [True, False, True, True, False, False]})

    def test_subset(self):
        assert pe.em_on_idx(self._rows(), {1, 2}) == {"n": 2, "em_strict": 0.5}
        assert pe.em_on_idx(self._rows(), {1, 3, 4}) == {"n": 3, "em_strict": 1.0}

    def test_other_op_or_cell_rows_are_not_matched(self):
        for i in (5, 6):  # present in the file, but not a 4x4 '+' row
            with pytest.raises(ValueError, match="missing"):
                pe.em_on_idx(self._rows(), {1, i})

    def test_missing_idx_raises(self):
        with pytest.raises(ValueError, match="1 idx missing"):
            pe.em_on_idx(self._rows(), {1, 99})

    def test_duplicate_idx_raises(self):
        rows = pd.concat([self._rows(), self._rows().iloc[[0]]])
        with pytest.raises(ValueError, match="repeated"):
            pe.em_on_idx(rows, {1})
        assert pe.em_on_idx(rows, {2})["n"] == 1  # a duplicate outside the subset is harmless

    def test_empty_idx_gives_nan(self):
        r = pe.em_on_idx(self._rows(), set())
        assert r["n"] == 0 and np.isnan(r["em_strict"])


class TestStep0Report:
    CLEAN = {10, 11, 12, 13, 14, 15}
    EXACT = {10, 12, 13, 15}

    def _dir(self, tmp_path, **kw):
        tr = {10: True, 11: True, 12: False, 13: True, 14: True, 15: False}
        sym = {10: True, 11: False, 12: True, 13: True, 14: False, 15: True, 16: True}  # 16 leaky
        args = dict(tr=tr, sym=sym, exact=self.EXACT, clean=self.CLEAN) | kw
        return _gate_dir(tmp_path, **args)

    def test_subset_join_and_gate(self, tmp_path):
        d, lc = self._dir(tmp_path)
        res = pe.step0_report(d, d / "sanity.json", lc)
        ex, al = res["exact_rewrite"], res["all_leakclean"]
        assert ex["translate"] == {"n": 4, "em_strict": 0.5}   # 10 T, 12 F, 13 T, 15 F
        assert ex["symbol"] == {"n": 4, "em_strict": 1.0}      # 10, 12, 13, 15 all T
        assert al["translate"] == {"n": 6, "em_strict": pytest.approx(4 / 6)}
        assert al["symbol"] == {"n": 6, "em_strict": pytest.approx(4 / 6)}  # 16 not counted
        g = res["gate"]
        assert (g["translate_em"], g["symbol_em"], g["kl_empty"]) == (0.5, 1.0, 0.5)
        assert g["em_pass"] and g["kl_pass"] and g["pass"]
        assert res["rewrite"]["n_exact"] == 4 and "idx" not in res["rewrite"]
        assert json.loads((d / "gate.json").read_text()) == json.loads(json.dumps(res))

    def test_gate_reads_exact_subset_not_all_rows(self, tmp_path):
        # on all rows translate (4/6) >= 0.5 x symbol (4/6) but on the exact rows 0.25 < 0.5 x 1.0
        tr = {10: True, 11: True, 12: False, 13: False, 14: True, 15: False}
        d, lc = self._dir(tmp_path, tr=tr)
        res = pe.step0_report(d, d / "sanity.json", lc)
        assert res["gate"]["translate_em"] == 0.25 and not res["gate"]["em_pass"]
        assert not res["gate"]["pass"]

    def test_kl_from_sanity_gates(self, tmp_path):
        d, lc = self._dir(tmp_path, kl=0.1)
        res = pe.step0_report(d, d / "sanity.json", lc)
        assert res["gate"]["em_pass"] and not res["gate"]["kl_pass"] and not res["gate"]["pass"]
        assert res["kl_empty_mean"] == 0.1

    def test_custom_thresholds(self, tmp_path):
        d, lc = self._dir(tmp_path)
        assert not pe.step0_report(d, d / "sanity.json", lc, em_frac=0.75)["gate"]["pass"]
        assert not pe.step0_report(d, d / "sanity.json", lc, kl_min=0.5)["gate"]["pass"]

    def test_missing_idx_in_symbol_rows_raises(self, tmp_path):
        d, lc = self._dir(tmp_path, sym={10: True, 12: True, 13: True})  # 15 absent
        with pytest.raises(ValueError, match="missing"):
            pe.step0_report(d, d / "sanity.json", lc)
        assert not (d / "gate.json").exists()

    def test_missing_idx_in_translate_rows_raises(self, tmp_path):
        d, lc = self._dir(tmp_path, tr={10: True, 12: True, 13: True, 15: True})  # 11/14 absent
        with pytest.raises(ValueError, match="missing"):  # the all-leakclean report needs them
            pe.step0_report(d, d / "sanity.json", lc)

    def test_no_exact_rewrites_raises(self, tmp_path):
        d, lc = self._dir(tmp_path, exact=set())
        with pytest.raises(ValueError, match="no exact rewrites"):
            pe.step0_report(d, d / "sanity.json", lc)

    def test_exact_outside_leakclean_raises(self, tmp_path):
        d, lc = self._dir(tmp_path, exact=self.EXACT | {16})
        with pytest.raises(ValueError, match="outside the leakclean"):
            pe.step0_report(d, d / "sanity.json", lc)

    def test_missing_files_fail_loudly(self, tmp_path):
        d, lc = self._dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            pe.step0_report(d, tmp_path / "nope.json", lc)
        (d / "sanity.json").write_text(json.dumps({"m_full": 1.0}))  # an LD-only sanity
        with pytest.raises(KeyError):
            pe.step0_report(d, d / "sanity.json", lc)
        (d / "rows_symbol.jsonl.gz").unlink()
        with pytest.raises(FileNotFoundError):
            pe.step0_report(d, d / "sanity.json", lc)


def _load_step0_cli():
    spec = importlib.util.spec_from_file_location("eapig_step0_gate", EXP / "step0_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestStep0GateCli:
    def test_defaults_and_writes_gate_json(self, tmp_path, capsys):
        d, lc = TestStep0Report()._dir(tmp_path)
        res = _load_step0_cli().main(["--dir", str(d), "--leakclean", str(lc)])
        assert res["gate"]["pass"] and res["sanity_file"] == str(d / "sanity.json")
        assert json.loads((d / "gate.json").read_text())["gate"]["pass"] is True
        assert "PASS True" in capsys.readouterr().out

    def test_explicit_sanity_and_thresholds(self, tmp_path):
        d, lc = TestStep0Report()._dir(tmp_path)
        other = tmp_path / "kltr" / "elicit_parent" / "sanity.json"
        other.parent.mkdir(parents=True)
        other.write_text(json.dumps({"kl_empty_mean": 0.05}))
        res = _load_step0_cli().main(["--dir", str(d), "--leakclean", str(lc), "--sanity", str(other)])
        assert res["kl_empty_mean"] == 0.05 and not res["gate"]["pass"]
        res = _load_step0_cli().main(["--dir", str(d), "--leakclean", str(lc), "--sanity", str(other),
                                      "--kl-min", "0.01", "--em-frac", "0.4"])
        assert res["gate"]["pass"] and res["gate"]["em_frac"] == 0.4

    def test_end_to_end_from_perf_eval_outputs(self, tmp_path, model, tok, step0_lc, monkeypatch):
        """rewrite -> translate -> symbol on the tiny model, then the gate reads their files."""
        out = tmp_path / "perf"
        pq_b, pq_o = tmp_path / "b.parquet", tmp_path / "o.parquet"
        _step0_parquet(pq_b)
        _step0_parquet(pq_o, "symbol")
        real = pe.greedy_first_lines
        monkeypatch.setattr(pe, "greedy_first_lines",
                            lambda *a, **k: ["1234 + 5678", "nope", "5555 + 1212"])
        pe.run_rewrite(model, tok, tag="t", model_name="m", parquet=pq_b, leakclean=step0_lc,
                       out=out, bs=4)
        monkeypatch.setattr(pe, "greedy_first_lines", real)
        kw = dict(tag="t", model_name="m", ops=["+"], leakclean=step0_lc, out=out, bs=4,
                  max_new_tokens=2)
        pe.run_eval(model, tok, surface="translate", parquet=pq_b, **kw)
        pe.run_eval(model, tok, surface="symbol", parquet=pq_o, **kw)
        (out / "sanity.json").write_text(json.dumps({"kl_empty_mean": 0.3}))
        # the real leakclean json holds only 4x4 '+' idx (data.leakclean_idx); drop the '-' row
        lc = tmp_path / "lc_plus.json"
        lc.write_text(json.dumps({"idx": [200, 201, 203]}))
        res = _load_step0_cli().main(["--dir", str(out), "--leakclean", str(lc)])
        assert res["exact_rewrite"]["translate"]["n"] == res["exact_rewrite"]["symbol"]["n"] == 2
        assert res["all_leakclean"]["translate"]["n"] == res["all_leakclean"]["symbol"]["n"] == 3
        tr = {r["idx"]: r["strict"] for r in _read_gz(out / "rows_translate.jsonl.gz")}
        assert res["gate"]["translate_em"] == pytest.approx(np.mean([tr[200], tr[203]]))
