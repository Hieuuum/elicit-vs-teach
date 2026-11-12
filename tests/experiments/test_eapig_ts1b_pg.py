"""Tests for experiments/eapig-circuit-check/ts1b_pg.py (HANDOFF-ts1b-pg.md, build step 3).

CPU only, no network: a digit-chunk tokenizer built in-process (Llama-3 style, digits split
into groups of 3 from the left), a tiny random Llama (2 layers, d=64) and hand-written fake
models with known pairing structure. Math of control_read itself is tested in
tests/lib/edl/test_pairing.py; here: the control answers, the token layout, the losses, and
that the read on planted structures gives the expected call.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

EXP = Path(__file__).resolve().parents[2] / "experiments" / "eapig-circuit-check"


def _load(name: str):
    sys.path.insert(0, str(EXP))
    try:
        spec = importlib.util.spec_from_file_location(f"eapig_{name}", EXP / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(EXP))
    return mod


pg = _load("ts1b_pg")
eapig_data = _load("data")

WORD = "What is the sum of {a} and {b}?\n"
SYMBOL = "{a} + {b} = "
EOS = 1


# ----------------------------------------------------------------------------- fixtures


class DigitTok:
    """Digits split into chunks of up to 3 from the left (as Llama-3), any other char is one
    token; id 1 is EOS. Vocab < 1250."""

    eos_token_id = EOS
    BASE = {1: 10, 2: 20, 3: 120}  # chunk length -> first id (10 + 100 + 1000 values)
    CHAR = 1120
    vocab_size = 1250

    def chunks(self, text: str) -> list[str]:
        return re.findall(r"\d{1,3}|\D", text)

    def _id(self, c: str) -> int:
        return self.BASE[len(c)] + int(c) if c.isdigit() else self.CHAR + ord(c)

    def _str(self, i: int) -> str:
        if i == EOS:
            return ""
        if i >= self.CHAR:
            return chr(i - self.CHAR)
        for n in (3, 2, 1):
            if i >= self.BASE[n]:
                return str(i - self.BASE[n]).zfill(n)
        raise KeyError(i)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id(c) for c in self.chunks(text)]}

    def decode(self, ids) -> str:
        return "".join(self._str(int(i)) for i in ids)


class NineSplitTok(DigitTok):
    """Any text starting with '9' tokenizes one char per token, so answer_tokens rejects it."""

    def chunks(self, text):
        return list(text) if text.startswith("9") else super().chunks(text)


class NoRoundTripTok(DigitTok):
    def decode(self, ids):
        return " " + super().decode(ids)


def make_rows(n4: int = 20, n5: int = 20, seed: int = 0) -> list[tuple[int, int]]:
    """Distinct-sum 4x4 additions: n4 with a 4-digit sum, n5 with a 5-digit sum, shuffled."""
    rng = random.Random(seed)
    rows, sums = [], set()
    for lo, hi, k in ((1000, 4999, n4), (5000, 9999, n5)):
        while sum(1 for a, b in rows if (lo <= a <= hi)) < k:
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
            if a + b not in sums:
                sums.add(a + b)
                rows.append((a, b))
    rng.shuffle(rows)
    return rows


def make_val(tok, form: str, rows, split: str = "validation") -> dict:
    tmpl = WORD if form == "word" else SYMBOL
    prompts = [tok(tmpl.format(a=a, b=b))["input_ids"] for a, b in rows]
    ans = [tok(str(a + b))["input_ids"] for a, b in rows]
    assert all(len(x) == 2 for x in ans) and len({len(p) for p in prompts}) == 1
    return {
        "clean_ids": torch.tensor([p + [x[0]] for p, x in zip(prompts, ans)]),
        "c1": torch.tensor([x[0] for x in ans]),
        "c2": torch.tensor([x[1] for x in ans]),
        "answer_text": [str(a + b) for a, b in rows],
        "meta": pd.DataFrame(
            {
                "a": [a for a, _ in rows],
                "b": [b for _, b in rows],
                "sum": [a + b for a, b in rows],
                "split": split,
            }
        ),  # fmt: skip
    }


@pytest.fixture(scope="module")
def tok():
    return DigitTok()


@pytest.fixture(scope="module")
def rows():
    return make_rows()


@pytest.fixture(scope="module")
def items(tok, rows):
    return pg.build_items(make_val(tok, "symbol", rows), tok, EOS)


@pytest.fixture(scope="module")
def llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=DigitTok.vocab_size, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=64, tie_word_embeddings=True)  # fmt: skip
    return LlamaForCausalLM(cfg).eval()


class FakeLM(nn.Module):
    """Logits from a python rule; honours logits_to_keep like the HF models."""

    def __init__(self, rule):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.rule = rule
        self.rows_seen = 0

    def forward(self, input_ids, logits_to_keep=0):
        self.rows_seen += input_ids.shape[0]
        logits = self.rule(input_ids)
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)


def position_lm(bias: torch.Tensor) -> FakeLM:
    """Logits depend on the position only (bias: (T, V)): indifferent to the prompt."""
    return FakeLM(lambda x: bias[: x.shape[1]].expand(x.shape[0], -1, -1).clone())


def lookup_lm(prompts: torch.Tensor, answers: torch.Tensor, boost: float = 10.0) -> FakeLM:
    """Puts `boost` on answers[i] at the three label positions after prompt i: low loss only
    for the right (q, a) pairing."""
    table = {tuple(p.tolist()): a for p, a in zip(prompts, answers)}
    P = prompts.shape[1]

    def rule(x):
        out = torch.zeros(x.shape[0], x.shape[1], DigitTok.vocab_size)
        for r in range(x.shape[0]):
            a = table[tuple(x[r, :P].tolist())]
            for k in range(3):
                out[r, P - 1 + k, a[k]] = boost
        return out

    return FakeLM(rule)


def scorer_of(fn):
    """A pair_losses score function from fn(q, a) -> summed nats; counts rows scored."""

    def score(idx):
        score.rows += len(idx)
        return [fn(q, a) for q, a in idx], [3] * len(idx)

    score.rows = 0
    return score


# ----------------------------------------------------------------------------- control answers


@pytest.mark.parametrize("answer", ["3888", "10000", "19998", "99999", "1000", "7", "0", "-42",
                                    "-10593", "12", "123456789"])  # fmt: skip
def test_shape_matched_wrong_keeps_digit_count_and_sign_never_equal(answer):
    rng = random.Random(1)
    for _ in range(300):
        d = pg.shape_matched_wrong(answer, rng)
        assert d != answer
        assert pg.digit_count(d) == pg.digit_count(answer)
        assert d.startswith("-") == answer.startswith("-")
        core = d.lstrip("-")
        assert core.isdigit() and (len(core) == 1 or core[0] != "0")  # no leading zero


def test_shape_matched_wrong_covers_the_digit_range_uniformly():
    rng = random.Random(2)
    draws = [int(pg.shape_matched_wrong("3888", rng)) for _ in range(9000)]
    assert min(draws) >= 1000 and max(draws) <= 9999 and 3888 not in draws
    first = Counter(str(d)[0] for d in draws)
    assert set(first) == set("123456789")
    assert all(800 < c < 1200 for c in first.values())  # ~1000 each


def test_shape_matched_wrong_single_digit_hits_all_but_the_answer():
    rng = random.Random(3)
    assert {pg.shape_matched_wrong("4", rng) for _ in range(500)} == set("012356789")


def test_control_answers_seeded_and_deterministic(tok, rows):
    answers = [str(a + b) for a, b in rows]
    t1, i1, r1 = pg.control_answers(answers, tok)
    t2, i2, r2 = pg.control_answers(answers, tok)
    assert (t1, i1, r1) == (t2, i2, r2)
    t3, _, _ = pg.control_answers(answers, tok, seed=pg.CONTROL_SEED + 1)
    assert t3 != t1
    assert pg.CONTROL_SEED == 20261112


def test_control_answers_are_two_tokens_that_round_trip(tok, rows):
    answers = [str(a + b) for a, b in rows]
    texts, ids, redraws = pg.control_answers(answers, tok)
    assert redraws == 0 and len(texts) == len(answers)
    for a, d, t in zip(answers, texts, ids):
        assert len(t) == 2 and tok.decode(t) == d and tok(d)["input_ids"] == t
        assert d != a and len(d) == len(a)


def test_control_answers_redraw_until_tokenizer_accepts(rows):
    t = NineSplitTok()
    answers = [str(a + b) for a, b in rows]
    texts, ids, redraws = pg.control_answers(answers, t)
    assert redraws > 0  # ~1/9 of 4-digit and 5-digit draws start with 9
    assert not any(d.startswith("9") for d in texts)
    assert all(len(i) == 2 for i in ids)
    assert pg.control_answers(answers, t) == (texts, ids, redraws)
    # the plain tokenizer accepts the first draw, so the streams differ once a 9 is redrawn
    assert texts != pg.control_answers(answers, DigitTok())[0]


def test_control_answers_raise_when_no_draw_round_trips():
    with pytest.raises(RuntimeError, match="no 2-token control"):
        pg.control_answers(["3888"], NoRoundTripTok())


def test_control_identical_across_forms(tok, rows):
    w = pg.build_items(make_val(tok, "word", rows), tok, EOS)
    s = pg.build_items(make_val(tok, "symbol", rows), tok, EOS)
    assert w["control_text"] == s["control_text"]
    assert torch.equal(w["control"], s["control"]) and torch.equal(w["data"], s["data"])
    assert w["control_check"]["sha256"] == s["control_check"]["sha256"]
    assert w["data_sha256"] == s["data_sha256"]
    assert w["prompts"].shape[1] != s["prompts"].shape[1]  # the forms really differ


def test_control_check_passes_on_clean_data(items):
    c = items["control_check"]
    assert c["pass"] and c["n"] == 40 and c["n_redraws"] == 0
    for k in ("digit_count_ok", "sign_ok", "never_equal", "two_tokens_roundtrip",
              "true_tokens_match"):  # fmt: skip
        assert c[k] is True


def test_control_check_fails_when_stored_tokens_disagree_with_tokenizer(tok, rows):
    val = make_val(tok, "symbol", rows)
    val["c2"] = val["c2"].clone()
    val["c2"][3] += 1  # the stored second answer token no longer matches answer_text
    c = pg.build_items(val, tok, EOS)["control_check"]
    assert c["true_tokens_match"] is False and c["pass"] is False


def test_build_items_rejects_clean_ids_not_ending_in_c1(tok, rows):
    val = make_val(tok, "symbol", rows)
    val["clean_ids"] = val["clean_ids"].clone()
    val["clean_ids"][0, -1] += 1
    with pytest.raises(ValueError, match="prompt \\+ first answer token"):
        pg.build_items(val, tok, EOS)


def test_build_items_groups_are_digit_counts(items):
    assert items["groups"] == [len(a) for a in items["answer_text"]]
    assert Counter(items["groups"]) == {4: 20, 5: 20}


# ----------------------------------------------------------------------------- layout


def test_build_batch_rows_are_prompt_i_plus_answer_j_plus_eos(tok, rows, items):
    val = make_val(tok, "symbol", rows)
    n = len(rows)
    idx = [(i, j) for i in range(n) for j in range(n)]
    x = pg.build_batch(items["prompts"], items["data"], idx)
    P = val["clean_ids"].shape[1] - 1
    assert x.shape == (n * n, P + 3)
    for r, (i, j) in enumerate(idx):
        assert x[r].tolist() == val["clean_ids"][i, :-1].tolist() + [
            int(val["c1"][j]), int(val["c2"][j]), EOS]  # fmt: skip
        assert tok.decode(x[r]) == SYMBOL.format(a=rows[i][0], b=rows[i][1]) + str(sum(rows[j]))
    xc = pg.build_batch(items["prompts"], items["control"], [(2, 5)])
    assert tok.decode(xc[0]) == SYMBOL.format(a=rows[2][0], b=rows[2][1]) + items["control_text"][5]
    assert xc[0, -1] == EOS


def test_true_items_keep_the_stored_clean_ids(tok, rows, items):
    val = make_val(tok, "word", rows)
    it = pg.build_items(val, tok, EOS)
    x = pg.build_batch(it["prompts"], it["data"], [(i, i) for i in range(len(rows))])
    assert torch.equal(x[:, :-2], val["clean_ids"])  # prompt + c1, untouched


def test_losses_use_exactly_the_three_label_positions(items):
    P = items["prompts"].shape[1]
    T = P + 3
    torch.manual_seed(5)
    bias = torch.randn(T, DigitTok.vocab_size)
    idx = [(0, 0), (1, 2), (7, 3)]
    base, cnt = pg.item_losses(position_lm(bias), items["prompts"], items["data"], idx, 2)
    assert cnt == [3, 3, 3]
    for t in range(T):
        b2 = bias.clone()
        b2[t] += torch.randn(DigitTok.vocab_size) * 3
        got, _ = pg.item_losses(position_lm(b2), items["prompts"], items["data"], idx, 2)
        changed = any(abs(g - b) > 1e-6 for g, b in zip(got, base))
        assert changed == (P - 1 <= t <= P + 1), t


def test_item_losses_match_hand_cross_entropy_on_tiny_llama(llama, items):
    idx = [(i, j) for i in range(6) for j in range(6)]
    got, cnt = pg.item_losses(llama, items["prompts"], items["data"], idx, 5)
    x = pg.build_batch(items["prompts"], items["data"], idx)
    P = items["prompts"].shape[1]
    with torch.no_grad():
        logits = llama(x).logits.double()
    want = [
        float(F.cross_entropy(logits[r, P - 1 : P + 2], x[r, P : P + 3], reduction="sum"))
        for r in range(len(idx))
    ]
    assert got == pytest.approx(want, abs=1e-4)
    assert cnt == [3] * len(idx) and all(g > 0 for g in got)


@pytest.mark.parametrize("bs", [1, 3, 64])
def test_item_losses_do_not_depend_on_batch_size(llama, items, bs):
    idx = [(i, (i * 7) % 40) for i in range(40)]
    ref, _ = pg.item_losses(llama, items["prompts"], items["control"], idx, 40)
    got, _ = pg.item_losses(llama, items["prompts"], items["control"], idx, bs)
    assert got == pytest.approx(ref, abs=1e-5)


def test_item_losses_leave_grad_off(llama, items):
    pg.item_losses(llama, items["prompts"], items["data"], [(0, 1)], 1)
    assert torch.is_grad_enabled()  # decorator restores it
    assert all(p.grad is None for p in llama.parameters())


# ----------------------------------------------------------------------------- the read


@pytest.mark.parametrize("scale", [1.0, 1e3])
def test_additive_loss_cancels_every_margin(items, scale):
    n = len(items["groups"])
    rng = random.Random(7)
    p = [rng.uniform(0, 30) * scale for _ in range(n)]
    r, rc = [rng.uniform(0, 30) * scale for _ in range(n)], [rng.uniform(0, 30) for _ in range(n)]
    rec = pg.pg_read(scorer_of(lambda q, a: p[q] + r[a]), scorer_of(lambda q, a: p[q] + rc[a]),
                     n, items["groups"], 0)  # fmt: skip
    tol = 1e-9 * max(1.0, scale)
    assert max(abs(m) for m in rec["margins_data"]) < tol
    assert max(abs(m) for m in rec["margins_control"]) < tol
    if scale == 1.0:  # within TIE_NATS: every pair a tie
        assert rec["acc"] == 0.5 and rec["s"] == 0.0 and rec["call"] == "teach"


def test_control_equal_to_data_gives_exactly_half(llama, items):
    n = len(items["groups"])
    sc = pg.scorer(llama, items["prompts"], items["data"], 16)
    rec = pg.pg_read(sc, sc, n, items["groups"], 0)
    assert rec["acc"] == 0.5 and rec["acc_ci"] == [0.5, 0.5]
    assert rec["s"] == 0.0 and rec["call"] == "teach"
    assert rec["margins_data"] == rec["margins_control"]
    assert rec["m21_excess_nats_per_token"] == 0.0


def test_planted_lookup_reads_elicit(items):
    n = len(items["groups"])
    m = lookup_lm(items["prompts"], items["data"])
    rec = pg.pg_read(pg.scorer(m, items["prompts"], items["data"], 16),
                     pg.scorer(m, items["prompts"], items["control"], 16), n, items["groups"], 0)  # fmt: skip
    assert rec["acc"] > 0.95 and rec["acc_ci"][0] > 0.5
    assert rec["call"] == "elicit" and rec["s"] > 0.9
    assert rec["m21_data_nats_per_token"] > rec["m21_control_nats_per_token"] + 1.0
    assert rec["true_loss_data_mean_nats"] < rec["true_loss_control_mean_nats"]


def test_prompt_indifferent_model_reads_teach(items):
    n = len(items["groups"])
    torch.manual_seed(8)
    m = position_lm(torch.randn(items["prompts"].shape[1] + 3, DigitTok.vocab_size) * 3)
    rec = pg.pg_read(pg.scorer(m, items["prompts"], items["data"], 7),
                     pg.scorer(m, items["prompts"], items["control"], 7), n, items["groups"], 0)  # fmt: skip
    assert all(x == 0.0 for x in rec["margins_data"] + rec["margins_control"])
    assert rec["acc"] == 0.5 and rec["call"] == "teach" and rec["s"] == 0.0


def test_model_linking_prompts_to_control_answers_reads_teach_with_zero_score(items):
    n = len(items["groups"])
    m = lookup_lm(items["prompts"], items["control"])  # memorised the wrong answers
    rec = pg.pg_read(pg.scorer(m, items["prompts"], items["data"], 16),
                     pg.scorer(m, items["prompts"], items["control"], 16), n, items["groups"], 0)  # fmt: skip
    assert rec["acc"] < 0.05 and rec["acc_ci"][1] < 0.5
    assert rec["s"] == 0.0 and rec["call"] == "teach"


def test_random_tiny_llama_reads_near_chance(llama, items):
    n = len(items["groups"])
    rec = pg.pg_read(pg.scorer(llama, items["prompts"], items["data"], 32),
                     pg.scorer(llama, items["prompts"], items["control"], 32), n,
                     items["groups"], 0)  # fmt: skip
    assert rec["acc_ci"][0] <= 0.5 + 0.15  # no planted link; small n keeps the CI wide
    assert all(math.isfinite(x) for x in rec["margins_data"] + rec["margins_control"])


def test_read_scores_exactly_the_needed_sequences(items):
    n = len(items["groups"])
    sd, sc = scorer_of(lambda q, a: float(q == a)), scorer_of(lambda q, a: 0.0)
    rec = pg.pg_read(sd, sc, n, items["groups"], 0)
    assert sd.rows == sc.rows == n + 2 * rec["n_pairs"]
    assert rec["n_sequences"] == sd.rows + sc.rows


def test_bootstrap_seed_changes_only_the_interval(items):
    n = len(items["groups"])
    rng = random.Random(9)
    noise = {(q, a): rng.gauss(0, 1) for q in range(n) for a in range(n)}
    noise2 = {(q, a): rng.gauss(0, 1) for q in range(n) for a in range(n)}
    f = scorer_of(lambda q, a: noise[(q, a)] - 0.3 * (q == a))
    g = scorer_of(lambda q, a: noise2[(q, a)])
    r0, r1 = pg.pg_read(f, g, n, items["groups"], 0), pg.pg_read(f, g, n, items["groups"], 1)
    assert r0["acc"] == r1["acc"] and r0["margins_data"] == r1["margins_data"]
    assert r0["acc_ci"] != r1["acc_ci"]


# ----------------------------------------------------------------------------- jackknife


def test_jackknife_matches_hand_computation_on_small_example():
    # 4 items, pairs (0,1) (1,2) (2,3) (0,3); data beats control on 3 pairs, one tie
    pairs = [(0, 1), (1, 2), (2, 3), (0, 3)]
    md, mc = [2.0, 1.0, 0.5, 3.0], [1.0, 1.0, 1.0, 0.0]  # wins 1, 1/2 (tie), 0, 1
    out = pg.jackknife_read(pairs, md, mc, 4)
    # W = 2.5, P = 4; item k is in 2 pairs: w_0 = 2, w_1 = 1.5, w_2 = 0.5, w_3 = 1
    loo = [(2.5 - 2) / 2, (2.5 - 1.5) / 2, (2.5 - 0.5) / 2, (2.5 - 1) / 2]
    m = sum(loo) / 4
    se = math.sqrt(3 / 4 * sum((x - m) ** 2 for x in loo))
    assert out["acc_se_jackknife"] == pytest.approx(se)
    assert out["acc_ci_jackknife"] == pytest.approx([2.5 / 4 - 1.96 * se, 2.5 / 4 + 1.96 * se])
    assert out["call_jackknife"] == "teach"


def test_jackknife_ties_use_tie_nats():
    pairs = [(0, 1), (1, 2), (0, 2)]
    tiny = pg.TIE_NATS / 2
    out = pg.jackknife_read(pairs, [tiny, -tiny, 0.0], [0.0, 0.0, 0.0], 3)
    assert out["acc_se_jackknife"] == 0.0 and out["acc_ci_jackknife"] == [0.5, 0.5]


def test_jackknife_item_in_no_pair_keeps_full_acc_and_rejects_degenerate():
    out = pg.jackknife_read([(0, 1), (1, 2), (0, 2)], [1, 1, 1], [0, 0, 0], 4)  # item 3 unpaired
    assert out["acc_ci_jackknife"] == [1.0, 1.0] and out["call_jackknife"] == "elicit"
    with pytest.raises(ValueError, match="every pair"):
        pg.jackknife_read([(0, 1)], [1.0], [0.0], 2)
    with pytest.raises(ValueError, match="no pairs"):
        pg.jackknife_read([], [], [], 3)


def test_jackknife_acc_equals_control_read_acc(items):
    n = len(items["groups"])
    rng = random.Random(10)
    nd = {(q, a): rng.gauss(0, 1) - 0.4 * (q == a) for q in range(n) for a in range(n)}
    nc = {(q, a): rng.gauss(0, 1) for q in range(n) for a in range(n)}
    rec = pg.pg_read(scorer_of(lambda q, a: nd[(q, a)]), scorer_of(lambda q, a: nc[(q, a)]), n,
                     items["groups"], 0)  # fmt: skip
    lo, hi = rec["acc_ci_jackknife"]
    assert (lo + hi) / 2 == pytest.approx(rec["acc"])
    assert hi - lo == pytest.approx(2 * 1.96 * rec["acc_se_jackknife"])
    assert rec["call_jackknife"] == ("elicit" if lo > 0.5 else "teach")


def test_jackknife_control_equal_to_data_gives_half_and_teach(llama, items):
    n = len(items["groups"])
    sc = pg.scorer(llama, items["prompts"], items["data"], 16)
    rec = pg.pg_read(sc, sc, n, items["groups"], 0)
    assert rec["acc_ci_jackknife"] == [0.5, 0.5] and rec["acc_se_jackknife"] == 0.0
    assert rec["call_jackknife"] == "teach"


def test_jackknife_planted_lookup_reads_elicit(items):
    n = len(items["groups"])
    m = lookup_lm(items["prompts"], items["data"])
    rec = pg.pg_read(pg.scorer(m, items["prompts"], items["data"], 16),
                     pg.scorer(m, items["prompts"], items["control"], 16), n, items["groups"], 0)  # fmt: skip
    assert rec["call_jackknife"] == "elicit" and rec["acc_ci_jackknife"][0] > 0.5


def test_jackknife_is_calibrated_under_a_null_where_the_bootstrap_is_not():
    """iid N(0,1) losses for data and control (no link): the jackknife SE tracks the true
    spread of acc*, the item bootstrap's half-width is narrower."""
    from geode.edl.pairing import control_read, needed_cross

    n, reps = 256, 30
    groups = [4, 5] * (n // 2)
    pairs = pg.make_pairs(n, pg.OFFSETS, groups=groups)
    need = needed_cross(pairs)
    accs, ses, boot_hw, jk_hw = [], [], [], []
    for r in range(reps):
        rng = random.Random(100 + r)
        dt, ct = [rng.gauss(0, 1) for _ in range(n)], [rng.gauss(0, 1) for _ in range(n)]
        dc, cc = {k: rng.gauss(0, 1) for k in need}, {k: rng.gauss(0, 1) for k in need}
        out = control_read((dt, dc), (ct, cc), pairs, seed=r)
        jk = pg.jackknife_read(pairs, out["margins_data"], out["margins_control"], n)
        accs.append(out["acc"])
        ses.append(jk["acc_se_jackknife"])
        boot_hw.append((out["acc_ci"][1] - out["acc_ci"][0]) / 2)
        jk_hw.append((jk["acc_ci_jackknife"][1] - jk["acc_ci_jackknife"][0]) / 2)
    m = sum(accs) / reps
    sd = math.sqrt(sum((a - m) ** 2 for a in accs) / (reps - 1))
    assert sum(ses) / reps >= 0.9 * sd
    assert sum(boot_hw) / reps < sum(jk_hw) / reps


# ----------------------------------------------------------------------------- pairs


def test_pairs_stay_within_digit_count(items):
    n = len(items["groups"])
    rec = pg.pg_read(scorer_of(lambda q, a: 0.0), scorer_of(lambda q, a: 0.0), n,
                     items["groups"], 0)  # fmt: skip
    g = items["groups"]
    assert rec["pairs_within_group"] is True
    assert all(g[i] == g[j] and i < j for i, j in rec["pairs"])
    assert rec["n_pairs"] == len(rec["pairs"]) == 4 * n  # 4 ring offsets per item per group
    per = Counter(k for p in rec["pairs"] for k in p)
    assert set(per.values()) == {8}
    assert len(rec["margins_data"]) == len(rec["margins_control"]) == rec["n_pairs"]


def test_pairs_small_group_clips_offsets_and_never_crosses():
    groups = [4, 4, 4] + [5] * 12
    rec = pg.pg_read(scorer_of(lambda q, a: 0.0), scorer_of(lambda q, a: 0.0), len(groups),
                     groups, 0)  # fmt: skip
    small = [p for p in rec["pairs"] if groups[p[0]] == 4]
    assert sorted(map(tuple, small)) == [(0, 1), (0, 2), (1, 2)]
    assert rec["n_pairs"] == 3 + 4 * 12 and rec["pairs_within_group"]


def test_single_item_groups_have_no_pairs_and_raise():
    with pytest.raises(ValueError, match="no pairs"):
        pg.pg_read(scorer_of(lambda q, a: 0.0), scorer_of(lambda q, a: 0.0), 2, [4, 5], 0)


# ----------------------------------------------------------------------------- end to end

KEYS = {
    "tag", "model", "form", "data_dir", "n_items", "groups", "offsets", "seed", "control_seed",
    "eos_id", "prompt_len", "n_label", "data_sha256", "control_check", "n_pairs", "acc",
    "acc_ci", "s", "call", "m21_data_nats_per_pair", "m21_control_nats_per_pair",
    "margins_data", "margins_control", "m21_data_nats_per_token",
    "m21_control_nats_per_token", "m21_excess_nats_per_token", "pairs", "pairs_within_group",
    "true_loss_data_mean_nats", "true_loss_control_mean_nats", "n_sequences", "elapsed_s",
    "acc_se_jackknife", "acc_ci_jackknife", "call_jackknife",
}  # fmt: skip


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory, tok, rows):
    d = tmp_path_factory.mktemp("pg_data")
    disc = make_rows(2, 2, seed=11)
    for form, fname in (("word", eapig_data.ADDITION_PAIRS_FILENAME),
                        ("symbol", eapig_data.SYMBOL_PAIRS_FILENAME)):  # fmt: skip
        v, dd = make_val(tok, form, rows), make_val(tok, form, disc, split="discovery")
        both = {k: torch.cat([dd[k], v[k]]) for k in ("clean_ids", "c1", "c2")}
        both["answer_text"] = dd["answer_text"] + v["answer_text"]
        both["meta"] = pd.concat([dd["meta"], v["meta"]], ignore_index=True)
        eapig_data.save_pairs(both, d / fname)
    eapig_data.save_pairs({"clean_ids": torch.zeros(1, 3, dtype=torch.long)},
                          d / eapig_data.COPY_PAIRS_FILENAME)  # fmt: skip
    return d


def run_main(monkeypatch, llama, tok, data_dir, out, form, tag="ep_symbol"):
    monkeypatch.setattr(pg, "load_tokenizer", lambda: tok)
    monkeypatch.setattr(pg.run, "load_model", lambda path, device: llama)
    return pg.main(["--tag", tag, "--model", "fake/model", "--form", form, "--data-dir",
                    str(data_dir), "--out", str(out), "--batch-size", "32", "--device", "cpu"])  # fmt: skip


@pytest.mark.parametrize("form", ["word", "symbol"])
def test_main_end_to_end_writes_json_with_all_keys(monkeypatch, llama, tok, data_dir, tmp_path,
                                                   form):  # fmt: skip
    out = tmp_path / "pg" / f"x_{form}.json"
    assert run_main(monkeypatch, llama, tok, data_dir, out, form) == 0
    rec = json.loads(out.read_text())
    assert set(rec) == KEYS
    assert rec["form"] == form and rec["n_items"] == 40  # validation rows only
    assert rec["groups"] == {"4": 20, "5": 20} and rec["offsets"] == 4
    assert rec["n_pairs"] == 160 and len(rec["margins_data"]) == 160
    assert rec["control_check"]["pass"] and rec["pairs_within_group"]
    assert rec["call"] in ("elicit", "teach") and 0.0 <= rec["s"] <= 1.0
    assert rec["acc_ci"][0] <= rec["acc"] <= rec["acc_ci"][1]
    assert rec["eos_id"] == EOS and rec["n_label"] == 3 and rec["control_seed"] == 20261112
    assert rec["prompt_len"] == (30 if form == "word" else 10)
    assert rec["n_sequences"] == 2 * (40 + 2 * 160)
    assert all(math.isfinite(rec[k]) for k in KEYS if k.endswith("_nats") or "nats_per" in k)


def test_main_forms_share_control_and_differ_in_losses(monkeypatch, llama, tok, data_dir, tmp_path):
    recs = {}
    for form in ("word", "symbol"):
        out = tmp_path / f"{form}.json"
        run_main(monkeypatch, llama, tok, data_dir, out, form)
        recs[form] = json.loads(out.read_text())
    w, s = recs["word"], recs["symbol"]
    assert w["control_check"]["sha256"] == s["control_check"]["sha256"]
    assert w["data_sha256"] == s["data_sha256"] and w["pairs"] == s["pairs"]
    assert w["margins_data"] != s["margins_data"]


def test_main_is_deterministic(monkeypatch, llama, tok, data_dir, tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    run_main(monkeypatch, llama, tok, data_dir, a, "symbol")
    run_main(monkeypatch, llama, tok, data_dir, b, "symbol")
    ra, rb = json.loads(a.read_text()), json.loads(b.read_text())
    ra.pop("elapsed_s"), rb.pop("elapsed_s")
    assert ra == rb


@pytest.mark.parametrize("argv", [
    ["--tag", "t", "--model", "m", "--data-dir", "d", "--out", "o"],  # no --form
    ["--tag", "t", "--model", "m", "--form", "translate", "--data-dir", "d", "--out", "o"],
    ["--model", "m", "--form", "word", "--data-dir", "d", "--out", "o"],  # no --tag
])  # fmt: skip
def test_parse_args_rejects_bad_invocations(argv):
    with pytest.raises(SystemExit):
        pg.parse_args(argv)


def test_parse_args_defaults():
    a = pg.parse_args(["--tag", "t", "--model", "m", "--form", "word", "--data-dir", "d",
                       "--out", "o"])  # fmt: skip
    assert a.batch_size == 64 and a.seed == 0 and isinstance(a.data_dir, Path)
