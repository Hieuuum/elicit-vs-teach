"""``probe1b_digits.py`` (probe1b phase 1 — per-digit probes on Llama-3.2-1B
vs evt-ts1b-base; ``docs/plan-probe1b-digits-phase1.md`` §4-9).

Silent failure modes guarded:

- ``cheat_and_affected``'s ``affected_k`` must be defined by cheat-WRONGNESS
  at digit k, never by carry PRESENCE -- a carry two places right of k can
  still leave cheat_k correct at k (units-digit sum mod 10 is unaffected by
  which further-right carry produced it), and treating "there was a carry
  somewhere" as the affected criterion would silently mislabel the R-P1/R-P2
  reads. ``affected4`` (units, no digit is more significant to its right) is
  a mathematical identity -- always False for genuine addition -- checked
  over hand-worked examples and 500 random rows.
- ``answer_positions``' hard asserts (exactly 2 answer tokens, token text
  matches the 3-then-remainder chunk, the char before the answer matches
  ``prev_char``) must actually fire rather than silently mislocating the
  probe position -- checked against hand-computed token indices (including
  the BOS shift) on a mock chunked tokenizer, and against deliberately
  malformed inputs (a 3-token answer, a wrong ``prev_char``).
- ``extract_features``' RIGHT-padded batched path must be numerically
  IDENTICAL to the unpadded per-row path -- if attention_mask/position
  handling were subtly wrong, trailing pad tokens could leak into an earlier
  real position's hidden state with no crash, silently corrupting every
  probed layer for every batched row.
- ``fit_linear_probe_predictions``' three-way contract (predictions,
  argmax-of-logprobs agreement, near-zero log-prob on a separable planted
  problem, chance-level accuracy on shuffled labels) is checked against
  synthetic features with a planted linear signal -- a broken standardise/
  fit/log-softmax step would silently produce plausible-looking but wrong
  numbers.
- ``lens_logprobs`` at the LAST hidden index must reproduce the model's own
  final-layer log-probs (the plan's free behavioral anchor, §6) -- an
  off-by-one in which hidden state is treated as "final", or a missed
  ``model.model.norm`` application, would silently invalidate that anchor.
- ``fit_all_heads``' central efficiency claim -- placement B and C share the
  pos1 d1-d3 fits rather than refitting -- is checked by literally counting
  calls into a monkeypatched ``fit_linear_probe_predictions`` (exactly
  n_hidden * 5 distinct fits * 2 for real+shuffled) and by requiring the
  reported B/C rows for d1-d3 to be numerically identical, not just close in
  spirit. The ``agg`` row's ``prod_prob_all = exp(sum of the 4 heads' mean_
  logprob_all)`` identity is checked self-consistently against the same
  returned frame.
- ``make_data``'s spec (dedup, range, digit-column agreement with ``ans``,
  half/half split, seed-determinism, the >=400-affected-test-row guard)
  is checked directly, including that the guard actually fires on a tiny n.

CPU-only, no network: a mock 3-digit-chunk tokenizer (built in this file)
stands in for the real Llama tokenizer, and a tiny randomly-initialized
``LlamaForCausalLM`` (module-scoped fixture, ``torch.manual_seed`` first)
stands in for the two real 1B models. ``probe1b_digits.py`` is being written
in parallel against the pinned contract in
``.claude/jobs/02ff8c27/tmp/probe1b_contract.md``; it may not exist yet, so
the module is loaded lazily inside a fixture (``p1b``) rather than at import
time, keeping collection green regardless of the other agent's progress.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tests._scriptloader import load


# ---------------------------------------------------------------------------
# Mock 3-digit-chunk tokenizer (pinned call surface: see contract "Test
# file" section). Chunks each maximal digit run left-to-right in 3s
# ("12345" -> "123", "45"); each maximal non-digit run is one token. BOS
# (id 0) is prepended with a zero-width (0, 0) offset when
# ``add_special_tokens=True``. Vocab ids are assigned deterministically, in
# first-seen order, starting at 2 (id 1 doubles as EOS/pad, matching the
# real setup's ``tok.pad_token = tok.eos_token``).
# ---------------------------------------------------------------------------
_RUN_RE = re.compile(r"\d+|\D+")


class MockChunkTokenizer:
    BOS_ID = 0
    EOS_PAD_ID = 1

    def __init__(self) -> None:
        self._next_id = 2
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {self.BOS_ID: "<bos>", self.EOS_PAD_ID: "<eos>"}
        self.pad_token = "<eos>"
        self.eos_token = "<eos>"
        self.pad_token_id = self.EOS_PAD_ID
        self.eos_token_id = self.EOS_PAD_ID
        self.padding_side = "right"

    def _id_for(self, tok_str: str) -> int:
        if tok_str not in self.token_to_id:
            tid = self._next_id
            self._next_id += 1
            self.token_to_id[tok_str] = tid
            self.id_to_token[tid] = tok_str
        return self.token_to_id[tok_str]

    def _encode_one(self, text: str, add_special_tokens: bool):
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        if add_special_tokens:
            ids.append(self.BOS_ID)
            offsets.append((0, 0))
        pos = 0
        for run in _RUN_RE.findall(text):
            if run[0].isdigit():
                for i in range(0, len(run), 3):
                    chunk = run[i : i + 3]
                    start = pos + i
                    ids.append(self._id_for(chunk))
                    offsets.append((start, start + len(chunk)))
            else:
                ids.append(self._id_for(run))
                offsets.append((pos, pos + len(run)))
            pos += len(run)
        return ids, offsets

    def __call__(
        self,
        text_or_texts,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        return_tensors: str | None = None,
        padding: bool = False,
        **_ignored,
    ):
        if isinstance(text_or_texts, str):
            ids, offsets = self._encode_one(text_or_texts, add_special_tokens)
            out = {"input_ids": ids}
            if return_offsets_mapping:
                out["offset_mapping"] = offsets
            return out

        all_ids = [self._encode_one(t, add_special_tokens)[0] for t in text_or_texts]
        if return_tensors == "pt":
            maxlen = max(len(ids) for ids in all_ids)
            input_ids = torch.full((len(all_ids), maxlen), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(all_ids), maxlen), dtype=torch.long)
            for row, ids in enumerate(all_ids):
                n = len(ids)
                if self.padding_side == "left":
                    input_ids[row, maxlen - n :] = torch.tensor(ids, dtype=torch.long)
                    attention_mask[row, maxlen - n :] = 1
                else:
                    input_ids[row, :n] = torch.tensor(ids, dtype=torch.long)
                    attention_mask[row, :n] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": all_ids}

    def decode(self, ids) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        parts = [
            self.id_to_token.get(int(i), "")
            for i in ids
            if int(i) not in (self.BOS_ID, self.EOS_PAD_ID)
        ]
        return "".join(parts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def p1b():
    """Lazily loads ``probe1b_digits.py``. Guarded inside a fixture (rather
    than at module import time) so collection succeeds even while the module
    under test does not exist yet or is half-written by the parallel author.
    """
    return load("probe1b_digits")


@pytest.fixture
def mock_tok() -> MockChunkTokenizer:
    """A fresh mock tokenizer per test -- vocab ids are assigned in
    first-seen order, so a fresh instance keeps positions/ids predictable
    for hand-computed expectations."""
    return MockChunkTokenizer()


@pytest.fixture(scope="module")
def tiny_model() -> LlamaForCausalLM:
    """One small randomly-initialized LlamaForCausalLM, built once for the
    whole module (no network, no pretrained weights)."""
    config = LlamaConfig(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model = model.float()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# test_cheat_digits_worked_examples
# ---------------------------------------------------------------------------


def test_cheat_digits_worked_examples(p1b):
    # Plan §4 worked example: 47 + 85 = 132. units (7+5)%10=2=true -> not
    # affected; tens (4+8)%10=2 != 3 -> affected; hundreds (0+0)%10=0 != 1
    # -> affected (missing digit of a/b at that place value = 0).
    cheats, affected = p1b.cheat_and_affected(47, 85, 132)
    assert cheats == [0, 2, 2]
    assert affected == [True, True, False]

    # Carry-without-affected case: 4759 + 4249 = 9008. units: (9+9)%10=8,
    # true units digit of 9008 is 8 -- cheat is RIGHT even though 9+9=18
    # carries into the tens place. affected is defined by cheat-WRONGNESS
    # at that digit, never by whether a carry occurred there.
    cheats2, affected2 = p1b.cheat_and_affected(4759, 4249, 9008)
    assert cheats2[-1] == 8
    assert affected2[-1] is False

    # affected4 (units of a 4-digit answer) is a mathematical identity for
    # addition: units(ans) = (units(a) + units(b)) % 10 always, regardless
    # of carries elsewhere -- so cheat4 == true units digit for every valid
    # (a, b, ans) row. Checked over 500 seeded rows sampled the way
    # make_data draws them (U[100, 9899], kept iff 1000 <= a+b <= 9999).
    rng = np.random.default_rng(316)
    checked = 0
    while checked < 500:
        a = int(rng.integers(100, 9900))
        b = int(rng.integers(100, 9900))
        ans = a + b
        if not (1000 <= ans <= 9999):
            continue
        _cheats, affected_row = p1b.cheat_and_affected(a, b, ans)
        assert len(affected_row) == 4
        assert affected_row[-1] is False
        checked += 1


# ---------------------------------------------------------------------------
# test_answer_span_two_tokens
# ---------------------------------------------------------------------------

# Hand-computed against the mock tokenizer's chunking rule (digit runs
# chunked left-to-right in 3s; non-digit runs are one token each; BOS
# prepended at id 0 shifts every real token's index by +1). Worked by hand,
# e.g. "1234 + 5678 = 6912" (op) tokenizes (no BOS) to
# ["123","4"," + ","567","8"," = ","691","2"] (indices 0-7); with BOS the
# first answer token "691" sits at index 7 -> pos1=6, pos2=7.
ANSWER_POSITION_CASES = [
    # (a, b, ans, fmt, expected_pos1, expected_pos2)
    (1234, 5678, 6912, "op", 6, 7),
    (1234, 5678, 6912, "nl", 7, 8),
    (42, 958, 1000, "op", 4, 5),
    (42, 958, 1000, "nl", 5, 6),
    (9899, 100, 9999, "op", 5, 6),
    (9899, 100, 9999, "nl", 6, 7),
]


@pytest.mark.parametrize("a,b,ans,fmt,exp_pos1,exp_pos2", ANSWER_POSITION_CASES)
def test_answer_span_two_tokens(p1b, mock_tok, a, b, ans, fmt, exp_pos1, exp_pos2):
    full_text, prompt_text, prev_char = p1b.render(a, b, ans, fmt)
    if fmt == "op":
        assert full_text == f"{a} + {b} = {ans}"
        assert prompt_text == f"{a} + {b} = "
        assert prev_char == " "
    else:
        assert full_text == f"What is the sum of {a} and {b}?\n{ans}"
        assert prompt_text == f"What is the sum of {a} and {b}?\n"
        assert prev_char == "\n"

    pos1, pos2 = p1b.answer_positions(mock_tok, full_text, str(ans), prev_char)
    assert (pos1, pos2) == (exp_pos1, exp_pos2)

    # Independently confirm exactly 2 answer tokens via the offset mapping:
    # pos2 is idx_token1 (the first answer token's own index) and
    # pos2 + 1 is idx_token2; their offset spans must slice out the
    # 3-then-remainder chunk of the answer string.
    enc = mock_tok(full_text, add_special_tokens=True, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    s1, e1 = offsets[pos2]
    s2, e2 = offsets[pos2 + 1]
    ans_str = str(ans)
    assert full_text[s1:e1] == ans_str[:3]
    assert full_text[s2:e2] == ans_str[3:]


# ---------------------------------------------------------------------------
# test_span_asserts_fire
# ---------------------------------------------------------------------------


def test_span_asserts_fire(p1b, mock_tok):
    # "1234567" (7 digits) chunks into "123","456","7" -- THREE answer
    # tokens, violating the "exactly 2" invariant. ``match=`` pins the
    # contract's "fail loudly, print the offending row" requirement -- a
    # bare ``pytest.raises(AssertionError)`` would also pass for an
    # unrelated assert (e.g. a stray ``text.endswith(ans)`` failure), so we
    # require the offending ``text`` to actually appear in the message.
    full_text = "1 + 1 = 1234567"
    assert full_text.endswith("1234567")
    with pytest.raises(AssertionError, match=re.escape(full_text)):
        p1b.answer_positions(mock_tok, full_text, "1234567", " ")

    # Correct 2-token answer, but the caller passes the WRONG prev_char
    # (the true char before "6912" is " ", not "\n").
    full_text2, _prompt, true_prev_char = p1b.render(1234, 5678, 6912, "op")
    assert true_prev_char == " "
    with pytest.raises(AssertionError, match=re.escape(full_text2)):
        p1b.answer_positions(mock_tok, full_text2, "6912", "\n")


# ---------------------------------------------------------------------------
# test_pos_indices_survive_padding
# ---------------------------------------------------------------------------

_PADDING_ROWS = [
    (1, 999, 1000, "op"),
    (4500, 4500, 9000, "op"),
    (9899, 100, 9999, "op"),
    (42, 958, 1000, "nl"),
]


def _build_rows(p1b, tok, rows):
    texts, positions, token_ids = [], [], []
    for a, b, ans, fmt in rows:
        full_text, _prompt, prev_char = p1b.render(a, b, ans, fmt)
        pos1, pos2 = p1b.answer_positions(tok, full_text, str(ans), prev_char)
        enc = tok(full_text, add_special_tokens=True, return_offsets_mapping=True)
        ids = enc["input_ids"]
        texts.append(full_text)
        positions.append((pos1, pos2))
        token_ids.append((ids[pos2], ids[pos2 + 1]))
    return texts, positions, token_ids


def test_pos_indices_survive_padding(p1b, tiny_model, mock_tok):
    texts, positions, token_ids = _build_rows(p1b, mock_tok, _PADDING_ROWS)
    assert len({len(t) for t in texts}) > 1  # rows genuinely differ in length

    feats_batched, lp_batched = p1b.extract_features(
        tiny_model, mock_tok, texts, positions, token_ids, batch_size=2, device="cpu"
    )
    feats_single, lp_single = p1b.extract_features(
        tiny_model, mock_tok, texts, positions, token_ids, batch_size=1, device="cpu"
    )

    assert feats_batched.shape == feats_single.shape
    assert feats_batched.shape[0] == len(texts)
    feats_gap = (feats_batched - feats_single).abs().max().item()
    lp_gap = (lp_batched - lp_single).abs().max().item()
    assert feats_gap < 1e-5, f"batched vs unbatched features differ by {feats_gap}"
    assert lp_gap < 1e-5, f"batched vs unbatched ans_logprobs differ by {lp_gap}"


# ---------------------------------------------------------------------------
# test_probe_recovers_planted_direction / test_shuffled_labels_at_chance /
# test_logprob_matches_pred
# ---------------------------------------------------------------------------


@pytest.fixture
def planted_probe_data():
    """[n, d] features with a digit label planted linearly: x = W[label] +
    small noise, class means widely separated relative to the noise scale."""
    n, d, n_classes = 400, 16, 10
    gen = torch.Generator().manual_seed(1)
    y = torch.randint(0, n_classes, (n,), generator=gen)
    onehot = torch.nn.functional.one_hot(y, n_classes).float()
    class_means = torch.randn(n_classes, d, generator=gen) * 6.0
    noise = 0.4 * torch.randn(n, d, generator=gen)
    x = onehot @ class_means + noise
    return x, y, n_classes


def test_probe_recovers_planted_direction(p1b, planted_probe_data):
    x, y, n_classes = planted_probe_data
    n = x.shape[0]
    half = n // 2
    x_train, y_train = x[:half], y[:half]
    x_test, y_test = x[half:], y[half:]

    _train_pred, test_pred, _test_logprobs = p1b.fit_linear_probe_predictions(
        x_train, y_train, x_test, y_test, n_classes, l2=1e-3
    )
    acc = (test_pred == y_test).double().mean().item()
    assert acc > 0.95


def test_shuffled_labels_at_chance(p1b, planted_probe_data):
    # SAME features as the planted-direction test; only the labels are
    # permuted, decorrelating x from y so E[test accuracy] = 1/n_classes
    # exactly, regardless of what the (possibly overfit) train-time fit
    # learns -- the fitted decision rule is a fixed function of x_test, and
    # y_test is independent of both x_test and that rule.
    x, y, n_classes = planted_probe_data
    n = x.shape[0]
    gen = torch.Generator().manual_seed(123)
    y_shuffled = y[torch.randperm(n, generator=gen)]
    half = n // 2
    x_train, y_train = x[:half], y_shuffled[:half]
    x_test, y_test = x[half:], y_shuffled[half:]

    _train_pred, test_pred, _test_logprobs = p1b.fit_linear_probe_predictions(
        x_train, y_train, x_test, y_test, n_classes, l2=1e-3
    )
    acc = (test_pred == y_test).double().mean().item()
    se = math.sqrt(0.1 * 0.9 / y_test.shape[0])
    assert abs(acc - 0.1) <= 3 * se


def test_logprob_matches_pred(p1b):
    n, d, n_classes = 300, 12, 10
    half = n // 2

    # Part 1: argmax(test_logprobs) == test_pred, by construction, on an
    # ordinary (moderately separable) planted problem.
    gen = torch.Generator().manual_seed(3)
    y = torch.randint(0, n_classes, (n,), generator=gen)
    onehot = torch.nn.functional.one_hot(y, n_classes).float()
    means = torch.randn(n_classes, d, generator=gen) * 5.0
    x = onehot @ means + 0.5 * torch.randn(n, d, generator=gen)
    x_train, y_train = x[:half], y[:half]
    x_test, y_test = x[half:], y[half:]
    _tp, test_pred, test_logprobs = p1b.fit_linear_probe_predictions(
        x_train, y_train, x_test, y_test, n_classes, l2=1e-3
    )
    assert torch.equal(test_pred, test_logprobs.argmax(dim=1))

    # Part 2: on a CLEANLY separable planted problem (huge signal, tiny
    # noise), the mean log-prob of the TRUE label should sit near 0 nats.
    gen2 = torch.Generator().manual_seed(4)
    y2 = torch.randint(0, n_classes, (n,), generator=gen2)
    onehot2 = torch.nn.functional.one_hot(y2, n_classes).float()
    means2 = torch.randn(n_classes, d, generator=gen2) * 20.0
    x2 = onehot2 @ means2 + 0.05 * torch.randn(n, d, generator=gen2)
    x2_train, y2_train = x2[:half], y2[:half]
    x2_test, y2_test = x2[half:], y2[half:]
    _tp2, _pred2, logprobs2 = p1b.fit_linear_probe_predictions(
        x2_train, y2_train, x2_test, y2_test, n_classes, l2=1e-3
    )
    mean_lp_true = logprobs2.gather(1, y2_test.view(-1, 1)).mean().item()
    assert mean_lp_true > -0.1


# ---------------------------------------------------------------------------
# test_lens_layer_final_equals_model
# ---------------------------------------------------------------------------

_LENS_ROWS = [
    (1, 999, 1000, "op"),
    (4500, 4500, 9000, "nl"),
    (9899, 100, 9999, "op"),
]


def test_lens_layer_final_equals_model(p1b, tiny_model, mock_tok):
    texts, positions, token_ids = _build_rows(p1b, mock_tok, _LENS_ROWS)
    feats, ans_logprobs = p1b.extract_features(
        tiny_model, mock_tok, texts, positions, token_ids, batch_size=2, device="cpu"
    )
    token_ids_t = torch.tensor(token_ids, dtype=torch.long)

    _top1, lens_lp = p1b.lens_logprobs(tiny_model, feats, token_ids_t, device="cpu")

    n_hidden = feats.shape[1]
    assert lens_lp.shape == (len(texts), n_hidden, 2)
    # The last hidden index IS the model's own final hidden state, so
    # norm+lm_head applied there must reproduce extract_features' own
    # directly-computed final-logit log-probs (an independent code path).
    # Tolerance is the module's own pinned constant (plan §6: "assert
    # equality to the direct forward log-probs within 1e-3"; contract:
    # ``LENS_ATOL_NATS = 1e-3``, the same bound the --fit driver enforces
    # at runtime) -- even float32 two independent matmul/RMSNorm paths
    # through the SAME weights land ~1e-4 apart, not exactly 0.
    assert torch.allclose(lens_lp[:, n_hidden - 1, :], ans_logprobs, atol=p1b.LENS_ATOL_NATS)


# ---------------------------------------------------------------------------
# test_b_and_c_share_pos1_fit
# ---------------------------------------------------------------------------

_PINNED_FIT_ALL_HEADS_COLUMNS = {
    "model",
    "format",
    "layer",
    "placement",
    "head",
    "top1_acc_all",
    "top1_acc_affected",
    "mean_logprob_all",
    "mean_logprob_affected",
    "n_all",
    "n_affected",
    "shuffled_top1_all",
    "shuffled_top1_affected",
    "cheat_acc_all",
    "cheat_acc_affected",
    "majority_acc",
    "prod_prob_all",
}


def _synthetic_fit_all_heads_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"d{k}": rng.integers(0, 10, size=n) for k in range(1, 5)}
    for k in range(1, 5):
        data[f"cheat{k}"] = rng.integers(0, 10, size=n)
    for k in range(1, 4):
        data[f"affected{k}"] = rng.integers(0, 2, size=n).astype(bool)
    data["affected4"] = np.zeros(n, dtype=bool)  # units digit: always False
    data["split"] = np.array(["train"] * (n // 2) + ["test"] * (n - n // 2))
    return pd.DataFrame(data)


def test_b_and_c_share_pos1_fit(p1b, monkeypatch):
    n_hidden, d, n = 3, 5, 40
    gen = torch.Generator().manual_seed(0)
    features = torch.randn(n, n_hidden, 2, d, generator=gen)
    df = _synthetic_fit_all_heads_df(n, seed=7)

    call_count = {"n": 0}
    original_fit = p1b.fit_linear_probe_predictions

    def counting_fit(*args, **kwargs):
        call_count["n"] += 1
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(p1b, "fit_linear_probe_predictions", counting_fit)

    result = p1b.fit_all_heads(features, df, "llama", "op", l2=1e-3, seed=316)

    # 3 layers * 5 distinct fits (d1,d2,d3,d4@pos1, d4@pos2) * 2 (real +
    # shuffled) -- placement B/C reuse the SAME pos1 d1-d3 fits, so this
    # must NOT be 3 * 6 * 2 (which a naive "fit every placement row" bug
    # would produce).
    assert call_count["n"] == n_hidden * 5 * 2

    assert set(result.columns) == _PINNED_FIT_ALL_HEADS_COLUMNS
    assert len(result) == n_hidden * 2 * 5  # 2 placements * (d1..d4, agg)
    assert set(result["placement"]) == {"B", "C"}
    assert set(result["head"]) == {"d1", "d2", "d3", "d4", "agg"}

    compare_cols = [
        "top1_acc_all",
        "top1_acc_affected",
        "mean_logprob_all",
        "mean_logprob_affected",
        "n_all",
        "n_affected",
        "shuffled_top1_all",
        "shuffled_top1_affected",
        "cheat_acc_all",
        "cheat_acc_affected",
        "majority_acc",
    ]
    b_rows = (
        result[(result["placement"] == "B") & (result["head"].isin(["d1", "d2", "d3"]))]
        .sort_values(["layer", "head"])
        .reset_index(drop=True)
    )
    c_rows = (
        result[(result["placement"] == "C") & (result["head"].isin(["d1", "d2", "d3"]))]
        .sort_values(["layer", "head"])
        .reset_index(drop=True)
    )
    assert len(b_rows) == len(c_rows) == n_hidden * 3
    for col in compare_cols:
        left = b_rows[col].to_numpy(dtype=float)
        right = c_rows[col].to_numpy(dtype=float)
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-10, equal_nan=True)

    # The other half of the property: d4 is NOT shared -- B fits it on pos1,
    # C fits it on pos2, independent random feature slices, so a degenerate
    # implementation that fit d4 once and reported it under both placements
    # would (silently) pass every check above. Require it to actually differ.
    b_d4 = result[(result["placement"] == "B") & (result["head"] == "d4")].sort_values("layer")
    c_d4 = result[(result["placement"] == "C") & (result["head"] == "d4")].sort_values("layer")
    b_d4_lp = b_d4["mean_logprob_all"].to_numpy(dtype=float)
    c_d4_lp = c_d4["mean_logprob_all"].to_numpy(dtype=float)
    assert not np.allclose(b_d4_lp, c_d4_lp)

    for layer in range(n_hidden):
        for placement in ("B", "C"):
            agg = result[
                (result["layer"] == layer)
                & (result["placement"] == placement)
                & (result["head"] == "agg")
            ]
            assert len(agg) == 1
            heads = result[
                (result["layer"] == layer)
                & (result["placement"] == placement)
                & (result["head"].isin(["d1", "d2", "d3", "d4"]))
            ]
            assert len(heads) == 4
            expected_prod = float(np.exp(heads["mean_logprob_all"].sum()))
            assert math.isclose(
                agg["prod_prob_all"].iloc[0], expected_prod, rel_tol=1e-9, abs_tol=1e-12
            )


# ---------------------------------------------------------------------------
# test_make_data_spec (bonus: cheap, high-value)
# ---------------------------------------------------------------------------


def test_make_data_spec(p1b):
    n = 6000
    df = p1b.make_data(n=n, seed=316)

    assert len(df) == n
    assert df.duplicated(subset=["a", "b"]).sum() == 0
    assert df["a"].between(100, 9899).all()
    assert df["b"].between(100, 9899).all()
    assert df["ans"].between(1000, 9999).all()
    assert (df["ans"] == df["a"] + df["b"]).all()

    expected_digits = df["ans"].astype(str).str.zfill(4)
    for k in range(1, 5):
        assert (df[f"d{k}"] == expected_digits.str[k - 1].astype(int)).all()

    counts = df["split"].value_counts()
    assert counts.get("train", 0) == n // 2
    assert counts.get("test", 0) == n - n // 2

    assert (~df["affected4"]).all()

    # Deterministic under the same seed.
    df_again = p1b.make_data(n=n, seed=316)
    pd.testing.assert_frame_equal(df, df_again)

    # Different under another seed.
    df_other_seed = p1b.make_data(n=n, seed=1)
    assert not df["a"].equals(df_other_seed["a"])


def test_make_data_raises_on_insufficient_affected_rows(p1b):
    # A tiny n gives a test half far below the >=400-affected-rows guard
    # (plan §4 / contract): this must raise, not silently proceed.
    with pytest.raises(ValueError):
        p1b.make_data(n=10, seed=1)
