"""V5.46 — ``geode.arith.decode.greedy_completions`` (promoted 2026-07-21).

The decode backs the G1/G2 gates AND the installers' in-loop format-validity
stopping eval (specs/02 §6, §8): a silent left-padding or EOS-handling bug
would score garbage and either stop an installer early or never stop it.
CPU-only, tiny in-process model + tokenizer (tests/conftest.py fixtures).
"""

from __future__ import annotations

import torch

from geode.arith import greedy_completions

# Ragged token-prefix prompts (BOS id 2, word ids >= 4): unequal lengths so
# batching exercises the left-padding path.
_PROMPTS = [
    [2, 10],
    [2, 11, 12, 13, 14],
    [2, 15, 16],
    [2, 17, 18, 19],
    [2, 20],
]


def test_v5_46_batch_size_invariance(tiny_llama, tiny_tokenizer):
    # V5.46: completions are identical whether prompts run one-at-a-time
    # (no padding at all) or all in one batch (maximal left-padding). This
    # is the property a wrong pad id, missing attention mask, or right-side
    # padding silently breaks.
    model = tiny_llama(seed=0)
    tok = tiny_tokenizer()
    one_at_a_time = greedy_completions(model, tok, _PROMPTS, device="cpu", batch_size=1)
    one_batch = greedy_completions(model, tok, _PROMPTS, device="cpu", batch_size=len(_PROMPTS))
    assert one_at_a_time == one_batch
    assert len(one_batch) == len(_PROMPTS)


def test_v5_46_greedy_is_deterministic(tiny_llama, tiny_tokenizer):
    # V5.46: greedy decode — two calls agree exactly.
    model = tiny_llama(seed=1)
    tok = tiny_tokenizer()
    a = greedy_completions(model, tok, _PROMPTS, device="cpu", batch_size=2)
    b = greedy_completions(model, tok, _PROMPTS, device="cpu", batch_size=2)
    assert a == b


def test_v5_46_stops_at_eos(tiny_llama, tiny_tokenizer):
    # V5.46: generation stops at the tokenizer's EOS and the EOS itself is
    # not decoded into the completion. Force it: make EOS (id 3) the argmax
    # at every position -> every completion is empty.
    model = tiny_llama(seed=2)
    tok = tiny_tokenizer()
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[3].fill_(1.0)
    completions = greedy_completions(model, tok, _PROMPTS, device="cpu", batch_size=2)
    assert completions == [""] * len(_PROMPTS)
