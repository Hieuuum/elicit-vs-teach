"""sft_item_nll_nats: the per-item companion of evaluate_sft_nll_nats (used by V1.14)."""

from __future__ import annotations

import pytest

from geode.edl.masking import TaskFormat
from geode.train.sft import evaluate_sft_nll_nats, sft_item_nll_nats


def test_item_sums_aggregate_to_the_set_mean_and_ignore_batching(tiny_llama, copy_token_task):
    model = tiny_llama(seed=0)
    ex = copy_token_task(seed=1, n=7)
    fmt = TaskFormat(name="tofu_qa", format_version="llama3_chat_v1")
    s1, c1 = sft_item_nll_nats(model, ex, fmt, batch_size=2, device="cpu")
    s2, c2 = sft_item_nll_nats(model, ex, fmt, batch_size=7, device="cpu")
    assert c1 == c2 and s1 == pytest.approx(s2, rel=1e-5)
    mean = evaluate_sft_nll_nats(model, ex, fmt, batch_size=3, device="cpu")
    assert sum(s1) / sum(c1) == pytest.approx(mean, rel=1e-5)
