"""SETUP-0 acceptance checks: fixtures import, build offline, and behave.

Not spec-derived tests — those start at ZOO-1. This file only proves the
shared fixtures work (the retired build plan (git history) SETUP-0 acceptance).
"""

from __future__ import annotations

import torch

from tests.conftest import BOS_ID, DEFAULT_VOCAB_SIZE, FIRST_WORD_ID


def test_tiny_llama_builds_and_forwards(tiny_llama):
    model = tiny_llama(seed=0)
    ids = torch.tensor([[BOS_ID, 5, 5]])
    with torch.no_grad():
        logits = model(ids).logits
    assert logits.shape == (1, 3, DEFAULT_VOCAB_SIZE)


def test_tiny_llama_seed_determinism(tiny_llama):
    a = tiny_llama(seed=7)
    b = tiny_llama(seed=7)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_tiny_tokenizer_roundtrip(tiny_tokenizer):
    tok = tiny_tokenizer()
    assert tok.vocab_size == DEFAULT_VOCAB_SIZE
    ids = tok.encode("t0 t5 t99")
    assert ids == [FIRST_WORD_ID, FIRST_WORD_ID + 5, FIRST_WORD_ID + 99]
    assert tok.decode(ids) == "t0 t5 t99"


def test_copy_token_task_label_copies_prompt(copy_token_task):
    examples = copy_token_task(seed=0, n=32)
    assert len(examples) == 32
    for ex in examples:
        start, end = ex.label_span
        assert 0 <= start < end <= len(ex.input_ids)
        assert ex.label_ids() == [ex.input_ids[1]]
        assert all(FIRST_WORD_ID <= t < DEFAULT_VOCAB_SIZE for t in ex.input_ids[1:])


def test_random_label_task_spans_valid_and_deterministic(random_label_task):
    a = random_label_task(seed=3, n=32)
    b = random_label_task(seed=3, n=32)
    assert a == b
    for ex in a:
        start, end = ex.label_span
        assert 0 <= start < end <= len(ex.input_ids)
        assert all(FIRST_WORD_ID <= t < DEFAULT_VOCAB_SIZE for t in ex.input_ids[1:])


def test_geode_store_env_points_at_tmp(geode_store):
    import os

    assert os.environ["GEODE_STORE"] == str(geode_store)
    assert geode_store.is_dir()
