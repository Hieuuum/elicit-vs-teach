"""Shared fixtures for the geode test suite.

Testing policy (CLAUDE.md): fixture models are randomly initialized tiny
configs built in-process; no network access, no pretrained weights, CPU only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

# Vocab layout shared by tiny_tokenizer and the synthetic tasks:
# ids 0..3 are specials, ids 4..vocab_size-1 are word tokens "t{i}".
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[BOS]", "[EOS]")
BOS_ID = 2
FIRST_WORD_ID = len(SPECIAL_TOKENS)
DEFAULT_VOCAB_SIZE = 128


@dataclass(frozen=True)
class TaskExample:
    """One synthetic example: a token sequence plus its label-token span.

    ``label_span`` is a half-open ``[start, end)`` index range into
    ``input_ids`` (OQ-8: the dataset builder emits per-example label-token
    spans; mask construction from spans belongs to ``geode.edl``).
    """

    input_ids: list[int]
    label_span: tuple[int, int]

    def label_ids(self) -> list[int]:
        start, end = self.label_span
        return self.input_ids[start:end]


@pytest.fixture
def tiny_llama() -> Callable[..., LlamaForCausalLM]:
    """Factory for randomly initialized in-process Llama-style models."""

    def make(
        seed: int,
        n_layers: int = 2,
        d_model: int = 64,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
    ) -> LlamaForCausalLM:
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            intermediate_size=2 * d_model,
            num_hidden_layers=n_layers,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=BOS_ID,
            eos_token_id=3,
        )
        torch.manual_seed(seed)
        model = LlamaForCausalLM(config)
        model.eval()
        return model

    return make


@pytest.fixture
def tiny_tokenizer() -> Callable[..., PreTrainedTokenizerFast]:
    """Factory for an in-process word-level tokenizer (no files, no network)."""

    def make(vocab_size: int = DEFAULT_VOCAB_SIZE) -> PreTrainedTokenizerFast:
        words = [f"t{i}" for i in range(vocab_size - len(SPECIAL_TOKENS))]
        vocab = {tok: i for i, tok in enumerate([*SPECIAL_TOKENS, *words])}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = WhitespaceSplit()
        return PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="[PAD]",
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
        )

    return make


def _word_ids(rng: random.Random, k: int, vocab_size: int) -> list[int]:
    return [rng.randrange(FIRST_WORD_ID, vocab_size) for _ in range(k)]


@pytest.fixture
def copy_token_task() -> Callable[..., list[TaskExample]]:
    """Trivially learnable task (V1.2): the label copies the prompt token.

    Each example is ``[BOS, x, x]`` with the final position as the label span.
    """

    def make(seed: int, n: int, vocab_size: int = DEFAULT_VOCAB_SIZE) -> list[TaskExample]:
        rng = random.Random(seed)
        examples = []
        for _ in range(n):
            (x,) = _word_ids(rng, 1, vocab_size)
            examples.append(TaskExample(input_ids=[BOS_ID, x, x], label_span=(2, 3)))
        return examples

    return make


@pytest.fixture
def random_label_task() -> Callable[..., list[TaskExample]]:
    """Unlearnable task (V1.1): labels are i.i.d. uniform, independent of x.

    Each example is ``[BOS, x, y]`` with the final position as the label span.
    """

    def make(seed: int, n: int, vocab_size: int = DEFAULT_VOCAB_SIZE) -> list[TaskExample]:
        rng = random.Random(seed)
        examples = []
        for _ in range(n):
            x, y = _word_ids(rng, 2, vocab_size)
            examples.append(TaskExample(input_ids=[BOS_ID, x, y], label_span=(2, 3)))
        return examples

    return make


@pytest.fixture
def geode_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point $GEODE_STORE at a fresh temporary directory."""
    store = tmp_path / "geode_store"
    store.mkdir()
    monkeypatch.setenv("GEODE_STORE", str(store))
    return store
