"""Corpus packing + train/val split (specs/02 §6.1).

``pack_corpus`` turns a stream of documents into fixed-length rows for
next-token pretraining; ``train_val_split`` partitions packed rows into a
seeded train/validation split. Neither function imports ``geode.zoo`` or
``datasets`` — this module consumes and returns in-memory token tensors only.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch


def pack_corpus(texts: Iterable[str], tokenizer: Any, seq_len: int) -> torch.LongTensor:
    """Pack ``texts`` into consecutive rows of length ``seq_len`` (V5.17).

    Each document is tokenized independently with ``add_special_tokens=False``
    (a tokenizer-added template, e.g. an auto-prepended BOS, is never applied
    here), and exactly one ``tokenizer.eos_token_id`` is appended after every
    document. The per-document token streams are concatenated in input order
    and sliced into consecutive rows of length ``seq_len``; a trailing
    partial row (strictly shorter than ``seq_len``) is dropped. Deterministic:
    a pure function of ``texts``, ``tokenizer``, and ``seq_len``.

    Raises ``ValueError`` if ``tokenizer.eos_token_id is None`` or
    ``seq_len < 2``.
    """
    if seq_len < 2:
        raise ValueError(f"pack_corpus: seq_len must be >= 2, got {seq_len}")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("pack_corpus: tokenizer.eos_token_id is None")

    stream: list[int] = []
    for text in texts:
        stream.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        stream.append(eos_token_id)

    n_rows = len(stream) // seq_len
    stream = stream[: n_rows * seq_len]
    return torch.tensor(stream, dtype=torch.long).reshape(n_rows, seq_len)


def train_val_split(
    seqs: torch.LongTensor, val_fraction: float, seed: int
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Seeded permutation, then split into train/val rows (V5.18).

    ``n_val = round(val_fraction * n)`` clamped to ``[1, n - 1]``. Requires
    ``0 < val_fraction < 1`` and ``n >= 2``, else raises ``ValueError``. Rows
    are preserved exactly (an exact partition of the input rows; ``seqs`` is
    never mutated).
    """
    if not (0 < val_fraction < 1):
        raise ValueError(f"train_val_split: val_fraction must be in (0, 1), got {val_fraction}")
    n = seqs.shape[0]
    if n < 2:
        raise ValueError(f"train_val_split: need at least 2 rows, got {n}")

    n_val = max(1, min(round(val_fraction * n), n - 1))
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).to(seqs.device)
    train = seqs[perm[:n_train]]
    val = seqs[perm[n_train:]]
    return train, val
