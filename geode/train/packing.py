"""Corpus packing + train/val split (specs/02 §6.1).

``split_documents`` cuts a raw-text line stream into documents on
delimiter-only lines (the TinyStories ``<|endoftext|>`` convention);
``pack_corpus`` turns a stream of documents into fixed-length rows for
next-token pretraining; ``train_val_split`` partitions packed rows into a
seeded train/validation split. Nothing here imports ``geode.zoo`` or
``datasets`` — this module consumes in-memory text/token streams only.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

import torch


def split_documents(lines: Iterable[str], delimiter: str = "<|endoftext|>") -> Iterator[str]:
    """Split a line stream into documents on delimiter-only lines (V5.26).

    A document is the text strictly between consecutive delimiter lines
    (lines whose stripped content equals ``delimiter``), stripped of
    surrounding whitespace; empty documents are dropped, so leading,
    trailing, or consecutive delimiter lines yield nothing. The delimiter
    never appears in any yielded document: a line that *contains* the
    delimiter without being exactly a delimiter line raises ``ValueError``
    (silently training on delimiter text would corrupt the corpus).
    Lazy: consumes ``lines`` incrementally, so multi-GB files stream.
    """
    buf: list[str] = []
    for line in lines:
        if line.strip() == delimiter:
            doc = "".join(buf).strip()
            if doc:
                yield doc
            buf = []
        elif delimiter in line:
            raise ValueError(f"split_documents: delimiter {delimiter!r} appears mid-line: {line!r}")
        else:
            buf.append(line)
    doc = "".join(buf).strip()
    if doc:
        yield doc


def pack_corpus(
    texts: Iterable[str], tokenizer: Any, seq_len: int, *, chunk_tokens: int = 1 << 20
) -> torch.LongTensor:
    """Pack ``texts`` into consecutive rows of length ``seq_len`` (V5.17, V5.27).

    Each document is tokenized independently with ``add_special_tokens=False``
    (a tokenizer-added template, e.g. an auto-prepended BOS, is never applied
    here), and exactly one ``tokenizer.eos_token_id`` is appended after every
    document. The per-document token streams are concatenated in input order
    and sliced into consecutive rows of length ``seq_len``; a trailing
    partial row (strictly shorter than ``seq_len``) is dropped.

    Streaming (V5.27): documents are pulled and tokenized one at a time, and
    completed rows move into tensor storage whenever the Python token buffer
    reaches ``chunk_tokens``, so the buffer never holds more than
    ``chunk_tokens`` tokens plus one document — packing a 500M-token corpus
    costs the memory of the output tensor, not of a full-stream int list.
    ``chunk_tokens`` only tunes that memory profile: the result is identical
    for every valid value (a pure function of ``texts``, ``tokenizer``, and
    ``seq_len``).

    Raises ``ValueError`` if ``tokenizer.eos_token_id is None``,
    ``seq_len < 2``, or ``chunk_tokens < seq_len``.
    """
    if seq_len < 2:
        raise ValueError(f"pack_corpus: seq_len must be >= 2, got {seq_len}")
    if chunk_tokens < seq_len:
        raise ValueError(
            f"pack_corpus: chunk_tokens must be >= seq_len, got {chunk_tokens} < {seq_len}"
        )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("pack_corpus: tokenizer.eos_token_id is None")

    chunks: list[torch.Tensor] = []
    buf: list[int] = []
    for text in texts:
        buf.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        buf.append(eos_token_id)
        if len(buf) >= chunk_tokens:
            n_rows = len(buf) // seq_len
            chunks.append(
                torch.tensor(buf[: n_rows * seq_len], dtype=torch.long).reshape(n_rows, seq_len)
            )
            del buf[: n_rows * seq_len]
    n_rows = len(buf) // seq_len
    chunks.append(torch.tensor(buf[: n_rows * seq_len], dtype=torch.long).reshape(n_rows, seq_len))
    return torch.cat(chunks) if len(chunks) > 1 else chunks[0]


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
