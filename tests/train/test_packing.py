"""TRAIN-1 packing tests — specs/02 §6.1 ``pack_corpus`` / ``train_val_split``.

Module under test (does not exist yet):

    from geode.train.packing import pack_corpus, train_val_split

Derived from the spec only:

- specs/02-training-run.md §6.1 ``pack_corpus`` contract — tokenize each
  document (no special tokens added), append exactly one ``eos_token_id``
  after every document, concatenate in input order, slice into consecutive
  rows of length ``seq_len``, drop the short tail; ValueError if
  ``eos_token_id is None`` or ``seq_len < 2``; deterministic (pure function).
- specs/02 §6.1 ``train_val_split`` contract — seeded permutation then split,
  ``n_val = round(val_fraction * n)`` clamped to ``[1, n-1]``; requires
  ``0 < val_fraction < 1`` and ``n >= 2``, else ValueError; rows preserved
  exactly (a partition, no mutation).
- V5.17 (packing), V5.18 (split), V5.27 (streamed packing).

Fixture note (verified against ``tests/conftest.py``): ``tiny_tokenizer``
builds word tokens ``t0..t123`` after the four specials, so the mapping is
``t{i} -> id i+4`` (``"t0 t1" -> [4, 5]``) and ``eos_token_id == 3``. Every
expected token stream below is hand-derived from that mapping (arithmetic in
comments), never round-tripped through ``pack_corpus``.
"""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from geode.train.packing import pack_corpus, split_documents, train_val_split


def _distinct_rows(n: int, seq_len: int = 4) -> torch.LongTensor:
    # Row i == [i, i, ...]; every row is uniquely identifiable by any element,
    # so a partition/mutation check can read a row's identity off column 0.
    return torch.arange(n).reshape(n, 1).repeat(1, seq_len)


# --------------------------------------------------------------------------
# V5.17 — pack_corpus
# --------------------------------------------------------------------------
def test_pack_sequences_exact_len_and_stream_order(tiny_tokenizer):
    # V5.17: every emitted row has length exactly seq_len; the flattened rows
    # equal the concatenated per-document token streams with one eos(3) after
    # each document, in input order.
    tok = tiny_tokenizer()
    # Precondition pinning the fixture mapping the hand-arithmetic relies on.
    assert tok("t0 t1", add_special_tokens=False)["input_ids"] == [4, 5]
    assert tok.eos_token_id == 3
    # t0,t1,t2 -> 4,5,6 | t3,t4 -> 7,8 | t5,t6,t7,t8 -> 9,10,11,12 ; eos=3.
    texts = ["t0 t1 t2", "t3 t4", "t5 t6 t7 t8"]
    expected_stream = [4, 5, 6, 3, 7, 8, 3, 9, 10, 11, 12, 3]  # len 12
    assert len(expected_stream) == 12
    packed = pack_corpus(texts, tok, seq_len=4)
    assert packed.dtype == torch.long
    # 12 // 4 == 3 rows, each of length exactly 4, no tail dropped here.
    assert packed.shape == (3, 4)
    assert all(len(row) == 4 for row in packed.tolist())
    # Row boundaries cross document boundaries ([7,8,3,9] and [10,11,12,3]),
    # which only holds if concatenation precedes slicing and order is kept.
    assert packed.flatten().tolist() == expected_stream


def test_pack_drops_short_tail(tiny_tokenizer):
    # V5.17: the trailing partial row (strictly shorter than seq_len) is
    # dropped; kept rows equal the concatenated stream up to the drop point.
    tok = tiny_tokenizer()
    # t0,t1,t2 -> 4,5,6 | t3,t4 -> 7,8 | t5,t6 -> 9,10 ; eos=3.
    texts = ["t0 t1 t2", "t3 t4", "t5 t6"]
    full_stream = [4, 5, 6, 3, 7, 8, 3, 9, 10, 3]  # len 10
    seq_len = 4  # deliberately != eos_id (3) so the two can't be confused
    assert len(full_stream) % seq_len != 0  # a tail must exist
    n_rows = len(full_stream) // seq_len  # 10 // 4 == 2
    packed = pack_corpus(texts, tok, seq_len=seq_len)
    assert packed.shape == (n_rows, seq_len) == (2, 4)
    assert packed.flatten().tolist() == full_stream[: n_rows * seq_len]
    assert packed.flatten().tolist() == [4, 5, 6, 3, 7, 8, 3, 9]
    dropped_tail = full_stream[n_rows * seq_len :]
    assert dropped_tail == [10, 3]  # the two dropped tokens
    assert 0 < len(dropped_tail) < seq_len  # strictly shorter tail


def test_pack_missing_eos_raises(tiny_tokenizer):
    # V5.17: a tokenizer with eos_token_id None must raise ValueError.
    tok = tiny_tokenizer()
    tok.eos_token = None
    assert tok.eos_token_id is None  # precondition for this test
    with pytest.raises(ValueError):
        pack_corpus(["t0 t1 t2"], tok, seq_len=4)


def test_pack_deterministic(tiny_tokenizer):
    # V5.17: output is a pure function of the inputs (deterministic).
    texts = ["t0 t1 t2", "t3 t4", "t5 t6 t7 t8"]
    a = pack_corpus(texts, tiny_tokenizer(), seq_len=4)
    b = pack_corpus(texts, tiny_tokenizer(), seq_len=4)
    assert torch.equal(a, b)


def test_pack_adds_no_special_tokens():
    # §6.1: "Tokenize each document (no special tokens added)". The conftest
    # tokenizer has no post-processor, so it never adds specials and cannot
    # exercise this clause; build a variant whose DEFAULT tokenization mode
    # prepends [BOS] (id 2) — as real Llama tokenizers do — and require that
    # no BOS leaks into the packed stream.
    words = [f"t{i}" for i in range(10)]
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[BOS]", "[EOS]", *words])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 2)])
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    # Preconditions: default mode DOES add BOS; the raw per-document streams
    # are t0,t1,t2 -> 4,5,6 and t3,t4,t5,t6 -> 7,8,9,10; eos == 3.
    assert tok("t0 t1")["input_ids"] == [2, 4, 5]
    assert tok("t0 t1", add_special_tokens=False)["input_ids"] == [4, 5]
    assert tok.eos_token_id == 3
    packed = pack_corpus(["t0 t1 t2", "t3 t4 t5 t6"], tok, seq_len=4)
    # Stream [4,5,6,3,7,8,9,10,3] (len 9) -> 2 rows, tail [3] dropped. A
    # per-document BOS would shift every token and fail the equality.
    assert packed.shape == (2, 4)
    assert packed.flatten().tolist() == [4, 5, 6, 3, 7, 8, 9, 10]


@pytest.mark.parametrize("bad_seq_len", [1, 0])
def test_pack_seq_len_too_small_raises(tiny_tokenizer, bad_seq_len):
    # V5.17 / §6.1 interface: seq_len < 2 must raise ValueError.
    with pytest.raises(ValueError):
        pack_corpus(["t0 t1 t2"], tiny_tokenizer(), seq_len=bad_seq_len)


# --------------------------------------------------------------------------
# V5.18 — train_val_split
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n, val_fraction, expected_n_val",
    [
        (10, 0.2, 2),  # round(2.0) == 2, clamp[1,9] -> 2
        (10, 0.3, 3),  # round(3.0) == 3
        (10, 0.01, 1),  # round(0.1) == 0, clamped up to the [1, ...] floor
        (10, 0.99, 9),  # round(9.9) == 10, clamped down to the [..., n-1] cap
        (20, 0.15, 3),  # round(3.0) == 3
    ],
)
def test_split_exact_partition_and_sizes(n, val_fraction, expected_n_val):
    # V5.18: train/val are an exact partition of the input rows; n_val matches
    # the clamped-round formula; rows are preserved exactly (no mutation).
    seqs = _distinct_rows(n)
    before = seqs.clone()
    train, val = train_val_split(seqs, val_fraction, seed=0)
    n_train = n - expected_n_val
    assert val.shape[0] == expected_n_val
    assert train.shape[0] == n_train
    # n_val and n_train are deliberately unequal so a field swap would fail.
    assert expected_n_val != n_train
    # Partition: row-ids 0..n-1 appear exactly once across train and val.
    all_ids = torch.cat([train, val], dim=0)[:, 0].tolist()
    assert sorted(all_ids) == list(range(n))
    assert set(train[:, 0].tolist()).isdisjoint(set(val[:, 0].tolist()))
    # No mutation: every returned row is still its original constant vector,
    # and the input tensor itself is untouched.
    for row in torch.cat([train, val], dim=0):
        assert torch.equal(row, torch.full((seqs.shape[1],), int(row[0])))
    assert torch.equal(seqs, before)


@pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.5])
def test_split_invalid_fraction_raises(bad_fraction):
    # V5.18 / §6.1 interface: requires 0 < val_fraction < 1, else ValueError.
    with pytest.raises(ValueError):
        train_val_split(_distinct_rows(10), bad_fraction, seed=0)


def test_split_too_few_rows_raises():
    # V5.18 / §6.1 interface: requires n >= 2, else ValueError.
    with pytest.raises(ValueError):
        train_val_split(_distinct_rows(1), 0.5, seed=0)


# --------------------------------------------------------------------------
# V5.26 — split_documents
# --------------------------------------------------------------------------
def test_split_documents_between_delimiter_lines():
    # V5.26: documents are the stripped text between delimiter-only lines,
    # in input order; internal newlines within a document are preserved.
    lines = [
        "story one line a\n",
        "story one line b\n",
        "<|endoftext|>\n",
        "story two\n",
        "<|endoftext|>\n",
    ]
    assert list(split_documents(lines)) == [
        "story one line a\nstory one line b",
        "story two",
    ]


def test_split_documents_drops_empties_and_never_emits_delimiter():
    # V5.26: leading, trailing, and consecutive delimiter lines yield no
    # empty documents; the delimiter string never appears in any output;
    # a final undelimited document is still emitted.
    lines = [
        "<|endoftext|>\n",  # leading
        "  \n",  # whitespace-only doc -> dropped after strip
        "<|endoftext|>\n",
        "<|endoftext|>\n",  # consecutive
        "alpha\n",
        "<|endoftext|>  \n",  # trailing whitespace on a delimiter line is fine
        "omega no trailing delimiter\n",
    ]
    docs = list(split_documents(lines))
    assert docs == ["alpha", "omega no trailing delimiter"]
    assert all("<|endoftext|>" not in d for d in docs)


def test_split_documents_mid_line_delimiter_raises():
    # V5.26: a line containing the delimiter that is not exactly a delimiter
    # line must raise (silently keeping delimiter text would corrupt the
    # corpus); the error must be raised at the offending line even lazily.
    lines = ["ok\n", "bad<|endoftext|>tail\n", "never reached\n"]
    with pytest.raises(ValueError):
        list(split_documents(lines))


def test_split_documents_is_lazy():
    # V5.26: consumes the line stream incrementally — documents before an
    # exhausted point are yielded without reading past their delimiter.
    def gen():
        yield "doc\n"
        yield "<|endoftext|>\n"
        raise RuntimeError("must not be pulled for the first document")

    it = split_documents(gen())
    assert next(it) == "doc"
    with pytest.raises(RuntimeError):
        next(it)


def test_split_documents_custom_delimiter():
    # V5.26: the delimiter is a parameter, not a hardcoded constant.
    lines = ["a\n", "###\n", "b\n"]
    assert list(split_documents(lines, delimiter="###")) == ["a", "b"]


def test_split_seeded_deterministic():
    # V5.18: same seed -> identical split; different seed -> different
    # permutation on non-trivial input.
    seqs = _distinct_rows(20)
    t0, v0 = train_val_split(seqs, 0.3, seed=7)
    t0b, v0b = train_val_split(seqs, 0.3, seed=7)
    assert torch.equal(t0, t0b) and torch.equal(v0, v0b)
    t1, v1 = train_val_split(seqs, 0.3, seed=8)
    # Same split sizes (same n and fraction) but a different permutation, so
    # the ordered concatenation differs (collision probability ~ 1/20!).
    assert v0.shape == v1.shape and t0.shape == t1.shape
    assert not torch.equal(torch.cat([t0, v0]), torch.cat([t1, v1]))


# --------------------------------------------------------------------------
# V5.27 — pack_corpus streams
# --------------------------------------------------------------------------
def test_v5_27_chunked_equals_reference(tiny_tokenizer):
    # V5.27: the packed tensor is identical for every valid chunk_tokens and
    # equals a naive tokenize-everything-then-slice reference built here,
    # independent of pack_corpus internals.
    tok = tiny_tokenizer()
    texts = [f"t{i} t{i + 1} t{i + 2}" for i in range(0, 60, 3)]
    ref_stream: list[int] = []
    for text in texts:
        ref_stream.extend(tok(text, add_special_tokens=False)["input_ids"])
        ref_stream.append(tok.eos_token_id)
    for seq_len in (2, 4, 7):
        n_rows = len(ref_stream) // seq_len
        expected = torch.tensor(ref_stream[: n_rows * seq_len], dtype=torch.long).reshape(
            n_rows, seq_len
        )
        for chunk_tokens in (seq_len, seq_len + 1, 3 * seq_len, 1 << 20):
            got = pack_corpus(texts, tok, seq_len, chunk_tokens=chunk_tokens)
            assert got.dtype == torch.long
            assert torch.equal(got, expected), (seq_len, chunk_tokens)


def test_v5_27_empty_and_subrow_corpora(tiny_tokenizer):
    # V5.27 edge: no texts, or fewer tokens than one full row, still yield a
    # (0, seq_len) long tensor regardless of chunking.
    tok = tiny_tokenizer()
    for texts in ([], ["t0"]):  # "t0" + eos == 2 tokens < seq_len 4
        packed = pack_corpus(texts, tok, seq_len=4, chunk_tokens=4)
        assert packed.shape == (0, 4)
        assert packed.dtype == torch.long


def test_v5_27_incremental_consumption(tiny_tokenizer):
    # V5.27: documents are pulled and tokenized one at a time — the iterable
    # is never drained into a list before tokenization starts.
    tok = tiny_tokenizer()
    events: list[str] = []

    class RecordingTokenizer:
        @property
        def eos_token_id(self) -> int:
            return tok.eos_token_id

        def __call__(self, text: str, **kwargs):
            events.append(f"tok:{text}")
            return tok(text, **kwargs)

    def gen():
        for i in range(3):
            events.append(f"yield:t{i}")
            yield f"t{i}"

    pack_corpus(gen(), RecordingTokenizer(), seq_len=2)
    assert events == ["yield:t0", "tok:t0", "yield:t1", "tok:t1", "yield:t2", "tok:t2"]


def test_v5_27_chunk_tokens_below_seq_len_raises(tiny_tokenizer):
    # V5.27 interface: chunk_tokens < seq_len must raise ValueError.
    with pytest.raises(ValueError):
        pack_corpus(["t0 t1"], tiny_tokenizer(), seq_len=4, chunk_tokens=3)
