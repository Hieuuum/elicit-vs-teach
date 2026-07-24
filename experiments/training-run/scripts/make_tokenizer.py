"""Train + freeze the run-1 custom tokenizer (spec 02 §12 OPEN(11), tokenizer half).

Byte-level BPE, vocab 10,000, trained on TinyStories-v2 (GPT-4-only file
inside roneneldan/TinyStories — verified 2026-07-18, the standalone
"TinyStoriesV2" repo id does not exist). Digits 0-9 are forced to single
tokens via a Digits pre-tokenizer split, so no learned merge ever contains
a digit; `+ - *` are single byte-level tokens by construction. The
`Question:` / `Answer:` template literals stay ordinary BPE output (owner
decision 2026-07-18: plain BPE over forced single tokens — subword pieces
carry pretrained embeddings into SFT).

Specials: `<|endoftext|>` (EOS — pack_corpus raises without one) and
`<|pad|>` (padding for later SFT/eval batches; adding it after the freeze
would change the vocab). No normalizer: byte-level round-trips exactly.

Output (frozen artifact, committed): experiments/training-run/tokenizer/
plus meta.json recording corpus provenance (sha256) and library versions.
Deterministic given the corpus + settings; no timestamp lands in the
artifact. CPU-only, no --confirm-cost needed (budget rule covers GPU jobs).

Usage:
    python3 make_tokenizer.py [--out DIR] [--vocab-size 10000] [--corpus-file PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from geode.arith.formats import render
from geode.train import split_documents

HF_REPO = "roneneldan/TinyStories"
CORPUS_FILE = "TinyStoriesV2-GPT4-train.txt"
EOS = "<|endoftext|>"
PAD = "<|pad|>"


def corpus_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, CORPUS_FILE, repo_type="dataset"))


def iter_docs(path: Path) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        yield from split_documents(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def train_tokenizer(corpus: Path, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    # Digits split first: every digit becomes its own pre-token, so BPE can
    # never merge digit pairs ("9999" is always four tokens) and no digit
    # ever fuses with a preceding space or letter.
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS, PAD],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(iter_docs(corpus), trainer=trainer)
    return tok


def verify(tok_dir: Path, vocab_size: int) -> dict[str, int]:
    """Post-hoc asserts on the *reloaded* artifact — what training will see."""
    tk = AutoTokenizer.from_pretrained(tok_dir)
    enc = lambda s: tk(s, add_special_tokens=False)["input_ids"]  # noqa: E731

    assert len(tk) == vocab_size, f"vocab {len(tk)} != {vocab_size}"
    assert tk.eos_token_id is not None, "EOS missing — pack_corpus would raise"
    assert tk.pad_token_id is not None and tk.pad_token_id != tk.eos_token_id

    # Digit forcing: "9999" is exactly four copies of the single "9" token.
    ids_9999 = enc("9999")
    assert len(ids_9999) == 4 and len(set(ids_9999)) == 1, f"9999 -> {ids_9999}"
    for d in "0123456789":
        assert len(enc(d)) == 1, f"digit {d} not a single token"
    # No learned token contains an ASCII digit (the ten bare digit tokens
    # aside). isdigit() would be wrong here: byte-level alphabet chars like
    # '²' count as digits to Python but are opaque byte symbols, not 0-9.
    for token in tk.get_vocab():
        if any(c in "0123456789" for c in token):
            assert token in "0123456789", f"digit inside learned token {token!r}"
    # Ops are single tokens standalone (byte-level alphabet guarantees this).
    for op in "+-*":
        assert len(enc(op)) == 1, f"op {op} not a single token"

    # Byte-level round-trip on frozen-template renders (worst-case shapes).
    for a, b, op, ans, fmt in [
        (9999, 9999, "+", 19998, "operator"),
        (1, 9999, "-", -9998, "nl"),
        (9999, 9999, "*", 99980001, "operator"),
        (23, 45, "+", 68, "nl"),
    ]:
        text, _ = render(a, b, op, ans, fmt)
        assert tk.decode(enc(text)) == text, f"round-trip failed: {text!r}"

    return {
        "question_prefix_tokens": len(enc("Question: ")),
        "answer_prefix_tokens": len(enc("\nAnswer: ")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent.parent / "tokenizer"
    )
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--corpus-file", type=Path, default=None, help="local corpus override")
    args = parser.parse_args()

    corpus = corpus_path(args.corpus_file)
    print(f"[tok] corpus: {corpus}", flush=True)
    tok = train_tokenizer(corpus, args.vocab_size)

    wrapped = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=EOS, pad_token=PAD)
    args.out.mkdir(parents=True, exist_ok=True)
    wrapped.save_pretrained(args.out)

    template_counts = verify(args.out, args.vocab_size)

    import tokenizers as tokenizers_lib
    import transformers

    meta = {
        "corpus": {"hf_repo": HF_REPO, "file": CORPUS_FILE, "sha256": sha256_file(corpus)},
        "vocab_size": args.vocab_size,
        "special_tokens": {"eos": EOS, "pad": PAD},
        "scheme": "byte-level BPE; Digits(individual_digits) pre-split; no normalizer",
        "template_literals": "plain BPE (owner decision 2026-07-18)",
        "versions": {
            "tokenizers": tokenizers_lib.__version__,
            "transformers": transformers.__version__,
        },
    } | template_counts
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[tok] frozen to {args.out}  (vocab {args.vocab_size}, all asserts passed)")
    print(
        f"[tok] 'Question: ' -> {template_counts['question_prefix_tokens']} tokens, "
        f"'\\nAnswer: ' -> {template_counts['answer_prefix_tokens']} tokens"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
