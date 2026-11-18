"""Shared in-process TOFU-shaped fixture: synthetic data through the real
``experiments/unlearning/data/prepare.py`` build path + a byte-level BPE with
the Llama-3 chat special tokens. No network, no files outside tmp dirs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "unlearning" / "data"))
import prepare  # noqa: E402

SPECIALS = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]


def build_frames(n_authors: int = 2, seed: int = 316):
    return prepare.build(prepare.synthetic(n_authors, seed), prepare.DATE, seed)


def write_frames(frames: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    return out


def build_bpe(frames: dict, vocab_size: int = 2500):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    from geode.adapt import fresh_names

    ev = frames["tofu_eval"]
    texts = ev["prompt_text"].tolist() + ev["answer_text"].tolist() + ev["para_prompt_text"].tolist()
    texts += [d for ds in ev["distractor_texts"] for d in ds]
    texts += frames["relearn_forgetA"]["full_text"].tolist()
    names = fresh_names(" ".join(texts))
    texts += [f"{n} said {n}.\n\n{n}" for n in names] * 20
    texts += ["x abc abc abcd abc"] * 50
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=tok, bos_token=SPECIALS[0], eos_token=SPECIALS[3],
                                   pad_token=SPECIALS[3])


def tiny_model(vocab_size: int, seed: int = 0, layers: int = 2, d_model: int = 64):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=vocab_size, hidden_size=d_model, intermediate_size=2 * d_model,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, tie_word_embeddings=True)
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg).eval()
