"""CPU smoke fixtures for the unlearning launcher: no network, no pretrained weights.

Builds, under one directory:
  data/            prepare.py --synthetic output (TOFU-shaped, invented authors)
  tokenizer/       a byte-level BPE trained in-process on that text, with the
                   Llama-3 chat special tokens (<|begin_of_text|>,
                   <|start_header_id|>, <|end_header_id|>, <|eot_id|>) so the
                   real chat-rendered prompts tokenize the same way in shape
  original/ unlearned/ retain/    tiny random Llama-architecture models (the
                   chosen family: GQA, tied embeddings, like Llama-3.2-1B),
                   different seeds, each saved with the tokenizer

Usage:  python3 smoke_fixtures.py <dir> [--layers 2] [--d-model 64]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPECIALS = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]


def build_tokenizer(texts: list[str], out: Path, vocab_size: int = 4000):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|begin_of_text|>",
                                   eos_token="<|eot_id|>", pad_token="<|eot_id|>")
    fast.save_pretrained(out)
    return fast


def build_model(tokenizer, seed: int, out: Path, layers: int, d_model: int):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=len(tokenizer), hidden_size=d_model, intermediate_size=2 * d_model,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, tie_word_embeddings=True,
                      bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                      pad_token_id=tokenizer.pad_token_id)
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--authors", type=int, default=3)
    args = ap.parse_args()
    import pandas as pd

    data = args.out / "data"
    subprocess.run([sys.executable, str(HERE / "data" / "prepare.py"), "--synthetic", str(args.authors),
                    "--out-dir", str(data), "--confirm"], check=True, stdout=subprocess.DEVNULL)
    ev = pd.read_parquet(data / "tofu_eval.parquet")
    texts = ev["prompt_text"].tolist() + ev["answer_text"].tolist() + ev["para_prompt_text"].tolist()
    texts += [d for ds in ev["distractor_texts"] for d in ds]
    for f in ("relearn_forgetA", "relearn_holdoutA"):
        texts += pd.read_parquet(data / f"{f}.parquet")["full_text"].tolist()
    sys.path.insert(0, str(HERE.parents[1]))
    from geode.adapt import fresh_names

    names = fresh_names(" ".join(texts))  # so invented names merge like real names do
    texts += [f"{n} said {n}.\n\n{n}" for n in names] * 30
    tok = build_tokenizer(texts, args.out / "tokenizer")
    for name, seed in (("original", 1), ("unlearned", 2), ("retain", 3)):
        build_model(tok, seed, args.out / name, args.layers, args.d_model)
    print(f"[smoke-fixtures] {args.out}: data ({len(ev)} probe rows), tokenizer ({len(tok)} tokens), "
          "models original/unlearned/retain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
