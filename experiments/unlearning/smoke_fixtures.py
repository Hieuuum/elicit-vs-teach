"""CPU smoke fixtures for the unlearning launcher: no network, no pretrained weights.

--dataset wmdp (default): under <dir>
  data/        prepare.py --dataset wmdp --synthetic output (harmless invented
               MCQ items shaped like WMDP bio/cyber + MMLU)
  tokenizer/   a byte-level BPE trained in-process on that text, with the
               Mistral/Zephyr specials <s> </s> <unk>
  original/ unlearned/   tiny random MISTRAL-architecture models (the chosen
               family: GQA, untied embeddings, like Zephyr-7B-beta), two seeds

--dataset tofu: data/ (TOFU-shaped, invented authors), a BPE with the Llama-3
  chat specials, tiny random Llama models original/ unlearned/ retain/.

Usage:  python3 smoke_fixtures.py <dir> [--dataset wmdp|tofu] [--layers 2] [--d-model 64]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPECIALS = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
MISTRAL_SPECIALS = ["<unk>", "<s>", "</s>"]


def build_tokenizer(texts: list[str], out: Path, vocab_size: int = 4000, mistral: bool = False):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = MISTRAL_SPECIALS if mistral else SPECIALS
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    if mistral:
        fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>",
                                       unk_token="<unk>", pad_token="</s>")
    else:
        fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|begin_of_text|>",
                                       eos_token="<|eot_id|>", pad_token="<|eot_id|>")
    fast.save_pretrained(out)
    return fast


def build_model(tokenizer, seed: int, out: Path, layers: int, d_model: int, family: str = "llama"):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM

    common = dict(vocab_size=len(tokenizer), hidden_size=d_model, intermediate_size=2 * d_model,
                  num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=1024, bos_token_id=tokenizer.bos_token_id,
                  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    torch.manual_seed(seed)
    if family == "mistral":
        model = MistralForCausalLM(MistralConfig(**common, tie_word_embeddings=False, sliding_window=None))
    else:
        model = LlamaForCausalLM(LlamaConfig(**common, tie_word_embeddings=True))
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--dataset", choices=("wmdp", "tofu"), default="wmdp")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--authors", type=int, default=3, help="tofu: invented authors per split")
    ap.add_argument("--items", type=int, default=48, help="wmdp: invented items per domain")
    args = ap.parse_args()
    import pandas as pd

    data = args.out / "data"
    n = args.items if args.dataset == "wmdp" else args.authors
    subprocess.run([sys.executable, str(HERE / "data" / "prepare.py"), "--dataset", args.dataset,
                    "--synthetic", str(n), "--out-dir", str(data), "--confirm"],
                   check=True, stdout=subprocess.DEVNULL)
    if args.dataset == "wmdp":
        ev = pd.read_parquet(data / "wmdp_eval.parquet")
        texts = ev["prompt_text"].tolist() + [" A", " B", " C", " D"] * 50
        for f in ("relearn_bioA", "relearn_mmluA"):
            texts += pd.read_parquet(data / f"{f}.parquet")["full_text"].tolist()
        tok = build_tokenizer(texts, args.out / "tokenizer", vocab_size=1500, mistral=True)
        for name, seed in (("original", 1), ("unlearned", 2)):
            build_model(tok, seed, args.out / name, args.layers, args.d_model, family="mistral")
        stories = pd.read_parquet(data / "relearn_bioA.parquet")["full_text"].tolist()
        (data / "story.txt").write_text("<|endoftext|>".join(stories * 4))
        print(f"[smoke-fixtures] {args.out}: wmdp data ({len(ev)} items), tokenizer ({len(tok)}), "
              "mistral models original/unlearned")
        return 0
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
