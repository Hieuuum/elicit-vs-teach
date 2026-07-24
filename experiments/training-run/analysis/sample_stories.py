"""Qualitative story sampler — generate seeded stories from a run-1 checkpoint.

Formerly the gate-G0 sampler; G0 was removed from the protocol 2026-07-20
(spec 02 §8: run 1 ships at the paper recipe's validation-loss convergence,
ungated on sample quality). Kept as an inspection tool: it generates n
seeded samples and archives them next to the checkpoint so any qualitative
claim about a base model stays auditable.

Generation starts from a single EOS token — in packed pretraining every
story begins right after an EOS, so this samples the model's uncondi-
tional story distribution. Pure temperature sampling, one seeded batch.

Usage:
    python3 sample_stories.py --checkpoint <run_dir>/pretrain/model [--device cuda]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP_DIR = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=EXP_DIR / "tokenizer")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=316)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out", type=Path, default=None, help="default: <checkpoint>/floor1_samples.txt"
    )
    args = parser.parse_args()

    tk = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(args.device).eval()

    torch.manual_seed(args.seed)
    context = torch.full((args.n, 1), tk.eos_token_id, dtype=torch.long, device=args.device)
    with torch.no_grad():
        out = model.generate(
            context,
            do_sample=True,
            temperature=args.temperature,
            top_k=0,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tk.pad_token_id,
            eos_token_id=tk.eos_token_id,
        )

    header = (
        f"# Story samples — checkpoint={args.checkpoint} seed={args.seed} "
        f"temperature={args.temperature} n={args.n}\n"
        "# Qualitative inspection only (gate G0 removed 2026-07-20, spec 02 §8);\n"
        "# this file is the auditable record of what the checkpoint writes.\n"
    )
    blocks = [header]
    for i, row in enumerate(out):
        text = tk.decode(row[1:], skip_special_tokens=True).strip()
        blocks.append(f"--- sample {i + 1:02d} ---\n{text}\n")
    report = "\n".join(blocks)
    print(report)

    out_path = args.out or args.checkpoint / "floor1_samples.txt"
    out_path.write_text(report)
    print(f"[g0] archived to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
