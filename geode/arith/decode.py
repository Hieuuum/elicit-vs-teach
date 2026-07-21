"""Greedy token-prefix decoding shared by the gates and the in-loop G4 eval.

Promoted from ``experiments/training-run/scripts/gates.py`` 2026-07-21 (the
promotion rule: the same decode now backs the G1/G2 gate evals *and* the
format installers' in-loop stopping signal, spec 02 §6, and its silent
failure would invalidate either).

Protocol (spec 02 §8; the 2026-07-21 G1=0 incident): prompts are
**token-level prefixes of the training tokenization**
(``input_ids[:label_span.start]``), never re-tokenized char slices — a
re-tokenized char-sliced prompt ends in a standalone space token the model
never saw in training and makes the merged `` -`` sign token unreachable.
Decoding is greedy, stops at the trained EOS (V5.43), and returns the first
line of each completion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@torch.no_grad()
def greedy_completions(
    model: torch.nn.Module,
    tokenizer: "PreTrainedTokenizerBase",
    prompt_ids: list[list[int]],
    *,
    device: str,
    batch_size: int,
    max_new_tokens: int = 12,
) -> list[str]:
    """First generated line per token-prefix prompt (greedy, left-padded, EOS-stopped)."""
    model.eval()
    tokenizer.padding_side = "left"
    out: list[str] = []
    for start in range(0, len(prompt_ids), batch_size):
        enc = tokenizer.pad(
            {"input_ids": prompt_ids[start : start + batch_size]},
            padding=True,
            return_tensors="pt",
        ).to(device)
        generated = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[:, enc["input_ids"].shape[1] :]
        for row in tokenizer.batch_decode(new_tokens, skip_special_tokens=True):
            out.append(row.split("\n")[0])
    return out
