"""Shared plumbing for the Phase-0 mechanistic-interpretability drivers
(``logit_lens.py``, ``weight_diff.py``, ``resid_shift.py``): model loading
that accepts either a zoo-registered run or a plain HF directory, the
residual-stream forward-only capture both logit_lens.py and resid_shift.py
need, and the char-span -> label-position bridge from ``geode.arith.spans``.

Residual-hook layer convention (pin this here, not per script, so a later
session cross-reading these tables doesn't get burned): ``capture_residuals``
names its rows the ``geode.probe.extract.residual_hook_names`` way —
``hook_embed`` is layer 0, ``blocks.{i}.hook_resid_post`` is layer ``i + 1``.
``logit_lens.py`` and ``resid_shift.py`` both report ``layer`` in this sense.
``weight_diff.py`` reports a DIFFERENT thing under the same column name — the
transformer block index a weight tensor lives in (``-1`` for
``embed_tokens``) — because it never touches the residual stream; see its
own module docstring.

Sibling-import convention: these three drivers ``import mech_lib`` as a bare
module (matching ``emergence.py``'s ``from steering import ...`` — the
launcher contract is ``cd analysis/ && python3 <script>.py``, and
``tests/_scriptloader.py`` puts this directory on ``sys.path`` for tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import nn

from geode.arith.spans import SftExample, tokenize_with_spans
from geode.probe import residual_hook_names
from geode.zoo import load_model


def load_any_model(spec: str, *, device: str, store: Path | None = None) -> nn.Module:
    """Load a model from ``run:<run_id>`` (``geode.zoo.load_model``, method-
    dispatched per the run's manifest) or ``dir:<path>`` (plain HF
    ``from_pretrained`` — full-FT parents, their ``sft_snapshots/step_*/``,
    and the base all load this way). Returns the model on ``device`` in eval
    mode.
    """
    if spec.startswith("run:"):
        return load_model(spec.removeprefix("run:"), store=store, device=device)
    if spec.startswith("dir:"):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(spec.removeprefix("dir:"))
        return model.to(device).eval()
    raise ValueError(f"model spec must start with 'run:' or 'dir:', got {spec!r}")


def residual_modules(model: nn.Module) -> tuple[nn.Module, Sequence[nn.Module]]:
    """The embedding module + decoder-block list of a Llama-style causal LM.

    Forward-only local twin of ``geode.probe.extract._residual_modules``
    (private there, and this call site never needs gradients, so it stays a
    small standalone helper rather than importing a private symbol).
    """
    embed = model.get_input_embeddings()
    decoder = getattr(model, "model", None)
    blocks = getattr(decoder, "layers", None)
    if embed is None or blocks is None:
        raise ValueError(
            "residual_modules: expected a Llama-style causal LM with "
            f"get_input_embeddings() and model.layers; got {type(model).__name__}"
        )
    return embed, blocks


def capture_residuals(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Forward-only residual-stream capture at every ``hook_embed`` /
    ``blocks.{i}.hook_resid_post`` point (module docstring's layer
    convention), no gradients. Hooks are removed before returning even on
    error.
    """
    embed, blocks = residual_modules(model)
    names = residual_hook_names(len(blocks))
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def embed_hook(_module, _inputs, output):
        captured["hook_embed"] = output

    def make_block_hook(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output[0] if isinstance(output, tuple) else output

        return hook

    handles.append(embed.register_forward_hook(embed_hook))
    for i, block in enumerate(blocks):
        handles.append(block.register_forward_hook(make_block_hook(names[i + 1])))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask.long())
    finally:
        for h in handles:
            h.remove()
    missing = [n for n in names if n not in captured]
    if missing:
        raise RuntimeError(f"capture_residuals: hooks never fired for {missing}")
    return {n: captured[n] for n in names}


def final_norm_and_head(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """``(model.model.norm, model.lm_head)``, or a clear error naming the
    model type instead of an ``AttributeError`` deep in a matmul (both stay
    plain modules even on a LoRA-wrapped model: neither is in
    ``geode.train.lora.LORA_TARGET_MODULES``).
    """
    decoder = getattr(model, "model", None)
    norm = getattr(decoder, "norm", None)
    head = getattr(model, "lm_head", None)
    if norm is None or head is None:
        raise ValueError(
            "final_norm_and_head: expected a Llama-style causal LM with "
            f"model.model.norm and model.lm_head; got {type(model).__name__}"
        )
    return norm, head


def load_task_examples(
    df: pd.DataFrame, tokenizer, *, limit: int | None = None
) -> list[SftExample]:
    """Tokenize a task parquet's ``full_text`` + answer char span rows via
    the shared ``geode.arith.spans`` bridge (``append_eos=True``, the
    training convention)."""
    if limit is not None:
        df = df.iloc[:limit]
    char_spans = list(zip(df["answer_char_start"].astype(int), df["answer_char_end"].astype(int)))
    return tokenize_with_spans(df["full_text"].tolist(), char_spans, tokenizer, append_eos=True)


def pad_examples(
    examples: Sequence[SftExample], *, pad_id: int = 0, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad ``input_ids`` to the batch max (pad id 0 — the geode.edl
    convention, ``geode.probe.extract._padded_batch``); returns
    ``(input_ids, attention_mask)``.
    """
    max_len = max(len(ex.input_ids) for ex in examples)
    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.bool)
    for i, ex in enumerate(examples):
        ids = torch.tensor(ex.input_ids, dtype=torch.long)
        input_ids[i, : ids.shape[0]] = ids
        attention_mask[i, : ids.shape[0]] = True
    return input_ids.to(device), attention_mask.to(device)


def load_generic_texts(path: Path, *, limit: int | None = None) -> list[str]:
    """Generic held-out text examples from a ``.txt`` (one example per
    non-blank line) or ``.parquet`` (a ``text`` column) file."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if "text" not in df.columns:
            raise ValueError(f"{path}: parquet has no 'text' column (got {list(df.columns)})")
        texts = [str(t) for t in df["text"].tolist()]
    else:
        texts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if limit is not None:
        texts = texts[:limit]
    if not texts:
        raise ValueError(f"{path}: no usable text examples found")
    return texts


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path``: parquet if the suffix says so, else CSV.

    Not ``geode.zoo.write_results`` — that writer requires the ZOO-4 8-column
    schema (``regime``, ``dataset_size``, ...) sourced from a run manifest,
    which a ``dir:``-loaded plain HF checkpoint (e.g. the base model) has no
    manifest to supply. These three drivers compare arbitrary model pairs,
    zoo-registered or not, so they write plain tables instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path, index=False)
