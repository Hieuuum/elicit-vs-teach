"""Row-masked input-embedding warm-start training.

The language-model head is untied before an input row moves. Otherwise a tied
row is also an output classifier row and receives gradients even where its
token is absent from the prompt, which changes the intervention being tested.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from geode.edl.masking import TaskFormat, label_mask
from geode.train.sft import _mean_masked_ce_nats

WARMSTART_TOKEN_IDS = {"Ġs": 261, "um": 492, ":": 27}


class _SpanExample(Protocol):
    input_ids: list[int]
    label_span: tuple[int, int]


@dataclass(frozen=True)
class EmbeddingTrainResult:
    """Outcome of one fixed-dose embedding intervention."""

    final_loss_nats: float
    row_delta_l2: dict[int, float]
    moved_rows: tuple[int, ...]


def select_warmstart_candidate(candidates: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    """Select the best held-out NL candidate without consulting target data or EDL."""
    if not candidates:
        raise ValueError("embedding warm-start: candidates must be non-empty")
    return min(
        candidates,
        key=lambda candidate: (
            -candidate["nl_accuracy"],
            candidate["nl_loss_nats"],
            candidate["lr"],
        ),
    )


def resolve_warmstart_rows(tokenizer, tokens: Sequence[str]) -> tuple[int, ...]:
    """Resolve and pin the three token rows allowed by the Phase-3 warm-start."""
    rows = []
    for token in tokens:
        if token not in WARMSTART_TOKEN_IDS:
            raise ValueError(f"embedding warm-start: unsupported token {token!r}")
        row = tokenizer.convert_tokens_to_ids(token)
        expected = WARMSTART_TOKEN_IDS[token]
        if row != expected:
            raise ValueError(
                f"embedding warm-start: token {token!r} is row {row}, expected {expected}"
            )
        rows.append(int(row))
    if len(rows) != len(set(rows)):
        raise ValueError("embedding warm-start: duplicate token rows")
    return tuple(rows)


def untie_lm_head(model: torch.nn.Module) -> None:
    """Give ``lm_head`` a frozen copy of the input embedding matrix."""
    embedding = model.get_input_embeddings().weight
    if model.lm_head.weight.data_ptr() == embedding.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(embedding.detach().clone())
        model.config.tie_word_embeddings = False
    model.lm_head.weight.requires_grad_(False)
    if model.lm_head.weight.data_ptr() == embedding.data_ptr():
        raise ValueError("embedding warm-start: lm_head is still tied")


def assert_untie_is_noop(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    *,
    device: str,
) -> None:
    """Untie ``lm_head`` and require a bitwise-identical forward pass."""
    model.to(device).eval()
    input_ids = input_ids.to(device)
    with torch.no_grad():
        before = model(input_ids).logits.clone()
    untie_lm_head(model)
    model.to(device)
    with torch.no_grad():
        after = model(input_ids).logits
    if not torch.equal(before, after):
        drift = float((before - after).abs().max().item())
        raise ValueError(f"embedding warm-start: untie changed logits (max |delta|={drift})")


def train_embedding_rows(
    model: torch.nn.Module,
    examples: Sequence[_SpanExample],
    task_format: TaskFormat,
    *,
    rows: Sequence[int],
    lr: float,
    steps: int,
    micro_batch_size: int,
    device: str,
    betas: tuple[float, float] = (0.9, 0.999),
    grad_clip: float | None = None,
    seed: int,
) -> EmbeddingTrainResult:
    """Train only ``rows`` of the untied input embedding on a fixed SFT dose."""
    rows = tuple(int(row) for row in rows)
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("embedding warm-start: rows must be non-empty and unique")
    if steps <= 0 or micro_batch_size <= 0:
        raise ValueError("embedding warm-start: steps and micro_batch_size must be positive")
    if not examples:
        raise ValueError("embedding warm-start: examples must be non-empty")

    torch.manual_seed(seed)
    model.to(device)
    embedding = model.get_input_embeddings().weight
    if model.lm_head.weight.data_ptr() == embedding.data_ptr():
        raise ValueError("embedding warm-start: untie lm_head before training")
    if min(rows) < 0 or max(rows) >= embedding.shape[0]:
        raise ValueError(
            f"embedding warm-start: row outside vocabulary of size {embedding.shape[0]}"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embedding.requires_grad_(True)
    original = embedding.detach().clone()

    max_len = max(len(example.input_ids) for example in examples)
    input_ids = torch.zeros((len(examples), max_len), dtype=torch.long, device=device)
    for i, example in enumerate(examples):
        input_ids[i, : len(example.input_ids)] = torch.tensor(example.input_ids, device=device)
    mask = label_mask(examples, task_format).to(device)

    row_mask = torch.zeros((embedding.shape[0], 1), device=device)
    row_mask[list(rows)] = 1.0
    hook = embedding.register_hook(lambda gradient: gradient * row_mask)
    optimizer = torch.optim.AdamW([embedding], lr=lr, betas=betas, weight_decay=0.0)
    model.train()
    final_loss = float("nan")
    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            final_loss = 0.0
            for start in range(0, len(examples), micro_batch_size):
                ids = input_ids[start : start + micro_batch_size]
                batch_mask = mask[start : start + micro_batch_size]
                loss = _mean_masked_ce_nats(model(ids).logits, ids, batch_mask)
                fraction = ids.shape[0] / len(examples)
                (loss * fraction).backward()
                final_loss += float(loss.item()) * fraction
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([embedding], grad_clip)
            optimizer.step()
    finally:
        hook.remove()

    row_delta = (embedding.detach() - original).norm(dim=1)
    moved = tuple(row_delta.nonzero().flatten().tolist())
    unexpected = set(moved) - set(rows)
    if unexpected:
        raise RuntimeError(
            f"embedding warm-start: rows outside {sorted(rows)} moved: {sorted(unexpected)[:10]}"
        )
    return EmbeddingTrainResult(
        final_loss_nats=final_loss,
        row_delta_l2={row: float(row_delta[row].item()) for row in rows},
        moved_rows=moved,
    )
