"""Prequential per-batch loss + accumulator (specs/01 §1, §2 M2; specs/00 §3).

``prequential_step`` evaluates the pre-update, masked (M1) label-token
cross-entropy for one batch, under ``no_grad`` — this is the per-batch term
of the prequential MDL sum (specs/01 §1). ``PrequentialAccumulator`` collects
those per-batch terms epoch-1-only (M2: "the accumulator is structurally
incapable of adding epoch>1 records") and flushes them to
``logs/prequential.jsonl`` (specs/00 §3) via the ZOO-2 writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import torch

from geode.zoo import PrequentialRecord, write_jsonl
from geode.zoo.store import prequential_log_path


class _HasInputIds(Protocol):
    input_ids: list[int]


@dataclass(frozen=True)
class StepLoss:
    """Result of one pre-update ``prequential_step`` evaluation, in nats."""

    loss_sum_nats: float
    label_token_count: int


def prequential_step(
    model: torch.nn.Module,
    batch: Sequence[_HasInputIds],
    mask: torch.BoolTensor,
) -> StepLoss:
    """D-7: sum of masked label-token cross-entropy (nats) for one batch.

    Right-pads ``batch``'s ``input_ids`` to the batch's max length (pad id
    0), forwards the model under ``torch.no_grad()`` on the model's own
    device, and for each masked position ``p`` of example ``i`` (standard
    causal-LM shift) adds ``-log_softmax(logits[i, p-1])[input_ids[i][p]]``.
    Pre-update by contract: call this before stepping the optimizer on
    ``batch``. Leaves no gradients on ``model``.
    """
    device = next(model.parameters()).device
    max_len = max(len(ex.input_ids) for ex in batch)
    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ex in enumerate(batch):
        ids = torch.tensor(ex.input_ids, dtype=torch.long)
        input_ids[i, : ids.shape[0]] = ids
    input_ids = input_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        loss_sum_nats = 0.0
        for i in range(mask.shape[0]):
            for p in range(mask.shape[1]):
                if mask[i, p]:
                    loss_sum_nats += -log_probs[i, p - 1, input_ids[i, p]].item()

    return StepLoss(
        loss_sum_nats=float(loss_sum_nats),
        label_token_count=int(mask.sum().item()),
    )


@dataclass
class PrequentialAccumulator:
    """M2: accumulates epoch-1 ``PrequentialRecord``s; no epoch>1 entry point.

    ``add_epoch1`` is the only way to add a record and raises ``ValueError``
    for ``record.epoch != 1``, without mutating accumulated state. ``flush``
    persists the accumulated records, in insertion order, to a run's
    ``logs/prequential.jsonl`` (specs/00 §3).
    """

    _records: list[PrequentialRecord] = field(default_factory=list, init=False)

    def add_epoch1(self, record: PrequentialRecord) -> None:
        """Add ``record``; raises ``ValueError`` on ``record.epoch != 1`` (M2, D-8)."""
        if record.epoch != 1:
            raise ValueError(
                f"PrequentialAccumulator.add_epoch1: expected epoch=1, got "
                f"epoch={record.epoch} (M2: epoch>1 records are not accumulated)"
            )
        self._records.append(record)

    def mdl_nats(self) -> float:
        """Sum of ``loss_sum_nats`` over all accumulated (epoch-1) records."""
        return sum(record.loss_sum_nats for record in self._records)

    def flush(self, run_id: str, *, store: Path | None = None) -> None:
        """Write the accumulated records, in insertion order, to ``logs/prequential.jsonl``."""
        write_jsonl(prequential_log_path(run_id, store=store), self._records)
