"""Label masking (M1) + masking config hash (specs/01 §2, §3; specs/00 §5).

M1 (spec 01 §2): loss is computed on label tokens only, never prompt or
formatting tokens. ``label_mask`` is the single shared construction path from
a batch of span-carrying examples to a boolean mask — both the training loop
and the test-loss evaluator build their masks here, and nowhere else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

import torch


class _LabeledExample(Protocol):
    input_ids: list[int]
    label_span: tuple[int, int]


@dataclass(frozen=True)
class TaskFormat:
    """Identifies a task's label-span rule for the mask-parity hash (OQ-8).

    ``label_mask`` builds masks from each example's own ``label_span``, not
    from ``TaskFormat`` (spans are supplied per-example by the dataset
    builder). ``span_source`` documents which rule produced those spans; it
    exists so that a train/test mismatch in the span rule still shows up in
    ``masking_config_hash`` (spec 00 §5), even though it plays no role in
    ``label_mask`` itself.
    """

    name: str
    format_version: str
    span_source: str = "dataset_builder"


def label_mask(batch: Sequence[_LabeledExample], task_format: TaskFormat) -> torch.BoolTensor:
    """M1: True at each example's ``[start, end)`` label-span positions.

    False everywhere else, including prompt tokens, formatting tokens, and
    padding positions of shorter sequences in the batch. ``task_format`` is
    accepted for API stability (OQ-8: the same signature also feeds
    ``masking_config_hash``) but is not consulted — spans are per-example.
    """
    del task_format
    max_len = max(len(ex.input_ids) for ex in batch)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, ex in enumerate(batch):
        start, end = ex.label_span
        mask[i, start:end] = True
    return mask


def masking_config_hash(task_format: TaskFormat, tokenizer_hash: str) -> str:
    """OQ-7: sha256 hex digest of the canonical JSON mask-rule config.

    Canonicalized as sorted-key, compact-separator JSON of ``task_format``'s
    fields (name, format_version, and every mask-rule parameter) plus
    ``tokenizer_hash`` — deterministic across processes and sensitive to
    every component.
    """
    payload = {**asdict(task_format), "tokenizer_hash": tokenizer_hash}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
