"""Label-masked SFT training loop (specs/02 §6, §6.1; runs 2-4 "sft" mode).

``train_sft`` fine-tunes on span-carrying Question/Answer examples with the
loss computed on label (answer) tokens only: question/format tokens and
padding are excluded via the standard label=-100 convention. Masks are built
by ``geode.edl.masking.label_mask`` — the single mask path (specs/02 §5) — so
this trainer, the prequential loop, and the test-loss evaluator provably
agree on what a label position is. ``evaluate_sft_nll_nats`` is the matching
held-out evaluator (mean masked-label CE in nats, batch-size invariant).

Implementation notes:

- **Mirrors ``train_full`` deliberately** (same AdamW/eval/stopping/logging
  skeleton, reusing its ``_batch_stream``/``_device_type`` so epoch
  permutations are byte-identical for a given seed): the pretrain loop is
  validated and frozen (V5.21-V5.25), so the SFT deltas stay confined to
  batching examples instead of packed rows and to the masked loss.
- **Right padding, pad id 0, no attention mask** — the ``geode.edl.loop``
  convention: under causal attention a right-pad position can never influence
  the logits at earlier (label) positions, and the mask is False there, so
  padding contributes neither loss nor gradient (V5.30).
- **Span guard** (V5.31): every ``label_span`` must satisfy
  ``1 <= start < end <= len(input_ids)``, checked up front. A label at
  position 0 has no predecessor under the causal shift (the ``geode.edl``
  p=0 guard); an empty span contributes no loss; a span past the sequence
  end would silently mark padding positions as labels.
- ``masking_config_hash`` recording (specs/00 §5) stays with the launch
  script: this module has no tokenizer hash and never touches the zoo
  registry (§6.1 scope).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Protocol, Sequence

import torch
import torch.nn.functional as F

from geode.edl.masking import TaskFormat, label_mask
from geode.train.loop import TrainResult, _batch_stream, _device_type
from geode.train.stopping import ConvergenceTracker, StoppingRule

_IGNORE_INDEX = -100  # standard "not a label" target id for cross_entropy


class _SpanExample(Protocol):
    """Span-carrying example (spec 00 OQ-8): token ids + [start, end) label span."""

    input_ids: list[int]
    label_span: tuple[int, int]


def _validate_label_spans(examples: Sequence[_SpanExample]) -> None:
    """V5.31 guard: require ``1 <= start < end <= len(input_ids)`` per example."""
    for i, ex in enumerate(examples):
        start, end = ex.label_span
        n_tokens = len(ex.input_ids)
        if not (1 <= start < end <= n_tokens):
            raise ValueError(
                f"sft: example {i} has label_span ({start}, {end}) over {n_tokens} tokens; "
                "require 1 <= start < end <= len(input_ids). A label at position 0 has no "
                "predecessor under the causal shift, an empty span contributes no loss, and "
                "a span past the sequence end would silently mark padding as labels."
            )


def _padded_inputs_and_mask(
    examples: Sequence[_SpanExample], task_format: TaskFormat
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    """Right-pad ``input_ids`` to the set max length (pad id 0) + the M1 mask.

    The mask comes from ``geode.edl.masking.label_mask`` — the single mask
    path — and is row-aligned with the padded ids by construction (both pad
    to the same set max length). Spans are validated first, so the mask can
    never mark a padding position.
    """
    _validate_label_spans(examples)
    max_len = max(len(ex.input_ids) for ex in examples)
    input_ids = torch.zeros((len(examples), max_len), dtype=torch.long)
    for i, ex in enumerate(examples):
        input_ids[i, : len(ex.input_ids)] = torch.tensor(ex.input_ids, dtype=torch.long)
    mask = label_mask(examples, task_format)
    return input_ids, mask


def _mean_masked_ce_nats(
    logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean cross-entropy (nats) over label positions only (causal shift).

    Non-label positions (question, formatting, padding) get target
    ``_IGNORE_INDEX`` and are excluded from both the sum and the divisor.
    """
    vocab = logits.shape[-1]
    labels = input_ids.masked_fill(~mask, _IGNORE_INDEX)
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab),
        labels[:, 1:].reshape(-1),
        ignore_index=_IGNORE_INDEX,
        reduction="mean",
    )


def evaluate_sft_nll_nats(
    model: torch.nn.Module,
    examples: Sequence[_SpanExample],
    task_format: TaskFormat,
    *,
    batch_size: int,
    device: str,
) -> float:
    """Mean masked-label cross-entropy, in nats, over all label positions.

    Runs under ``no_grad``. Sums the per-position loss and the label-token
    count across the *whole* set before dividing once, so the value is
    invariant to ``batch_size`` (V5.30) — the same aggregation contract as
    ``evaluate_nll_nats`` (V5.19).
    """
    input_ids_all, mask_all = _padded_inputs_and_mask(examples, task_format)
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    n_labels = 0
    with torch.no_grad():
        for start in range(0, input_ids_all.shape[0], batch_size):
            ids = input_ids_all[start : start + batch_size].to(device)
            mask = mask_all[start : start + batch_size].to(device)
            logits = model(ids).logits
            vocab = logits.shape[-1]
            labels = ids.masked_fill(~mask, _IGNORE_INDEX)
            batch_loss_sum = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, vocab),
                labels[:, 1:].reshape(-1),
                ignore_index=_IGNORE_INDEX,
                reduction="sum",
            )
            loss_sum += float(batch_loss_sum.item())
            n_labels += int(mask.sum().item())
    if was_training:
        model.train()
    return loss_sum / n_labels


def train_sft(
    model: torch.nn.Module,
    train_examples: Sequence[_SpanExample],
    val_examples: Sequence[_SpanExample],
    task_format: TaskFormat,
    *,
    lr: float,
    batch_size: int,
    stopping: StoppingRule,
    eval_every: int,
    max_steps: int | None,
    grad_clip: float,
    weight_decay: float,
    betas: tuple[float, float],
    device: str,
    seed: int,
    out_dir: Path,
    precision: Literal["fp32", "bf16"] = "fp32",
) -> TrainResult:
    """Label-masked SFT: ``train_full``'s contract, loss on label tokens only.

    Everything structural is inherited verbatim from ``train_full`` (specs/02
    §6.1): AdamW at constant ``lr``, seeded per-epoch permutation with
    fixed-size drop-last batches, 1-indexed optimizer steps, eval at every
    ``step % eval_every == 0`` plus the final step (``converged`` wins the
    tie-break), incremental ``train_log.jsonl`` / ``eval_log.jsonl``, final
    ``save_pretrained`` checkpoint under ``out_dir/model``, and
    ``training_meta.json``. The deltas: data is a sequence of span-carrying
    examples (right-padded, module docstring), ``train_loss_nats`` is the
    mean masked-label CE (question/padding excluded), evals go through
    ``evaluate_sft_nll_nats(val_examples)``, and the config echo additionally
    records ``task_format``.
    """
    n_train = len(train_examples)
    if n_train < batch_size:
        raise ValueError(
            f"train_sft: train_examples has {n_train} rows, fewer than batch_size={batch_size} "
            "(drop-last would yield zero batches per epoch)"
        )
    # Validate both splits before any training or disk write (V5.31).
    input_ids_all, mask_all = _padded_inputs_and_mask(train_examples, task_format)
    _validate_label_spans(val_examples)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    tracker = ConvergenceTracker(stopping)

    step = 0
    stop_reason: Literal["converged", "max_steps"] | None = None

    train_log_path = out_dir / "train_log.jsonl"
    eval_log_path = out_dir / "eval_log.jsonl"
    with train_log_path.open("w") as train_f, eval_log_path.open("w") as eval_f:
        for batch_idx in _batch_stream(n_train, batch_size, seed):
            ids = input_ids_all[batch_idx].to(device)
            mask = mask_all[batch_idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            if precision == "bf16":
                with torch.autocast(device_type=_device_type(device), dtype=torch.bfloat16):
                    loss = _mean_masked_ce_nats(model(ids).logits, ids, mask)
            else:
                loss = _mean_masked_ce_nats(model(ids).logits, ids, mask)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            step += 1

            train_record = {
                "step": step,
                "train_loss_nats": loss.item(),
                "lr": lr,
                "grad_norm": grad_norm.item(),
                "time_unix": time.time(),
            }
            train_f.write(json.dumps(train_record) + "\n")
            train_f.flush()

            is_periodic = step % eval_every == 0
            is_capped_final = max_steps is not None and step == max_steps
            if is_periodic or is_capped_final:
                val_loss_nats = evaluate_sft_nll_nats(
                    model, val_examples, task_format, batch_size=batch_size, device=device
                )
                eval_record = {
                    "step": step,
                    "val_loss_nats": val_loss_nats,
                    "time_unix": time.time(),
                }
                eval_f.write(json.dumps(eval_record) + "\n")
                eval_f.flush()
                if tracker.update(val_loss_nats):
                    stop_reason = "converged"
                elif is_capped_final:
                    stop_reason = "max_steps"
                if stop_reason is not None:
                    break

    checkpoint_dir = out_dir / "model"
    model.save_pretrained(str(checkpoint_dir))

    result = TrainResult(
        final_step=step,
        best_val_nats=tracker.best_nats,
        min_val_nats=tracker.min_nats,
        stop_reason=stop_reason,
        checkpoint_dir=checkpoint_dir,
    )
    meta = {
        "stop_reason": result.stop_reason,
        "final_step": result.final_step,
        "best_val_nats": result.best_val_nats,
        "min_val_nats": result.min_val_nats,
        "config": {
            "lr": lr,
            "batch_size": batch_size,
            "eval_every": eval_every,
            "max_steps": max_steps,
            "grad_clip": grad_clip,
            "weight_decay": weight_decay,
            "betas": betas,
            "seed": seed,
            "precision": precision,
            "stopping": {"eps_nats": stopping.eps_nats, "k": stopping.k},
            "task_format": asdict(task_format),
        },
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
    return result
