"""Full-FT / pretrain training loop (specs/02 §6.1; Runs 1-4 "pretrain" mode).

``evaluate_nll_nats`` is the shared held-out evaluator (mean next-token CE in
nats, batch-size invariant). ``train_full`` is a thin AdamW loop over *all*
next-token positions (no label masking — the runs-2-4 SFT mode lives in
``geode.train.sft``): seeded per-epoch data order, constant or
cosine-decayed LR, global-norm grad clipping,
periodic + final-step eval against a ``ConvergenceTracker``, incremental
JSONL logs, and a final ``save_pretrained`` checkpoint. Deliberately
independent of ``geode.edl.loop`` (no ``geode.zoo`` / ``datasets`` imports)
so the validated prequential path stays untouched.

Implementation notes:

- **Simultaneous stop triggers.** Tie-break pinned in spec §6.1 (2026-07-16):
  ``stop_reason="converged"`` wins when both conditions fire on the same
  final-step eval (the run really did converge; labeling it ``max_steps``
  would misreport run health in persisted artifacts).
- **Per-epoch data order** is a permutation from a *local* ``torch.Generator``
  seeded by a deterministic function of ``(seed, epoch)`` — never the global
  RNG — so two runs with identical inputs reproduce byte-identical batch
  orders (V5.22) without disturbing ambient torch random state.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import torch
import torch.nn.functional as F

from geode.train.stopping import ConvergenceTracker, StoppingRule


@dataclass(frozen=True)
class TrainResult:
    final_step: int
    best_val_nats: float
    min_val_nats: float
    stop_reason: Literal["converged", "max_steps"]
    checkpoint_dir: Path


def evaluate_nll_nats(
    model: torch.nn.Module,
    seqs: torch.LongTensor,
    *,
    batch_size: int,
    device: str,
) -> float:
    """Mean next-token cross-entropy, in nats, over all predicted positions.

    Positions ``0..L-2`` predict tokens ``1..L-1``. Runs under ``no_grad``.
    Sums the per-position loss and the position count across the *whole* set
    before dividing once, so the result never averages per-batch means over
    unequal batches and is invariant to ``batch_size`` (V5.19).
    """
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    n_positions = 0
    with torch.no_grad():
        for start in range(0, seqs.shape[0], batch_size):
            batch = seqs[start : start + batch_size].to(device)
            logits = model(batch).logits
            vocab = logits.shape[-1]
            shift_logits = logits[:, :-1, :].reshape(-1, vocab)
            shift_targets = batch[:, 1:].reshape(-1)
            batch_loss_sum = F.cross_entropy(shift_logits, shift_targets, reduction="sum")
            loss_sum += float(batch_loss_sum.item())
            n_positions += shift_targets.numel()
    if was_training:
        model.train()
    return loss_sum / n_positions


def train_full(
    model: torch.nn.Module,
    train_seqs: torch.LongTensor,
    val_seqs: torch.LongTensor,
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
    micro_batch_size: int | None = None,
    lr_schedule: Literal["constant", "cosine"] = "constant",
    min_lr: float | None = None,
) -> TrainResult:
    """Train ``model`` in place with AdamW (specs/02 §6.1).

    Data order is a seeded permutation of ``train_seqs`` per epoch (fixed-size,
    drop-last batches); epochs repeat until a ``ConvergenceTracker`` built
    from ``stopping`` signals a stop or ``max_steps`` optimizer updates are
    reached (``max_steps=None`` means no cap). Loss is the mean next-token
    cross-entropy (nats) over the batch; gradients are clipped to a global
    norm of ``grad_clip`` (the *pre-clip* norm is what gets logged).

    ``micro_batch_size`` (default: ``batch_size``) is plain gradient
    accumulation: each optimizer step runs ``batch_size // micro_batch_size``
    sequential micro-batches whose ``1/n_micro``-scaled losses accumulate
    gradients before the single clip + update. The effective batch, the
    logged ``train_loss_nats`` (mean over the full batch — exact because
    micro-batches are equal-sized and every position counts), and all
    step/eval/stopping semantics are unchanged; only peak memory drops.
    Must divide ``batch_size`` exactly. Held-out evals also run
    ``micro_batch_size`` rows at a time (value-safe by V5.19 batch-size
    invariance).

    ``lr_schedule="cosine"`` (added 2026-07-19, run-1 gate-G0 fix) decays the
    LR along a half cosine from exactly ``lr`` at step 1 to exactly ``min_lr``
    at step ``max_steps``, set on the optimizer before every update and logged
    per step. It requires ``max_steps`` (the schedule needs a fixed horizon)
    and ``min_lr`` in ``[0, lr]``; ``min_lr`` with a constant schedule raises
    (likely a config typo). Under cosine the plateau rule is inert — decay
    shrinks late-run improvements below any sensible ``eps_nats`` by design,
    so honoring it would cut the schedule short; the run always ends at
    exactly ``max_steps`` with ``stop_reason="max_steps"`` (the tracker still
    records ``best_val_nats`` and ``min_val_nats``).

    Evaluates ``evaluate_nll_nats(val_seqs)`` at every step where
    ``step % eval_every == 0`` and additionally at the final step (deduped).
    Steps are 1-indexed optimizer updates. Writes ``train_log.jsonl`` /
    ``eval_log.jsonl`` under ``out_dir`` incrementally, saves the final model
    to ``out_dir/model`` via ``save_pretrained``, and writes
    ``out_dir/training_meta.json``.
    """
    n_train = train_seqs.shape[0]
    if n_train < batch_size:
        raise ValueError(
            f"train_full: train_seqs has {n_train} rows, fewer than batch_size={batch_size} "
            "(drop-last would yield zero batches per epoch)"
        )
    if micro_batch_size is None:
        micro_batch_size = batch_size
    if not 1 <= micro_batch_size <= batch_size or batch_size % micro_batch_size:
        raise ValueError(
            f"train_full: micro_batch_size={micro_batch_size} must lie in [1, batch_size] "
            f"and divide batch_size={batch_size} exactly (equal micro-batches keep the "
            "accumulated mean-of-means equal to the full-batch mean)"
        )
    n_micro = batch_size // micro_batch_size
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"train_full: unknown lr_schedule {lr_schedule!r}")
    if lr_schedule == "cosine":
        if max_steps is None:
            raise ValueError(
                "train_full: lr_schedule='cosine' requires max_steps — the schedule "
                "decays over a fixed horizon"
            )
        if min_lr is None or not 0.0 <= min_lr <= lr:
            raise ValueError(
                f"train_full: lr_schedule='cosine' requires min_lr in [0, lr={lr}], got {min_lr}"
            )
    elif min_lr is not None:
        raise ValueError(
            f"train_full: min_lr={min_lr} is only meaningful with lr_schedule='cosine'"
        )

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
            optimizer.zero_grad(set_to_none=True)
            train_loss_nats = 0.0
            for micro_idx in batch_idx.split(micro_batch_size):
                batch = train_seqs[micro_idx.to(train_seqs.device)].to(device)
                if precision == "bf16":
                    with torch.autocast(device_type=_device_type(device), dtype=torch.bfloat16):
                        loss = _mean_ce_nats(model(batch).logits, batch)
                else:
                    loss = _mean_ce_nats(model(batch).logits, batch)
                (loss / n_micro).backward()
                train_loss_nats += loss.item() / n_micro
            grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            step_lr = _scheduled_lr(
                step + 1, lr=lr, min_lr=min_lr, max_steps=max_steps, schedule=lr_schedule
            )
            for group in optimizer.param_groups:
                group["lr"] = step_lr
            optimizer.step()
            step += 1

            train_record = {
                "step": step,
                "train_loss_nats": train_loss_nats,
                "lr": step_lr,
                "grad_norm": grad_norm.item(),
                "time_unix": time.time(),
            }
            train_f.write(json.dumps(train_record) + "\n")
            train_f.flush()

            is_periodic = step % eval_every == 0
            is_capped_final = max_steps is not None and step == max_steps
            if is_periodic or is_capped_final:
                val_loss_nats = evaluate_nll_nats(
                    model, val_seqs, batch_size=micro_batch_size, device=device
                )
                eval_record = {
                    "step": step,
                    "val_loss_nats": val_loss_nats,
                    "time_unix": time.time(),
                }
                eval_f.write(json.dumps(eval_record) + "\n")
                eval_f.flush()
                if tracker.update(val_loss_nats, step=step) and lr_schedule == "constant":
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
            "lr_schedule": lr_schedule,
            "min_lr": min_lr,
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "eval_every": eval_every,
            "max_steps": max_steps,
            "grad_clip": grad_clip,
            "weight_decay": weight_decay,
            "betas": betas,
            "seed": seed,
            "precision": precision,
            "stopping": {
                "eps_nats": stopping.eps_nats,
                "k": stopping.k,
                "min_steps": stopping.min_steps,
            },
        },
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
    return result


def _scheduled_lr(
    step: int, *, lr: float, min_lr: float | None, max_steps: int | None, schedule: str
) -> float:
    """LR applied at 1-indexed optimizer step ``step`` (spec §6.1): constant
    ``lr``, or a half cosine from ``lr`` (step 1) to ``min_lr`` (step
    ``max_steps``). ``max_steps == 1`` degenerates to a single step at ``lr``.
    Endpoints are returned exactly (the V5.35 contract), not via the formula,
    whose float rounding can miss ``lr`` by an ulp at step 1."""
    if schedule == "constant" or max_steps == 1 or step == 1:
        return lr
    if step == max_steps:
        return min_lr
    frac = (step - 1) / (max_steps - 1)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * frac))


def _device_type(device: str) -> str:
    """Derive the ``torch.autocast`` ``device_type`` from ``device`` (e.g.
    ``"cpu"`` -> ``"cpu"``, ``"<accelerator>:0"`` -> ``"<accelerator>"``);
    never hardcodes a specific accelerator name."""
    return device.split(":")[0]


def _epoch_batches(n: int, batch_size: int, seed: int, epoch: int) -> list[torch.Tensor]:
    """Fixed-size, drop-last batches of row indices for one epoch.

    The permutation comes from a local ``torch.Generator`` seeded by a
    deterministic function of ``(seed, epoch)`` (never the global RNG).
    """
    derived_seed = (seed * 1_000_003 + epoch) % (2**63 - 1)
    generator = torch.Generator().manual_seed(derived_seed)
    perm = torch.randperm(n, generator=generator)
    n_batches = n // batch_size
    return [perm[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)]


def _batch_stream(n: int, batch_size: int, seed: int) -> Iterator[torch.Tensor]:
    """Yield row-index batches epoch after epoch, indefinitely.

    Epochs repeat until the caller stops iterating (specs/02 §6.1: "epochs
    repeat until a stop condition fires").
    """
    epoch = 0
    while True:
        yield from _epoch_batches(n, batch_size, seed, epoch)
        epoch += 1


def _mean_ce_nats(logits: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy (nats) over one batch (causal shift)."""
    vocab = logits.shape[-1]
    shift_logits = logits[:, :-1, :].reshape(-1, vocab)
    shift_targets = batch[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_targets, reduction="mean")
