"""Full-FT / pretrain training loop (specs/05 §6.1; Runs 1-4 "pretrain" mode).

``evaluate_nll_nats`` is the shared held-out evaluator (mean next-token CE in
nats, batch-size invariant). ``train_full`` is a thin AdamW loop over *all*
next-token positions (no label masking — that arrives with the runs-2-4
task): seeded per-epoch data order, constant LR, global-norm grad clipping,
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
) -> TrainResult:
    """Train ``model`` in place with AdamW at a constant LR (specs/05 §6.1).

    Data order is a seeded permutation of ``train_seqs`` per epoch (fixed-size,
    drop-last batches); epochs repeat until a ``ConvergenceTracker`` built
    from ``stopping`` signals a stop or ``max_steps`` optimizer updates are
    reached (``max_steps=None`` means no cap). Loss is the mean next-token
    cross-entropy (nats) over the batch; gradients are clipped to a global
    norm of ``grad_clip`` (the *pre-clip* norm is what gets logged).

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
            batch = train_seqs[batch_idx.to(train_seqs.device)].to(device)

            optimizer.zero_grad(set_to_none=True)
            if precision == "bf16":
                with torch.autocast(device_type=_device_type(device), dtype=torch.bfloat16):
                    loss = _mean_ce_nats(model(batch).logits, batch)
            else:
                loss = _mean_ce_nats(model(batch).logits, batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            step += 1

            train_record = {
                "step": step,
                "train_loss_nats": loss.item(),
                "lr": lr,
                "grad_norm": grad_norm.item(),
            }
            train_f.write(json.dumps(train_record) + "\n")
            train_f.flush()

            is_periodic = step % eval_every == 0
            is_capped_final = max_steps is not None and step == max_steps
            if is_periodic or is_capped_final:
                val_loss_nats = evaluate_nll_nats(
                    model, val_seqs, batch_size=batch_size, device=device
                )
                eval_f.write(json.dumps({"step": step, "val_loss_nats": val_loss_nats}) + "\n")
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
        stop_reason=stop_reason,
        checkpoint_dir=checkpoint_dir,
    )
    meta = {
        "stop_reason": result.stop_reason,
        "final_step": result.final_step,
        "best_val_nats": result.best_val_nats,
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
        },
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
    return result


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

    Epochs repeat until the caller stops iterating (specs/05 §6.1: "epochs
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
