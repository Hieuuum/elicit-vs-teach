"""Prequential training-loop wrapper around the pinned LoRA adapter
(specs/01 §3; specs/00 §1–§5; specs/02 §6 "LoRA (runs 5-6 only)").

``train_prequential`` is the thin harness that turns a pretrained model + an
ordered dataset into the four run artifacts of specs/00: per-batch
``logs/prequential.jsonl`` (§3), per-step ``logs/gradstats.jsonl`` (§4),
full-model snapshots at the manifest's ``snapshot_steps`` (§1–§2), and the final
held-out ``eval/test_loss.json`` (§5). It is deliberately thin: the per-batch
pre-update loss routes through the shared ``prequential_step`` (``no_grad``, so
the logged evaluation can never contaminate the gradient of the update). The
epoch-1 (MDL-bearing) stream is accumulated in and written from
``PrequentialAccumulator``, so its M2 structural guard actually gates the
artifact; later-epoch records are logged and flagged alongside (L-8). Masking
routes through the shared ``label_mask``/``masking_config_hash`` so train and
test masks provably agree (M1, specs/00 §5).

The adapter is ``geode.train.lora.apply_lora`` — the spec-02 §6 pin (scaling
α/(2r), deliberately not PEFT's α/r; A init uniform ±1/√d_in through a
dedicated generator seeded by the explicit ``seed``; B zero, so θ_0 computes
the pretrained function exactly, specs/01 §1). The 2026-07-21 rewire replaced
the earlier internal PEFT ``get_peft_model`` path, whose α/r scaling
conflicted with the pin; the manifest's ``training.lora`` block still drives
rank/alpha/target_modules/dropout.

Design decisions where the spec is silent (all documented, none pinned by the
EDL-3 tests):

- **Batch order** is sequential and identical every epoch: examples are
  presented in a fixed order (specs/01 §1 "presented in a fixed order"). This
  enumerates each unique example exactly once in epoch 1 (spec 00 §3 invariant)
  and needs no RNG, so ``seed`` drives only the adapter initialisation.
- **Training objective** is the *mean* masked-label cross-entropy per batch
  (L-7 leaves this unpinned; sum would descend too). The value written to
  ``prequential.jsonl`` is always the shared ``prequential_step`` sum, never
  this training scalar.
- **Snapshot semantics** (L-5/A-1/A-2, format revised 2026-07-22):
  ``snapshots/step_{k}/adapter.safetensors`` holds exactly the trainable
  (adapter) tensors after ``k`` optimizer updates — the same θ_k under which
  batch ``k``'s pre-update loss is recorded; the frozen base + buffers are
  written once per run to ``snapshots/base/model.safetensors``.
  ``load_snapshot`` reassembles θ_k bit-exactly (and still strict-loads
  legacy pre-2026-07-22 full ``model.safetensors`` snapshots). ``k`` may
  equal the total update count ``T`` (the final state θ_T, taken after the
  last update). A declared ``k`` outside ``0..T`` is unreachable and
  rejected up front (spec 00 §2).
- **``per_module_grad_norm``** is keyed by ``module_name`` (spec 00 §4, A-3):
  each trainable parameter's grad is attributed to its owning module and a
  module's entry is the sqrt of its parameters' summed squared grad norms, so
  every entry is a sub-norm of the global norm.
- **Optimizer**: SGD (spec 00 §2 ``optimizer.name == "sgd"``); any other name
  raises rather than silently substituting.
- **p=0 guard** (deferred EDL-2 item, ``prequential.py`` silent wraparound): a
  causal LM predicts token ``p`` from position ``p-1``, so a label at position 0
  has no predecessor. The loop refuses such a mask with a clear error instead of
  letting ``prequential_step`` index ``[-1]`` and silently wrap.
- **``step_callback``** (2026-07-21, runs 5-6 launch surface — the "small
  logging extension" of specs/02 §6): an optional per-update hook receiving a
  ``PrequentialStepInfo`` (step, epoch, lr, training loss, label-token
  accuracy — the last two computed at the pre-update θ from the update
  forward's own logits, no extra pass). Returning ``True`` stops training
  after that update; every artifact is still written (accumulator flush,
  gradstats, the final-state snapshot if scheduled, test loss at the stopped
  θ_T). An early stop inside epoch 1 truncates the MDL stream to the seen
  prefix — the spec-00 §3 enumeration invariant then holds over that prefix
  only, which is the runs-5/6 design (their stopping rule is part of the EDL
  metric). ``step_callback=None`` reproduces the prior behavior exactly.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from safetensors.torch import load_file, load_model, save_file

from geode.edl.masking import TaskFormat, label_mask, masking_config_hash
from geode.edl.prequential import PrequentialAccumulator, prequential_step
from geode.train.lora import apply_lora
from geode.zoo import (
    GradStatRecord,
    PrequentialRecord,
    RunManifest,
    TestLoss,
    write_jsonl,
)
from geode.zoo.records import _serialize_record
from geode.zoo.store import (
    gradstats_log_path,
    manifest_path,
    prequential_log_path,
    run_dir,
    test_loss_path,
)


@dataclass(frozen=True)
class PrequentialStepInfo:
    """Per-update scalars handed to ``step_callback`` (specs/02 §6 extension).

    ``step`` counts completed optimizer updates (1 after the first update, so
    it names the θ the run has just reached); ``train_loss_nats`` is the
    masked-mean training objective and ``train_accuracy`` the argmax accuracy
    over the batch's label tokens, both computed at the pre-update θ (the same
    parameters under which the batch's prequential loss was recorded).
    """

    step: int
    epoch: int
    lr: float
    train_loss_nats: float
    train_accuracy: float


def train_prequential(
    model: torch.nn.Module,
    dataset: dict,
    task_format: TaskFormat,
    manifest: RunManifest,
    *,
    device: str,
    seed: int,
    gradstats_stride: int = 1,
    store: Path | None = None,
    step_callback: Callable[[PrequentialStepInfo], bool] | None = None,
) -> None:
    """Train ``model`` prequentially and write all four specs/00 run artifacts.

    ``dataset`` is ``{"train": [...], "test": [...], "tokenizer_hash": str}``
    where each split is a sequence of span-carrying examples (``input_ids`` +
    ``label_span``). The run is resolved from ``manifest.data["run_id"]`` under
    ``store`` (or ``$GEODE_STORE``); the caller registers it first (ZOO-1).

    Per batch, in θ_k (k = updates so far): save a snapshot if ``k`` is declared,
    record the pre-update loss via the shared ``prequential_step`` (``no_grad``),
    then run one grad-enabled update. Epoch-1 records are mandatory; later epochs
    are logged and flagged by their ``epoch`` field. Device-agnostic, seeded,
    single-device (specs/01 §5). ``step_callback`` (module docstring) is called
    once after every update; returning ``True`` ends training there, with all
    artifacts still written (test loss at the stopped θ_T).
    """
    run_id: str = manifest.data["run_id"]
    training = manifest.data["training"]
    optimizer_cfg = training["optimizer"]
    lora_cfg = training["lora"]
    batch_size = int(optimizer_cfg["batch_size"])
    epochs_total = int(training["epochs_total"])
    snapshot_steps = set(manifest.data["snapshot_steps"])

    train_examples = dataset["train"]
    test_examples = dataset["test"]
    tokenizer_hash = dataset["tokenizer_hash"]

    # A-2/spec 00 §2: the snapshot schedule is declared up front because a
    # missing checkpoint is the expensive failure mode. Validate it before any
    # training (or disk write) happens: with fixed-order, no-drop-last batches
    # the run performs T = ceil(n_train / batch_size) * epochs_total optimizer
    # updates, and under L-5/A-1 the reachable snapshot ks are exactly 0..T
    # (step_T = the final state θ_T). A declared step outside [0, T] can never
    # be honoured, so refuse it loudly instead of silently dropping it.
    total_updates = math.ceil(len(train_examples) / batch_size) * epochs_total
    unreachable = sorted(k for k in snapshot_steps if k < 0 or k > total_updates)
    if unreachable:
        raise ValueError(
            f"train_prequential: manifest.snapshot_steps {unreachable} cannot be "
            f"reached — training performs {total_updates} optimizer updates, so the "
            f"only snapshots that can be taken are step_0..step_{total_updates} "
            f"(spec 00 §2: a missing checkpoint is the expensive failure mode)."
        )

    # M1/§5 parity: the single mask-config hash stamped on both the manifest
    # (D-1 extra) and eval/test_loss.json, so the ZOO-2 guard and edl_nats agree.
    mask_hash = masking_config_hash(task_format, tokenizer_hash)
    manifest.data["masking_config_hash"] = mask_hash
    manifest.save(manifest_path(run_id, store=store))

    # Adapter init is the only randomness; ``apply_lora`` draws it from a
    # dedicated generator seeded by the explicit argument, so nothing leaks
    # from ambient RNG (V1.7). B is zero-initialised, so θ_0 computes the
    # pretrained function exactly (specs/01 §1: θ_0 is the pretrained state;
    # V5.47). ``apply_lora`` mutates ``model`` in place and returns it.
    wrapped = apply_lora(
        model,
        rank=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        seed=seed,
        target_modules=tuple(lora_cfg["target_modules"]),
        dropout=lora_cfg["dropout"],
    )
    wrapped.to(device)
    wrapped.eval()  # dropout=0 -> eval/train numerics identical; matches refs
    optimizer = _build_optimizer(wrapped, optimizer_cfg)
    grad_clip = optimizer_cfg.get("grad_clip")

    # Adapter-only snapshots (2026-07-22): the frozen base + buffers are
    # written once, before any step file, so a partially written run is
    # always reassemblable.
    trainable_names = {n for n, p in wrapped.named_parameters() if p.requires_grad}
    if snapshot_steps:
        _save_base(wrapped, run_id, trainable_names, store=store)

    # Epoch-1 records — the MDL-bearing stream — flow through the accumulator
    # and are written from it (below), so its M2 guard (rejects epoch != 1)
    # structurally protects the artifact. Later-epoch records are logged and
    # flagged by their epoch field (L-8) but never enter the accumulator.
    accumulator = PrequentialAccumulator()
    later_records: list[PrequentialRecord] = []
    gradstat_log: list[GradStatRecord] = []

    updates_done = 0
    stop_requested = False
    for epoch in range(1, epochs_total + 1):
        for example_ids in _batch_indices(len(train_examples), batch_size):
            batch = [train_examples[i] for i in example_ids]

            if updates_done in snapshot_steps:
                _save_snapshot(wrapped, run_id, updates_done, trainable_names, store=store)

            mask = _checked_label_mask(batch, task_format)

            # Pre-update (θ_k) loss via the shared no_grad path (specs/01 §1).
            step_loss = prequential_step(wrapped, batch, mask)
            record = PrequentialRecord(
                step=updates_done,
                epoch=epoch,
                example_ids=list(example_ids),
                label_token_count=step_loss.label_token_count,
                loss_sum_nats=step_loss.loss_sum_nats,
            )
            if epoch == 1:
                accumulator.add_epoch1(record)  # M2 structural guard
            else:
                later_records.append(record)

            # One grad-enabled update on the same batch (θ_k -> θ_{k+1}).
            optimizer.zero_grad(set_to_none=True)
            loss, accuracy = _masked_mean_loss(wrapped, batch, mask, device)
            loss.backward()
            if updates_done % gradstats_stride == 0:
                gradstat_log.append(_gradstat(wrapped, updates_done))  # pre-clip norms
            if grad_clip is not None:
                trainable = [p for p in wrapped.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            updates_done += 1

            if step_callback is not None:
                info = PrequentialStepInfo(
                    step=updates_done,
                    epoch=epoch,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    train_loss_nats=float(loss.item()),
                    train_accuracy=accuracy,
                )
                if step_callback(info):
                    stop_requested = True
                    break
        if stop_requested:
            break

    # A-1: the final state θ_T is only reachable after the last update.
    if updates_done in snapshot_steps:
        _save_snapshot(wrapped, run_id, updates_done, trainable_names, store=store)

    # Write the mandatory epoch-1 stream from the accumulator (M2-protected),
    # then append any flagged later-epoch records to the same log (spec 00 §3).
    accumulator.flush(run_id, store=store)
    if later_records:
        _append_records(prequential_log_path(run_id, store=store), later_records)
    write_jsonl(gradstats_log_path(run_id, store=store), gradstat_log)
    _write_test_loss(wrapped, test_examples, task_format, mask_hash, run_id, store=store)


def _build_optimizer(model: torch.nn.Module, optimizer_cfg: dict) -> torch.optim.Optimizer:
    """Build the optimizer over the trainable (adapter) parameters (spec 00 §2).

    ``sgd`` (the EDL-3 test fixture optimizer, L-7) and ``adamw`` (the runs-5/6
    recipe, specs/02 §6: AdamW betas + weight decay) are supported; any other
    name raises rather than silently substituting.
    """
    name = optimizer_cfg["name"]
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=optimizer_cfg["lr"],
            weight_decay=optimizer_cfg["weight_decay"],
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=optimizer_cfg["lr"],
            betas=tuple(optimizer_cfg["betas"]),
            weight_decay=optimizer_cfg["weight_decay"],
        )
    raise ValueError(
        f"train_prequential: unsupported optimizer name {name!r} (expected 'sgd' or 'adamw')"
    )


def _batch_indices(n: int, batch_size: int) -> list[list[int]]:
    """Sequential, no-drop-last batches of ``0..n-1`` (fixed order, specs/01 §1)."""
    return [list(range(start, min(start + batch_size, n))) for start in range(0, n, batch_size)]


def _checked_label_mask(batch: Sequence, task_format: TaskFormat) -> torch.BoolTensor:
    """Shared M1 mask with a p=0 guard (deferred EDL-2 wraparound fix)."""
    mask = label_mask(batch, task_format)
    if bool(mask[:, 0].any()):
        raise ValueError(
            "train_prequential: a label span includes position 0, but a causal LM "
            "predicts token p from position p-1, so position 0 has no predecessor "
            "(this would silently wrap to the last position). Label spans must "
            "start at index >= 1."
        )
    return mask


def _padded_input_ids(batch: Sequence, device: str) -> torch.Tensor:
    """Right-pad a batch's ``input_ids`` to its max length (pad id 0), on ``device``."""
    max_len = max(len(ex.input_ids) for ex in batch)
    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ex in enumerate(batch):
        ids = torch.tensor(ex.input_ids, dtype=torch.long)
        input_ids[i, : ids.shape[0]] = ids
    return input_ids.to(device)


def _masked_mean_loss(
    model: torch.nn.Module,
    batch: Sequence,
    mask: torch.BoolTensor,
    device: str,
) -> tuple[torch.Tensor, float]:
    """Grad-enabled mean cross-entropy over label tokens only (training objective).

    Mirrors the causal-LM shift of ``prequential_step`` (label at position ``p``
    predicted from position ``p-1``) so the gradient is taken at the same θ under
    which the pre-update loss was logged. Also returns the argmax accuracy over
    the batch's label tokens (a ``no_grad`` by-product of the same forward, for
    ``step_callback`` logging — specs/02 §6 "train-acc scalars per step").
    """
    input_ids = _padded_input_ids(batch, device)
    logits = model(input_ids=input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    token_ll = log_probs[:, :-1, :].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    label = mask[:, 1:].to(device=device, dtype=token_ll.dtype)
    loss = -(token_ll * label).sum() / label.sum()
    with torch.no_grad():
        hits = (logits[:, :-1, :].argmax(dim=-1) == input_ids[:, 1:]).to(label.dtype)
        accuracy = float(((hits * label).sum() / label.sum()).item())
    return loss, accuracy


def _gradstat(model: torch.nn.Module, step: int) -> GradStatRecord:
    """Per-step gradient statistics (spec 00 §4; A-3; OQ-14: overlap always null).

    ``per_module_grad_norm`` is keyed by ``module_name`` (spec 00 §4), not by
    parameter-tensor name: each trainable parameter's grad is attributed to its
    owning module (its name minus the trailing ``.weight``/``.bias`` attribute),
    and a module's norm is the sqrt of the sum of its parameters' squared grad
    norms. That aggregation is a sub-norm of the global norm, so every entry is
    <= ``global_grad_norm``.
    """
    module_sq: dict[str, float] = {}
    total_sq = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        param_sq = float(param.grad.detach().pow(2).sum().item())
        module_name = name.rsplit(".", 1)[0]  # drop the .weight/.bias tensor attr
        module_sq[module_name] = module_sq.get(module_name, 0.0) + param_sq
        total_sq += param_sq
    return GradStatRecord(
        step=step,
        global_grad_norm=math.sqrt(total_sq),
        per_module_grad_norm={name: math.sqrt(sq) for name, sq in module_sq.items()},
        topk_grad_subspace_overlap=None,
    )


def _append_records(path: Path, records: Sequence[PrequentialRecord]) -> None:
    """Append later-epoch records to an existing ``prequential.jsonl`` (L-8).

    Serialises through the shared ``_serialize_record`` — the same helper the
    accumulator's ``flush`` (via ``write_jsonl``) uses — so epoch-1 and
    later-epoch lines share one line format and the log stays byte-deterministic
    (V1.7); a future format change touches a single place.
    """
    with Path(path).open("a") as f:
        for record in records:
            f.write(_serialize_record(record))


def _save_base(
    model: torch.nn.Module, run_id: str, trainable: set[str], *, store: Path | None
) -> None:
    """Save the frozen complement of the adapter once per run (specs/00 §1).

    ``snapshots/base/model.safetensors`` holds every non-trainable state-dict
    tensor (frozen base params + buffers). Tied aliases (e.g. ``lm_head.weight``
    ↔ ``embed_tokens.weight``) share storage and are stored once — safetensors
    refuses shared tensors — and ``load_snapshot`` restores the dropped twin,
    the same convention as ``zoo.load_model``.
    """
    state = model.state_dict()
    seen: set[int] = set()
    base: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name in trainable:
            continue
        ptr = tensor.data_ptr()
        if ptr in seen:
            continue  # tied alias — restored from its twin on load
        seen.add(ptr)
        base[name] = tensor
    base_dir = run_dir(run_id, store=store) / "snapshots" / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    save_file(base, str(base_dir / "model.safetensors"))


def _save_snapshot(
    model: torch.nn.Module, run_id: str, step: int, trainable: set[str], *, store: Path | None
) -> None:
    """Save θ_step's adapter — the trainable tensors only (specs/00 §1).

    ``snapshots/step_{step}/adapter.safetensors`` holds exactly the trainable
    (A/B) tensors, unmerged — reload stays bit-exact (L-5; merging would
    perturb weights by float rounding) and the LoRA factors stay readable for
    the adapter-diff analysis. The frozen base lives once per run in
    ``snapshots/base/`` (``_save_base``): the 2026-07-18 self-contained
    decision was priced at ~77 MB/snapshot, but fp32 base+adapter at the
    runs-5/6 scale is ~203 MB × a 1024-step schedule — duplicating frozen
    tensors ~880× per run (owner 2026-07-22, decisions.md).
    """
    state = model.state_dict()
    snap_dir = run_dir(run_id, store=store) / "snapshots" / f"step_{step}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: state[k] for k in sorted(trainable)}, str(snap_dir / "adapter.safetensors"))


def load_snapshot(
    model: torch.nn.Module, run_id: str, step: int, *, store: Path | None = None
) -> torch.nn.Module:
    """Rebuild θ_step in ``model`` from its saved snapshot, bit-exactly (L-5).

    ``model`` must be the module tree the loop trained (base arch +
    ``apply_lora`` wrap mirroring the manifest); every tensor value is then
    overwritten. New-format snapshots load ``snapshots/base/model.safetensors``
    (once-per-run frozen state) plus the step's ``adapter.safetensors``; tied
    aliases dropped by the base save are restored from the twin they share
    storage with before a strict load. Legacy full snapshots
    (``model.safetensors``, pre-2026-07-22) strict-load directly. Returns
    ``model``.
    """
    snap_dir = run_dir(run_id, store=store) / "snapshots" / f"step_{step}"
    legacy = snap_dir / "model.safetensors"
    if legacy.is_file():
        load_model(model, str(legacy))
        return model
    base_file = run_dir(run_id, store=store) / "snapshots" / "base" / "model.safetensors"
    state = load_file(str(base_file)) | load_file(str(snap_dir / "adapter.safetensors"))
    expected = model.state_dict()
    loaded_by_ptr = {expected[k].data_ptr(): k for k in expected if k in state}
    for name, tensor in expected.items():
        if name in state:
            continue
        twin = loaded_by_ptr.get(tensor.data_ptr())
        if twin is None:
            raise ValueError(
                f"{run_id} snapshots/step_{step}: no saved state for {name!r} — it is in "
                "neither the base file nor the adapter and shares storage with no loaded "
                "tensor (not a tied alias); refusing a partial load"
            )
        state[name] = state[twin]
    model.load_state_dict(state, strict=True)
    return model


# Examples per forward in the final θ_T eval: bounds logits memory
# (chunk × seq × vocab) — a 5,000-row val split in one fp32 forward is
# multiple GB of logits alone. Label losses are padding-invariant under
# right-padding + causal attention, so the chunked sum equals the
# single-forward value.
_TEST_EVAL_CHUNK = 512


def _write_test_loss(
    model: torch.nn.Module,
    test_examples: Sequence,
    task_format: TaskFormat,
    mask_hash: str,
    run_id: str,
    *,
    store: Path | None,
) -> None:
    """Evaluate the held-out loss at the final model θ_T and write ``eval/test_loss.json``.

    Uses the shared ``label_mask``/``prequential_step`` path with the same
    ``masking_config_hash`` as the training loop (specs/00 §5, V0.5 producer),
    in chunks of ``_TEST_EVAL_CHUNK`` examples.
    """
    loss_sum_nats = 0.0
    label_token_count = 0
    for start in range(0, len(test_examples), _TEST_EVAL_CHUNK):
        chunk = test_examples[start : start + _TEST_EVAL_CHUNK]
        mask = _checked_label_mask(chunk, task_format)
        step_loss = prequential_step(model, chunk, mask)
        loss_sum_nats += step_loss.loss_sum_nats
        label_token_count += step_loss.label_token_count
    record = TestLoss(
        n_test_examples=len(test_examples),
        label_token_count=label_token_count,
        loss_sum_nats=loss_sum_nats,
        loss_per_label_token_nats=loss_sum_nats / label_token_count,
        masking_config_hash=mask_hash,
    )
    path = test_loss_path(run_id, store=store)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2) + "\n")
