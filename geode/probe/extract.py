"""Offline probe extraction: residual activations + activation gradients
(specs/02 §7 "Extraction pass", V5.9-V5.12).

For one snapshot model θ and the frozen probe set, ``extract_probe`` runs a
single forward + backward (label-token loss, **sum** reduction — spec §7) and
captures, at every residual point — the embedding output plus each post-block
residual, ``n_layers + 1`` hook points, never hardcoded — the per-example
activations AND the per-example gradients of the loss w.r.t. those
activations. Sum reduction makes example i's loss depend only on example i's
activations (causal attention, right-padding, no cross-example ops), so the
batched backward yields exactly the one-example-at-a-time gradients (V5.10) —
padding positions receive exactly zero gradient, and an example with zero
label-token loss has an all-zero gradient row.

Loss masking routes through the shared ``geode.edl.masking.label_mask`` (M1 —
the single mask-construction path), with the same p=0 guard as the training
loop. The per-example probe loss is returned and stored per snapshot (nats;
late gradients are numerically degenerate — analyses condition on nonzero
loss, spec §7).

Storage (spec §7): one safetensors file per (snapshot, quantity ∈ {acts,
grads}), each holding the ``n_layers + 1`` named residual tensors in bf16;
the padded ``input_ids`` + attention mask + label mask + per-example loss are
stored alongside (``probe_data.safetensors``); sidecar ``meta.json`` records
run_id, arm, step, probe_set_hash, tokenizer_hash, base_model_key, dtype,
plus the template/format identity (task name + format_version) the
matched-load guard checks. Dumps live at ``runs/{run_id}/probe/step_{k}/``.

Matched-load guard (V5.12, the spec-00 V0.4 pattern re-asserted at this
surface): ``load_matched_probe_pair`` refuses cross-arm comparison unless
probe_set_hash, tokenizer_hash, and template/format metadata all match.
run_id / arm / step / base_model_key legitimately differ across arms and
snapshots and are never checked.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import torch
from safetensors.torch import load_file, save_file

from geode.edl.masking import TaskFormat, label_mask
from geode.zoo.store import run_dir

ACTS_FILE = "acts.safetensors"
GRADS_FILE = "grads.safetensors"
PROBE_DATA_FILE = "probe_data.safetensors"
META_FILE = "meta.json"

# Storage dtype for acts/grads is pinned bf16 (spec 02 §7); the map exists so
# meta.dtype and the file contents agree by construction, never by convention.
_STORAGE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Meta fields that must agree for a cross-arm comparison (spec 02 §7 guard):
# the probe-set identity, the tokenizer identity, and the template/format
# identity. Everything else (run_id, arm, step, base_model_key) legitimately
# differs between the two dumps being compared.
_MATCH_FIELDS = ("probe_set_hash", "tokenizer_hash", "task_name", "format_version")


class _LabeledExample(Protocol):
    input_ids: list[int]
    label_span: tuple[int, int]


@dataclass(frozen=True)
class ProbeMeta:
    """Sidecar provenance for one probe dump (spec 02 §7).

    ``task_name`` + ``format_version`` are the template/format metadata the
    matched-load guard checks; ``dtype`` names the storage dtype of the
    acts/grads tensors (pinned ``"bfloat16"``, spec §7).
    """

    run_id: str
    arm: str
    step: int
    probe_set_hash: str
    tokenizer_hash: str
    base_model_key: str
    task_name: str
    format_version: str
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class ProbeCapture:
    """One snapshot's probe extraction, on CPU, in the model's compute dtype.

    ``acts`` and ``grads`` map hook name → ``[n_examples, seq, d_model]``;
    ``attention_mask`` is True at real (non-pad) positions, ``label_mask`` is
    the shared M1 mask, ``probe_loss_nats`` the per-example label-token loss
    sum (fp32 nats).
    """

    acts: dict[str, torch.Tensor]
    grads: dict[str, torch.Tensor]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label_mask: torch.Tensor
    probe_loss_nats: torch.Tensor


def residual_hook_names(n_layers: int) -> list[str]:
    """The ``n_layers + 1`` residual hook names, TransformerLens convention
    (spec 00 §6): ``hook_embed`` + ``blocks.{i}.hook_resid_post``."""
    return ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]


def _residual_modules(model: torch.nn.Module) -> tuple[torch.nn.Module, Sequence[torch.nn.Module]]:
    """The embedding module + the decoder-block list of a Llama-style causal LM."""
    embed = model.get_input_embeddings()
    decoder = getattr(model, "model", None)
    blocks = getattr(decoder, "layers", None)
    if embed is None or blocks is None:
        raise ValueError(
            "extract_probe: expected a Llama-style causal LM with "
            "get_input_embeddings() and model.layers; got "
            f"{type(model).__name__}"
        )
    return embed, blocks


def _padded_batch(
    examples: Sequence[_LabeledExample], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad ``input_ids`` to the set max (pad id 0 — the geode.edl
    convention) and build the real-token attention mask."""
    max_len = max(len(ex.input_ids) for ex in examples)
    input_ids = torch.zeros((len(examples), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.bool)
    for i, ex in enumerate(examples):
        ids = torch.tensor(ex.input_ids, dtype=torch.long)
        input_ids[i, : ids.shape[0]] = ids
        attention_mask[i, : ids.shape[0]] = True
    return input_ids.to(device), attention_mask


def extract_probe(
    model: torch.nn.Module,
    examples: Sequence[_LabeledExample],
    task_format: TaskFormat,
    *,
    device: str,
) -> ProbeCapture:
    """Forward + backward the probe set through ``model``; capture residuals.

    Loss is the **sum** over label tokens (spec §7) via the shared M1
    ``label_mask``; gradients are taken w.r.t. the captured residual tensors
    only (``torch.autograd.grad``), so no parameter ``.grad`` is ever touched.
    The embedding output is detached and re-marked as requiring grad before it
    enters the blocks: that makes the capture work on fully frozen models and
    stops the backward at the embedding, without altering any downstream value
    or gradient. Returns CPU tensors in the model's compute dtype (storage
    casts to bf16 happen in ``save_probe_dump``). ``model`` is moved to
    ``device`` and put in eval mode.
    """
    if not examples:
        raise ValueError("extract_probe: empty probe set")
    model.to(device)
    model.eval()

    mask = label_mask(examples, task_format)  # M1: the single shared mask path
    if bool(mask[:, 0].any()):
        raise ValueError(
            "extract_probe: a label span includes position 0, but a causal LM "
            "predicts token p from position p-1 — label spans must start at "
            "index >= 1 (same guard as the training loop)."
        )
    input_ids, attention_mask = _padded_batch(examples, device)

    embed, blocks = _residual_modules(model)
    names = residual_hook_names(len(blocks))
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def embed_hook(_module, _inputs, output):
        t = output.detach().requires_grad_(True)
        captured["hook_embed"] = t
        return t  # replaces the module output: the graph roots here

    def make_block_hook(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output[0] if isinstance(output, tuple) else output

        return hook

    handles.append(embed.register_forward_hook(embed_hook))
    for i, block in enumerate(blocks):
        handles.append(block.register_forward_hook(make_block_hook(names[i + 1])))

    try:
        with torch.enable_grad():
            logits = model(input_ids=input_ids).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            token_ll = log_probs[:, :-1, :].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            label = mask[:, 1:].to(device=token_ll.device, dtype=token_ll.dtype)
            per_example_loss = -(token_ll * label).sum(dim=1)  # [n], sum reduction
            missing = [name for name in names if name not in captured]
            if missing:
                raise RuntimeError(f"extract_probe: hooks never fired for {missing}")
            grads = torch.autograd.grad(per_example_loss.sum(), [captured[name] for name in names])
    finally:
        for handle in handles:
            handle.remove()

    return ProbeCapture(
        acts={name: captured[name].detach().cpu() for name in names},
        grads={name: g.detach().cpu() for name, g in zip(names, grads)},
        input_ids=input_ids.cpu(),
        attention_mask=attention_mask,
        label_mask=mask,
        probe_loss_nats=per_example_loss.detach().float().cpu(),
    )


def probe_dump_dir(run_id: str, step: int, *, store: Path | None = None) -> Path:
    """Directory of one probe dump: ``runs/{run_id}/probe/step_{step}``."""
    return run_dir(run_id, store=store) / "probe" / f"step_{step}"


def load_probe_dumps(root: Path, *, marker: str | tuple[str, ...] = META_FILE) -> list[int]:
    """Sorted step indices of the ``step_*`` dump directories under ``root`` (V5.72).

    A ``step_k`` subdirectory counts only if it carries ``marker`` — the probe
    dump sidecar (``meta.json``, the default) for probe dumps, or a tuple of
    filenames meaning "any present" (snapshot dirs carry ``adapter.safetensors``
    OR ``model.safetensors``). The glob order is arbitrary; the returned indices
    are ascending. Returns ``[]`` when none match, so the caller raises its own
    contextual error. Replaces the hand-rolled
    ``sorted(int(p.name.removeprefix("step_")) for p in root.glob("step_*") …)``
    idiom the analysis drivers and ``extract.py`` duplicated.
    """
    markers = (marker,) if isinstance(marker, str) else tuple(marker)
    return sorted(
        int(p.name.removeprefix("step_"))
        for p in root.glob("step_*")
        if any((p / m).is_file() for m in markers)
    )


def assert_probe_alignment(
    dump_probe_set_hash: str, probe_set_hash: str, *, run_id: str, step: int
) -> None:
    """Row-alignment guard (spec 00 §6, V5.72): a dump aligns with the frozen probe
    parquet only when the dump's recorded ``probe_set_hash`` equals the parquet's
    ``order_hash`` — else parquet row i and dump row i are not the same probe
    example. Raises ``ValueError`` naming the run and step; the drivers that
    hand-rolled this ``_guard_probe_hash`` check delegate here.
    """
    if dump_probe_set_hash != probe_set_hash:
        raise ValueError(
            f"{run_id}@step_{step}: dump probe_set_hash {dump_probe_set_hash} != probe "
            f"parquet order_hash {probe_set_hash} — parquet row i and dump row i must be "
            "the same frozen probe example (spec 00 §6); refusing to align mismatched sets"
        )


def save_probe_dump(capture: ProbeCapture, meta: ProbeMeta, *, store: Path | None = None) -> Path:
    """Write one dump (spec §7 layout) and return its directory.

    Acts/grads are cast to ``meta.dtype`` (pinned bf16); input_ids and the two
    masks are stored exactly; the per-example loss stays fp32 (losses are
    reported quantities, never down-cast).
    """
    if meta.dtype not in _STORAGE_DTYPES:
        raise ValueError(
            f"save_probe_dump: unsupported storage dtype {meta.dtype!r} "
            f"(expected one of {sorted(_STORAGE_DTYPES)})"
        )
    storage = _STORAGE_DTYPES[meta.dtype]
    out = probe_dump_dir(meta.run_id, meta.step, store=store)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.to(storage).contiguous() for k, v in capture.acts.items()}, out / ACTS_FILE)
    save_file({k: v.to(storage).contiguous() for k, v in capture.grads.items()}, out / GRADS_FILE)
    save_file(
        {
            "input_ids": capture.input_ids.contiguous(),
            "attention_mask": capture.attention_mask.contiguous(),
            "label_mask": capture.label_mask.contiguous(),
            "probe_loss_nats": capture.probe_loss_nats.to(torch.float32).contiguous(),
        },
        out / PROBE_DATA_FILE,
    )
    (out / META_FILE).write_text(json.dumps(asdict(meta), indent=2) + "\n")
    return out


def load_probe_dump(
    run_id: str, step: int, *, store: Path | None = None
) -> tuple[ProbeCapture, ProbeMeta]:
    """Load the dump written by ``save_probe_dump`` (values as stored: bf16)."""
    dump = probe_dump_dir(run_id, step, store=store)
    meta = ProbeMeta(**json.loads((dump / META_FILE).read_text()))
    data = load_file(dump / PROBE_DATA_FILE)
    capture = ProbeCapture(
        acts=load_file(dump / ACTS_FILE),
        grads=load_file(dump / GRADS_FILE),
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
        label_mask=data["label_mask"],
        probe_loss_nats=data["probe_loss_nats"],
    )
    return capture, meta


def load_matched_probe_pair(
    run_id_a: str,
    step_a: int,
    run_id_b: str,
    step_b: int,
    *,
    store: Path | None = None,
) -> tuple[ProbeCapture, ProbeCapture, ProbeMeta, ProbeMeta]:
    """Load two dumps for cross-arm comparison, or refuse (V5.12).

    Raises ``ValueError`` naming the first mismatching identity field unless
    ``probe_set_hash``, ``tokenizer_hash``, and the template/format metadata
    (``task_name``, ``format_version``) all match — the spec-00 V0.4 pattern
    re-asserted at this surface.
    """
    cap_a, meta_a = load_probe_dump(run_id_a, step_a, store=store)
    cap_b, meta_b = load_probe_dump(run_id_b, step_b, store=store)
    label_a, label_b = f"{run_id_a}@step_{step_a}", f"{run_id_b}@step_{step_b}"
    for field in _MATCH_FIELDS:
        value_a, value_b = getattr(meta_a, field), getattr(meta_b, field)
        if value_a != value_b:
            raise ValueError(
                f"matched-probe mismatch on '{field}': "
                f"{label_a}={value_a!r}, {label_b}={value_b!r} (spec 02 §7: "
                "cross-arm comparison requires matched probe set, tokenizer, "
                "and template/format)"
            )
    return cap_a, cap_b, meta_a, meta_b
