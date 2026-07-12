"""Activation storage: save/load + matched-input enforcement (specs/00 §6).

Safetensors, one file per (model, dataset, hook), with a JSON sidecar
recording provenance. ``load_matched_pair`` is the single gate through which
cross-model activation comparisons must pass (V0.4): it refuses to return
activations for two models unless their recorded dataset_key, position
policy, and tokenizer hash all agree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file, save_file

from geode.zoo.store import resolve_store


@dataclass(frozen=True)
class ActivationMeta:
    """Sidecar provenance for one stored activation tensor (specs/00 §6)."""

    model_key: str
    hf_id: str
    revision: str
    dataset_key: str
    position_policy: Literal["all", "answer_only", "last"]
    n_samples: int
    tokenizer_hash: str


def tokenizer_hash(tokenizer) -> str:
    """sha256 hex digest of the tokenizer's canonical JSON serialization (OQ-7).

    Deterministic across separately-constructed identical tokenizers, and
    sensitive to vocabulary changes.
    """
    canonical = tokenizer.backend_tokenizer.to_str()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _paths(
    model_key: str, dataset_key: str, hook_name: str, *, store: Path | None
) -> tuple[Path, Path]:
    """The §1 layout: activations/{model_key}/{dataset_key}/{hook_name}.{safetensors,meta.json}."""
    base = resolve_store(store) / "activations" / model_key / dataset_key
    return base / f"{hook_name}.safetensors", base / f"{hook_name}.meta.json"


def save_activations(
    acts: torch.Tensor,
    meta: ActivationMeta,
    hook_name: str,
    *,
    store: Path | None = None,
    dtype: torch.dtype = torch.float16,
) -> Path:
    """Write ``acts`` (cast to ``dtype``) and its sidecar; return the tensor path."""
    tensor_path, meta_path = _paths(meta.model_key, meta.dataset_key, hook_name, store=store)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({hook_name: acts.to(dtype).contiguous()}, tensor_path)
    meta_path.write_text(json.dumps(asdict(meta)))
    return tensor_path


def load_activations(
    model_key: str,
    dataset_key: str,
    hook_name: str,
    *,
    store: Path | None = None,
) -> tuple[torch.Tensor, ActivationMeta]:
    """Load the tensor and sidecar written by ``save_activations``."""
    tensor_path, meta_path = _paths(model_key, dataset_key, hook_name, store=store)
    tensor = load_file(tensor_path)[hook_name]
    meta = ActivationMeta(**json.loads(meta_path.read_text()))
    return tensor, meta


def _require_matched(
    field: str, value_a: object, value_b: object, label_a: str, label_b: str
) -> None:
    if value_a != value_b:
        raise ValueError(
            f"matched-input mismatch on '{field}': {label_a}={value_a!r}, {label_b}={value_b!r}"
        )


def load_matched_pair(
    model_key_a: str,
    model_key_b: str,
    dataset_key: str,
    hook_name: str,
    *,
    store: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, ActivationMeta]:
    """Load matched activations for two models (V0.4 gate).

    Refuses (``ValueError``) unless both sidecars' recorded ``dataset_key``
    equal the requested one and each other, and their ``position_policy`` and
    ``tokenizer_hash`` agree — the mechanical guard against comparing
    activations that were not extracted under matched conditions.
    """
    acts_a, meta_a = load_activations(model_key_a, dataset_key, hook_name, store=store)
    acts_b, meta_b = load_activations(model_key_b, dataset_key, hook_name, store=store)

    for label, meta in ((model_key_a, meta_a), (model_key_b, meta_b)):
        if meta.dataset_key != dataset_key:
            raise ValueError(
                f"matched-input mismatch on 'dataset_key': requested={dataset_key!r}, "
                f"{label}={meta.dataset_key!r}"
            )

    _require_matched(
        "dataset_key", meta_a.dataset_key, meta_b.dataset_key, model_key_a, model_key_b
    )
    _require_matched(
        "position_policy", meta_a.position_policy, meta_b.position_policy, model_key_a, model_key_b
    )
    _require_matched(
        "tokenizer_hash", meta_a.tokenizer_hash, meta_b.tokenizer_hash, model_key_a, model_key_b
    )

    return acts_a, acts_b, meta_a
