"""V0.9 tests: method-faithful checkpoint loading (``geode.zoo.load_model``).

Derived from specs/00-interfaces.md §8 and validation property V0.9: the
manifest's ``training.method`` decides how the checkpoint is loaded, a
checkpoint whose state-dict shape contradicts the method is a loud refusal,
and the LoRA path is bit-exact with the model that was saved (the 2026-07-22
G5 incident was plain ``from_pretrained`` silently random-initializing every
wrapped projection of a LoRA checkpoint).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from geode.train import apply_lora
from geode.zoo import load_model
from tests.zoo.test_manifest import make_manifest

LORA_KW = {"rank": 4, "alpha": 16.0, "target_modules": ("q_proj", "v_proj")}


def write_run(store: Path, run_id: str, model: nn.Module, method: str) -> None:
    d = store / "runs" / run_id
    d.mkdir(parents=True)
    model.save_pretrained(d / "model")
    fields = make_manifest(run_id)
    fields["training"]["method"] = method
    fields["training"]["lora"].update(
        rank=LORA_KW["rank"],
        alpha=LORA_KW["alpha"],
        target_modules=list(LORA_KW["target_modules"]),
        dropout=0.0,
    )
    (d / "manifest.json").write_text(json.dumps(fields))


def perturb_adapters(model: nn.Module, seed: int) -> None:
    """Give A/B non-init values, as training would — a zero B would let a
    loader that drops the adapter pass the round-trip test."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("A.weight", "B.weight")):
                p.copy_(torch.rand(p.shape, generator=g) - 0.5)


def logits(model: nn.Module) -> torch.Tensor:
    ids = torch.arange(12).reshape(2, 6) + 4  # word tokens, fixed input
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_v0_9_lora_checkpoint_round_trip_bit_exact(tiny_llama, tmp_path: Path) -> None:
    store = tmp_path / "store"
    saved = apply_lora(tiny_llama(seed=0), seed=1, **LORA_KW)
    perturb_adapters(saved, seed=9)
    write_run(store, "run-lora", saved, "lora")

    loaded = load_model("run-lora", store=store)
    assert torch.equal(logits(loaded), logits(saved))


def test_v0_9_tied_embeddings_lora_round_trip_bit_exact(tiny_llama, tmp_path: Path) -> None:
    # Production base models tie embeddings, so save_pretrained drops the
    # lm_head.weight duplicate from safetensors (2026-07-22 G5 re-run failure:
    # strict reapply_lora raised "Missing key(s) ... lm_head.weight").
    store = tmp_path / "store"
    saved = apply_lora(tiny_llama(seed=0, tie_word_embeddings=True), seed=1, **LORA_KW)
    perturb_adapters(saved, seed=9)
    write_run(store, "run-tied", saved, "lora")

    loaded = load_model("run-tied", store=store)
    assert torch.equal(logits(loaded), logits(saved))


def test_v0_9_full_ft_checkpoint_round_trip_bit_exact(tiny_llama, tmp_path: Path) -> None:
    store = tmp_path / "store"
    saved = tiny_llama(seed=0)
    write_run(store, "run-ft", saved, "full_ft")

    loaded = load_model("run-ft", store=store)
    assert torch.equal(logits(loaded), logits(saved))


def test_v0_9_wrapped_checkpoint_under_full_ft_manifest_refused(tiny_llama, tmp_path: Path) -> None:
    # The G5 incident shape: a LoRA checkpoint reaching a plain-load path.
    store = tmp_path / "store"
    wrapped = apply_lora(tiny_llama(seed=0), seed=1, **LORA_KW)
    write_run(store, "run-bad", wrapped, "full_ft")

    with pytest.raises(ValueError, match="LoRA-wrapped"):
        load_model("run-bad", store=store)


def test_v0_9_plain_checkpoint_under_lora_manifest_refused(tiny_llama, tmp_path: Path) -> None:
    store = tmp_path / "store"
    write_run(store, "run-bad", tiny_llama(seed=0), "lora")

    with pytest.raises(ValueError, match="plain state dict"):
        load_model("run-bad", store=store)
