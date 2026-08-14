"""Unit test for ``train.lora_adapter_state_dict`` (owner directive
2026-07-31: every LoRA run — target runs via ``train_target.py``, install
runs via ``train_sft.py`` — must also write a compact adapter-only
``adapter.safetensors`` sidecar alongside the self-contained ``model/`` save,
so a run's weights stay recoverable without moving the full ~2.5GB state
dict). This is the pure key-filter behind that sidecar — CPU-only, no
checkpoint or tokenizer needed. Lives in ``train.py`` (promoted 2026-07-31
from ``train_target.py``, its original home) since both launchers import it.
"""

from __future__ import annotations

import torch

from tests._scriptloader import load

train = load("train")


def test_lora_adapter_state_dict_keeps_only_a_and_b_weights() -> None:
    state_dict = {
        "model.layers.0.self_attn.q_proj.base.weight": torch.randn(4, 4),
        "model.layers.0.self_attn.q_proj.A.weight": torch.randn(2, 4),
        "model.layers.0.self_attn.q_proj.B.weight": torch.randn(4, 2),
        "model.layers.0.mlp.gate_proj.A.weight": torch.randn(2, 4),
        "model.layers.0.mlp.gate_proj.B.weight": torch.randn(4, 2),
        "model.embed_tokens.weight": torch.randn(8, 4),  # plain, unwrapped
    }

    adapter = train.lora_adapter_state_dict(state_dict)

    assert set(adapter) == {
        "model.layers.0.self_attn.q_proj.A.weight",
        "model.layers.0.self_attn.q_proj.B.weight",
        "model.layers.0.mlp.gate_proj.A.weight",
        "model.layers.0.mlp.gate_proj.B.weight",
    }
    for name in adapter:
        assert torch.equal(adapter[name], state_dict[name])
