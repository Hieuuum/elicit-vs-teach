"""LoRA adapter tests — specs/02 §6 "LoRA (runs 5-6 only)" pin, V5.47-V5.52.

CPU-only, tiny in-process models, fp32. The pinned r=128 and the 12.1M-param
figure belong to the real 512-hidden/8-layer arch and are never hardcoded
here: the tests exercise the *formulas* (identity at B=0, scaling = α/(2r),
count = Σ rank·(d_in+d_out)) on a small rank over the tiny fixture.

No smoke through ``train_prequential``: its current surface builds its own
PEFT adapter internally from ``manifest.training.lora``, so it cannot accept
a pre-wrapped model yet — that wiring (and the α/2r-vs-PEFT-α/r scaling
mismatch it implies) is runs-5/6 launch-prep work outside this module.
"""

from __future__ import annotations

import pytest
import torch

from geode.train.lora import LORA_TARGET_MODULES, LoRALinear, apply_lora, merge_lora, reapply_lora
from tests.conftest import DEFAULT_VOCAB_SIZE

# Tiny stand-ins for the pinned r=128/α=32: small enough to be fast, and
# α/(2r) = 4.0 is float-exact, so scaling comparisons can demand bit equality.
_RANK = 4
_ALPHA = 32.0


def _random_ids(seed: int, batch: int = 2, seq: int = 8) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, DEFAULT_VOCAB_SIZE, (batch, seq), generator=gen)


def _logits(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(ids).logits


def _is_adapter(param_name: str) -> bool:
    return param_name.endswith(".A.weight") or param_name.endswith(".B.weight")


def _train_steps(model: torch.nn.Module, ids: torch.Tensor, n_steps: int = 2) -> None:
    """A few AdamW steps on the trainable params only (wd=0: grad-driven moves only)."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.0)
    vocab = model.config.vocab_size
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab), ids[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()


def _a_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name.endswith(".A.weight")
    }


# --------------------------------------------------------------------------
# LoRALinear.__init__ — rank must be positive
# --------------------------------------------------------------------------
@pytest.mark.parametrize("rank", [0, -3])
def test_lora_linear_rejects_nonpositive_rank(rank: int) -> None:
    base = torch.nn.Linear(4, 4)
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="rank must be positive"):
        LoRALinear(base, rank=rank, alpha=_ALPHA, generator=generator)


# --------------------------------------------------------------------------
# V5.47 — identity at init; forward implements base(x) + α/(2r)·B(A(x))
# --------------------------------------------------------------------------
def test_v5_47_identity_at_init_and_2r_scaling(tiny_llama):
    # B=0 at init ⇒ the wrapped model computes the base function exactly:
    # θ_0 is the pretrained state (specs/01 §1), bit for bit.
    model = tiny_llama(seed=0)
    ids = _random_ids(seed=1)
    base_logits = _logits(model, ids)
    apply_lora(model, rank=_RANK, alpha=_ALPHA, seed=2)
    assert torch.equal(_logits(model, ids), base_logits)

    # The 2r pin has teeth (α/(2r), not PEFT's α/r): perturb one B and check
    # the forward against the explicit formula with the literal constant.
    layer = model.model.layers[0].self_attn.q_proj
    assert isinstance(layer, LoRALinear)
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        layer.B.weight.copy_(torch.randn(layer.B.weight.shape, generator=gen))
    x = torch.randn(2, 5, model.config.hidden_size, generator=gen)
    expected = layer.base(x) + (_ALPHA / (2 * _RANK)) * layer.B(layer.A(x))
    assert torch.equal(layer(x), expected)


# --------------------------------------------------------------------------
# V5.48 — only the adapters train; every base tensor stays bit-identical
# --------------------------------------------------------------------------
def test_v5_48_only_adapters_train(tiny_llama):
    model = tiny_llama(seed=0)
    apply_lora(model, rank=_RANK, alpha=_ALPHA, seed=1)
    for name, param in model.named_parameters():
        assert param.requires_grad == _is_adapter(name), name

    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    # Two steps: at θ_0, B=0 makes A's gradient exactly zero, so step 1 moves
    # only B; step 2 (B≠0) moves A too.
    _train_steps(model, _random_ids(seed=2), n_steps=2)

    changed = {
        name for name, param in model.named_parameters() if not torch.equal(param, before[name])
    }
    for name in before:
        if not _is_adapter(name):
            assert name not in changed, f"base tensor {name} moved"
    assert any(name.endswith(".A.weight") for name in changed)
    assert any(name.endswith(".B.weight") for name in changed)


# --------------------------------------------------------------------------
# V5.49 — trainable count = Σ rank·(d_in+d_out); full target coverage; guards
# --------------------------------------------------------------------------
def test_v5_49_trainable_count_and_target_coverage(tiny_llama):
    n_layers = 3
    model = tiny_llama(seed=0, n_layers=n_layers)
    apply_lora(model, rank=_RANK, alpha=_ALPHA, seed=1)

    # Every listed target module on every layer got wrapped, and nothing else.
    wrapped = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(wrapped) == n_layers * len(LORA_TARGET_MODULES)
    for layer in model.model.layers:
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert isinstance(getattr(layer.self_attn, proj), LoRALinear)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert isinstance(getattr(layer.mlp, proj), LoRALinear)
    assert not isinstance(model.lm_head, LoRALinear)  # a Linear, but not a target

    # The count formula behind the spec's 12.1M figure at the real arch.
    expected = sum(_RANK * (m.base.in_features + m.base.out_features) for m in wrapped)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == expected

    # Loud-failure guards: the pinned dropout=0 and unmatched target names.
    with pytest.raises(ValueError, match="dropout"):
        apply_lora(tiny_llama(seed=0), rank=_RANK, alpha=_ALPHA, seed=1, dropout=0.1)
    with pytest.raises(ValueError, match="matched no"):
        apply_lora(
            tiny_llama(seed=0),
            rank=_RANK,
            alpha=_ALPHA,
            seed=1,
            target_modules=("q_proj", "nope_proj"),
        )


# --------------------------------------------------------------------------
# V5.50 — seeded A init: same seed ⇒ identical, different seed ⇒ different
# --------------------------------------------------------------------------
def test_v5_50_seeded_init_reproducible(tiny_llama):
    a1 = _a_tensors(apply_lora(tiny_llama(seed=0), rank=_RANK, alpha=_ALPHA, seed=7))
    a2 = _a_tensors(apply_lora(tiny_llama(seed=0), rank=_RANK, alpha=_ALPHA, seed=7))
    a3 = _a_tensors(apply_lora(tiny_llama(seed=0), rank=_RANK, alpha=_ALPHA, seed=8))
    assert a1.keys() == a2.keys() == a3.keys()
    assert all(torch.equal(a1[name], a2[name]) for name in a1)
    assert all(not torch.equal(a1[name], a3[name]) for name in a1)


# --------------------------------------------------------------------------
# V5.51 — full state_dict round-trip through a fresh base model
# --------------------------------------------------------------------------
def test_v5_51_state_dict_round_trip(tiny_llama):
    # The self-contained snapshot property (specs/00 §1): state_dict holds
    # base + adapter tensors, and fresh-base → reapply_lora → load reproduces
    # the outputs bit-exactly.
    model = tiny_llama(seed=0)
    apply_lora(model, rank=_RANK, alpha=_ALPHA, seed=1)
    ids = _random_ids(seed=2)
    # Train a little first: with B≠0 the round-trip fails if adapter tensors
    # were dropped, and a different-seed fresh base fails if base tensors were.
    _train_steps(model, ids, n_steps=2)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    reference = _logits(model, ids)

    restored = reapply_lora(tiny_llama(seed=9), state, rank=_RANK, alpha=_ALPHA)
    assert torch.equal(_logits(restored, ids), reference)


# --------------------------------------------------------------------------
# V5.52 — merge for cross-stage parent handoff (never snapshots)
# --------------------------------------------------------------------------
def test_v5_52a_merge_logits_match_wrapped(tiny_llama):
    # Nonzero B so the merge exercises the actual delta formula, not just the
    # B=0 identity already covered by V5.47/(b) below.
    model = tiny_llama(seed=0)
    apply_lora(model, rank=_RANK, alpha=_ALPHA, seed=1)
    gen = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.B.weight.copy_(torch.randn(module.B.weight.shape, generator=gen))
    ids = _random_ids(seed=3)
    wrapped_logits = _logits(model, ids)

    merge_lora(model)
    merged_logits = _logits(model, ids)
    torch.testing.assert_close(merged_logits, wrapped_logits)


def test_v5_52b_merge_at_init_bit_exact_to_base(tiny_llama):
    # B=0 at init (V5.47) ⇒ the merge delta is exactly zero: every merged
    # weight must be bit-identical to the pre-apply_lora base weight.
    model = tiny_llama(seed=0)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    apply_lora(model, rank=_RANK, alpha=_ALPHA, seed=1)

    merge_lora(model)
    after = dict(model.named_parameters())
    assert after.keys() == before.keys()
    for name, param in after.items():
        assert torch.equal(param, before[name]), name


def test_v5_52c_merged_state_dict_keys_match_unwrapped(tiny_llama):
    # Plain-checkpoint round-trip property: a merged model's keys must equal
    # a never-wrapped model's, and no LoRALinear may remain on the tree.
    model = tiny_llama(seed=0)
    apply_lora(model, rank=_RANK, alpha=_ALPHA, seed=1)
    merge_lora(model)

    reference = tiny_llama(seed=5)
    assert model.state_dict().keys() == reference.state_dict().keys()
    assert not any(isinstance(module, LoRALinear) for module in model.modules())


def test_v5_52d_merge_unwrapped_raises(tiny_llama):
    # Merging a model with no adapters is a caller bug, not a silent no-op.
    model = tiny_llama(seed=0)
    with pytest.raises(ValueError, match="no LoRALinear"):
        merge_lora(model)


def test_v5_52e_merge_reenables_grad_everywhere(tiny_llama):
    # apply_lora freezes everything, including non-wrapped params (e.g.
    # embeddings); merge_lora must undo that so the result behaves like a
    # freshly loaded plain model.
    model = tiny_llama(seed=0)
    apply_lora(model, rank=_RANK, alpha=_ALPHA, seed=1)
    merge_lora(model)
    assert all(param.requires_grad for param in model.parameters())
