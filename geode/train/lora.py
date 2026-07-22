"""LoRA adapter for the runs-5/6 target trainer (specs/02 §6, "LoRA (runs 5-6 only)").

``apply_lora`` rewrites a causal LM in place: every ``nn.Linear`` named
``q_proj``/``k_proj``/``v_proj``/``o_proj``/``gate_proj``/``up_proj``/``down_proj``
(Q,K,V,O,G,U,D — all layers) becomes a ``LoRALinear`` computing
``base(x) + scaling * B(A(x))``, every other parameter is frozen, and only the
A/B factors train. Pinned hyperparameters (specs/02 §6): r=128, α=32,
**scaling = α/(2r)** — note the 2r, deliberately not PEFT's α/r — dropout 0,
A init Kaiming 1/√d_in (uniform ±1/√d_in, the bound ``kaiming_uniform_(a=√5)``
resolves to for a fan_in-sized weight), B zero, so θ_0 computes the base
function exactly.

Snapshots stay self-contained (specs/00 §1): the wrapped model's
``state_dict()`` holds base + adapter tensors together (``<name>.base.weight``,
``<name>.A.weight``, ``<name>.B.weight``) and ``apply_lora`` rebuilds the
identical module tree on any fresh base model, so ``reapply_lora`` reloads a
saved state dict bit-exactly — no separate base checkpoint, no merge (merging
would perturb weights by float rounding and hide the factors the adapter-diff
analysis reads straight from the snapshot tensors).

Randomness: the A init is the only randomness and is drawn from a dedicated
CPU ``torch.Generator`` seeded by the explicit ``seed`` argument, in module
registration order, then copied to the base weight's device/dtype — so the
init is bit-identical for a given seed regardless of where the model lives,
and nothing leaks from ambient RNG.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# Spec-02 §6 pin: Q, K, V, O, G, U, D on every layer.
LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable rank-``rank`` update.

    ``forward(x) = base(x) + scaling * B(A(x))`` with
    ``scaling = alpha / (2 * rank)`` (specs/02 §6 — the 2r is the pin, V5.47).
    ``A`` (d_in → r) is initialised uniform ±1/√d_in through the caller's
    ``generator``; ``B`` (r → d_out) is zero, so a freshly wrapped layer
    computes exactly ``base(x)``. The base linear is frozen here, not just in
    ``apply_lora``, so the wrapper is safe standalone.
    """

    def __init__(
        self, base: nn.Linear, *, rank: int, alpha: float, generator: torch.Generator
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRALinear: rank must be positive, got {rank}")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / (2 * rank)
        self.base = base
        self.base.requires_grad_(False)
        d_in, d_out = base.in_features, base.out_features
        device, dtype = base.weight.device, base.weight.dtype
        self.A = nn.Linear(d_in, rank, bias=False, device=device, dtype=dtype)
        self.B = nn.Linear(rank, d_out, bias=False, device=device, dtype=dtype)
        # Draw the A init on CPU fp32 through the shared generator, then copy
        # to the base weight's device/dtype: bit-identical per seed on any
        # device, and the generator never has to live on the model's device.
        bound = 1.0 / math.sqrt(d_in)
        a_init = torch.empty(rank, d_in, dtype=torch.float32)
        a_init.uniform_(-bound, bound, generator=generator)
        with torch.no_grad():
            self.A.weight.copy_(a_init)
            self.B.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.B(self.A(x))


def apply_lora(
    model: nn.Module,
    *,
    rank: int = 128,
    alpha: float = 32.0,
    seed: int,
    target_modules: tuple[str, ...] = LORA_TARGET_MODULES,
    dropout: float = 0.0,
) -> nn.Module:
    """Freeze ``model`` and wrap every target ``nn.Linear`` in ``LoRALinear``, in place.

    After this call every base parameter has ``requires_grad=False`` and the
    only trainable tensors are the A/B factors — rank·(d_in+d_out) per wrapped
    linear (V5.48-V5.49). ``seed`` drives the A init through a dedicated CPU
    generator, drawn in module registration order (V5.50). A target name that
    matches no ``nn.Linear`` raises up front: a typo'd target would silently
    train a smaller adapter, the expensive quiet failure. ``dropout`` exists
    only to refuse non-zero values — the spec pins 0.

    Returns ``model`` (the same object, mutated).
    """
    if dropout != 0.0:
        raise ValueError(
            f"apply_lora: dropout is pinned to 0 for runs 5-6 (specs/02 §6), got {dropout}"
        )
    for param in model.parameters():
        param.requires_grad_(False)
    to_wrap: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent in model.modules():
        for attr, child in parent.named_children():
            if attr in target_modules and isinstance(child, nn.Linear):
                to_wrap.append((parent, attr, child))
    matched = {attr for _, attr, _ in to_wrap}
    missing = [name for name in target_modules if name not in matched]
    if missing:
        raise ValueError(
            f"apply_lora: target modules {missing} matched no nn.Linear on this model "
            "(already wrapped, or a naming mismatch with the arch's projection names)"
        )
    generator = torch.Generator().manual_seed(seed)
    for parent, attr, child in to_wrap:
        setattr(parent, attr, LoRALinear(child, rank=rank, alpha=alpha, generator=generator))
    return model


def reapply_lora(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    rank: int = 128,
    alpha: float = 32.0,
    target_modules: tuple[str, ...] = LORA_TARGET_MODULES,
) -> nn.Module:
    """Rebuild the LoRA module tree on a fresh base model and load a saved snapshot.

    The reload path for the self-contained snapshots of specs/00 §1: a
    ``state_dict()`` saved from an ``apply_lora``-wrapped model only fits the
    same module tree, so this wraps ``model`` with the same
    ``rank``/``alpha``/``target_modules`` and loads strictly. Every tensor —
    base and adapter — comes from the snapshot, so the throwaway init seed is
    irrelevant and the restored model is bit-exact (V5.51).
    """
    wrapped = apply_lora(model, rank=rank, alpha=alpha, seed=0, target_modules=target_modules)
    wrapped.load_state_dict(state_dict, strict=True)
    return wrapped
