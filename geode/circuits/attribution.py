"""OLMo 2 node attribution, exact patching, and residual feature extraction.

Attention nodes are individual concatenated query-head outputs before o_proj.
The MLP node is the output of post_feedforward_layernorm, i.e. the complete
write to the residual stream. The latter is deliberately *not* down_proj:
OLMo 2 normalizes the branch after the MLP, unlike Llama.

Scores are signed first-order corrupt-minus-clean changes in the mean answer
log probability (nats), or in a supplied correct/incorrect label logit margin.
Inputs include the teacher-forced answer. The mask marks TARGET token positions,
so position t is evaluated using logits at t-1. Pair construction is external.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor, nn


def node_names(model: nn.Module) -> list[str]:
    """Return the stable, layer-major node universe for an OLMo 2 model."""
    if model.config.model_type != "olmo2":
        raise ValueError("This implementation requires model_type='olmo2'")
    return [
        name
        for i in range(len(model.model.layers))
        for name in [
            *(f"layer.{i}.attn.{h}" for h in range(model.config.num_attention_heads)),
            f"layer.{i}.mlp",
        ]
    ]


@contextmanager
def _evaluation(model: nn.Module) -> Iterator[None]:
    states = [(module, module.training) for module in model.modules()]
    if getattr(model, "is_gradient_checkpointing", False):
        raise ValueError("Disable gradient checkpointing before circuit evaluation")
    model.eval()
    try:
        yield
    finally:
        for module, training in states:
            module.training = training


class NodeTaps:
    """Context-managed activation hooks; handles are removed even on errors."""

    def __init__(self, model: nn.Module):
        node_names(model)
        self.model = model
        self.acts: dict[str, Tensor] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> NodeTaps:
        for i, layer in enumerate(self.model.model.layers):
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._attention(i))
            )
            self.handles.append(
                layer.post_feedforward_layernorm.register_forward_hook(self._mlp(i))
            )
        return self

    def _attention(self, layer: int) -> Any:
        def hook(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            self.acts[f"layer.{layer}.attn"] = inputs[0]

        return hook

    def _mlp(self, layer: int) -> Any:
        def hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            self.acts[f"layer.{layer}.mlp"] = output

        return hook

    def __exit__(self, *_exc: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _prepare(
    model: nn.Module,
    clean_ids: Sequence[int] | Tensor,
    corrupt_ids: Sequence[int] | Tensor,
    answer_mask: Sequence[bool] | Tensor,
    device: str | torch.device | None,
    negative_ids: Sequence[int] | Tensor | None,
    attention_mask: Sequence[int] | Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
    node_names(model)
    device = device if device is not None else next(model.parameters()).device
    clean = torch.as_tensor(clean_ids, dtype=torch.long, device=device)
    corrupt = torch.as_tensor(corrupt_ids, dtype=torch.long, device=device)
    selected = torch.as_tensor(answer_mask, dtype=torch.bool, device=device)
    if clean.ndim != 1 or len(clean) < 2 or corrupt.shape != clean.shape:
        raise ValueError("Clean and corrupt sequences must be 1D and EXACTLY token aligned")
    if selected.shape != clean.shape or not selected.any() or selected[0]:
        raise ValueError("answer_mask must select target positions > 0 in the aligned sequence")
    if not torch.equal(clean[selected], corrupt[selected]):
        raise ValueError("Teacher-forced target answer IDs must be identical across the pair")
    visible = (
        torch.ones_like(clean)
        if attention_mask is None
        else torch.as_tensor(attention_mask, dtype=torch.long, device=device)
    )
    if visible.shape != clean.shape or not ((visible == 0) | (visible == 1)).all():
        raise ValueError("attention_mask must be aligned and binary")
    if not (visible[selected] == 1).all() or not (visible[:-1][selected[1:]] == 1).all():
        raise ValueError("Answer tokens and their predicting positions must not be padding")
    if (clean < 0).any() or (corrupt < 0).any():
        raise ValueError("Token IDs must be nonnegative")
    if (clean >= model.config.vocab_size).any() or (corrupt >= model.config.vocab_size).any():
        raise ValueError("Token ID outside model vocabulary")
    negative = None
    if negative_ids is not None:
        negative = torch.as_tensor(negative_ids, dtype=torch.long, device=device)
        if negative.shape != clean.shape:
            raise ValueError("negative_ids must align with the complete sequence")
        neg = negative[selected]
        if (neg < 0).any() or (neg >= model.config.vocab_size).any():
            raise ValueError("Selected negative token outside model vocabulary")
        if (neg == clean[selected]).any():
            raise ValueError("Correct and incorrect label IDs must differ")
    # This API processes one pair at a time. Strip only shared boundary padding
    # after enforcing original exact alignment; fully masked leading attention
    # rows can otherwise produce NaNs in some eager-attention implementations.
    start, stop = int(visible.nonzero()[0]), int(visible.nonzero()[-1]) + 1
    if not visible[start:stop].all():
        raise ValueError("Only boundary padding is supported; internal padding breaks alignment")
    return (
        clean[start:stop],
        corrupt[start:stop],
        selected[start:stop],
        None if negative is None else negative[start:stop],
        visible[start:stop],
    )


def answer_metric(
    logits: Tensor, target_ids: Tensor, answer_mask: Tensor, negative_ids: Tensor | None = None
) -> Tensor:
    """Mean teacher-forced log probability in nats, or mean label logit margin."""
    selected_logits = logits[0, :-1][answer_mask[1:]]
    return _selected_metric(selected_logits, target_ids, answer_mask, negative_ids)


def _selected_metric(
    selected_logits: Tensor, target_ids: Tensor, answer_mask: Tensor, negative_ids: Tensor | None
) -> Tensor:
    if selected_logits.dtype in (torch.float16, torch.bfloat16):
        selected_logits = selected_logits.float()
    targets = target_ids[answer_mask]
    if negative_ids is None:
        return selected_logits.log_softmax(-1).gather(1, targets[:, None]).mean()
    return (
        selected_logits.gather(1, targets[:, None])
        - selected_logits.gather(1, negative_ids[answer_mask, None])
    ).mean()


def _forward(model: nn.Module, ids: Tensor, visible: Tensor, selected: Tensor) -> Tensor:
    # Explicit positions make left-padding invariant, including on HF versions
    # whose ordinary forward does not infer positions from the attention mask.
    positions = (visible.cumsum(0) - 1).clamp_min(0)
    return model(
        input_ids=ids[None],
        attention_mask=visible[None],
        position_ids=positions[None],
        use_cache=False,
        logits_to_keep=selected.nonzero().flatten() - 1,
    ).logits


def _metric_forward(
    model: nn.Module,
    ids: Tensor,
    targets: Tensor,
    visible: Tensor,
    selected: Tensor,
    negative: Tensor | None,
) -> Tensor:
    logits = _forward(model, ids, visible, selected)
    if logits.shape[1] != selected.sum():
        raise RuntimeError("Installed OLMo 2 implementation must support tensor logits_to_keep")
    return _selected_metric(logits[0], targets, selected, negative)


def score_pair(
    model: nn.Module,
    clean_ids: Sequence[int] | Tensor,
    corrupt_ids: Sequence[int] | Tensor,
    answer_mask: Sequence[bool] | Tensor,
    *,
    device: str | torch.device | None = None,
    negative_ids: Sequence[int] | Tensor | None = None,
    attention_mask: Sequence[int] | Tensor | None = None,
) -> dict[str, Any]:
    """Score every node for one explicitly aligned pair without parameter grads.

    The metric is averaged over answer tokens, while node contributions sum
    over sequence positions/features. Frozen models are supported. Existing
    parameter gradients and train/eval state are preserved; no cache is used.
    """
    clean, corrupt, selected, negative, visible = _prepare(
        model, clean_ids, corrupt_ids, answer_mask, device, negative_ids, attention_mask
    )
    with _evaluation(model), NodeTaps(model) as taps:
        with torch.no_grad():
            corrupt_metric = _metric_forward(model, corrupt, clean, visible, selected, negative)
            corrupt_acts = {key: value.detach().clone() for key, value in taps.acts.items()}
        taps.acts.clear()

        def enable_grad(_module: nn.Module, _inputs: Any, output: Tensor) -> Tensor:
            return output.requires_grad_(True)

        handle = model.get_input_embeddings().register_forward_hook(enable_grad)
        try:
            with torch.enable_grad():
                metric = _metric_forward(model, clean, clean, visible, selected, negative)
                keys = list(taps.acts)
                gradients = torch.autograd.grad(metric, [taps.acts[key] for key in keys])
        finally:
            handle.remove()
        scores: dict[str, float] = {}
        for key, gradient in zip(keys, gradients, strict=True):
            product = (
                corrupt_acts[key].float() - taps.acts[key].detach().float()
            ) * gradient.float()
            if not torch.isfinite(product).all():
                raise FloatingPointError(f"Nonfinite attribution at {key}")
            if key.endswith(".attn"):
                n_heads = model.config.num_attention_heads
                per_head = product.reshape(*product.shape[:-1], n_heads, -1).sum((0, 1, 3))
                scores.update({f"{key}.{h}": value.item() for h, value in enumerate(per_head)})
            else:
                scores[key] = product.sum().item()
        names = node_names(model)
        result = {
            "node_names": names,
            "scores": [scores[name] for name in names],
            "metric": metric.item(),
            "corrupt_metric": corrupt_metric.item(),
            "metric_name": "label_logit_margin" if negative is not None else "answer_logprob_nats",
            "details": {
                "answer_tokens": selected.sum().item(),
                "sequence_tokens": visible.sum().item(),
                "score_sign": "corrupt_minus_clean",
                "alignment": "exact_token_positions",
                "boundary_padding_removed": len(clean_ids) - len(clean),
                "logit_projection": "answer_predicting_positions_only",
            },
        }
        if not torch.isfinite(torch.tensor([result["metric"], result["corrupt_metric"]])).all():
            raise FloatingPointError("Nonfinite answer metric")
        return result


def patch_pair(
    model: nn.Module,
    clean_ids: Sequence[int] | Tensor,
    corrupt_ids: Sequence[int] | Tensor,
    answer_mask: Sequence[bool] | Tensor,
    nodes: Sequence[str],
    *,
    scale: float = 1.0,
    device: str | torch.device | None = None,
    negative_ids: Sequence[int] | Tensor | None = None,
    attention_mask: Sequence[int] | Tensor | None = None,
) -> dict[str, float]:
    """Replace selected clean node activations with aligned corrupt activations.

    scale=1 is exact activation patching; small scale estimates the derivative
    for checking attribution. All sequence positions are patched. Simultaneous
    patches use the current clean-trajectory activation at each intervention.
    """
    clean, corrupt, selected, negative, visible = _prepare(
        model, clean_ids, corrupt_ids, answer_mask, device, negative_ids, attention_mask
    )
    if not set(nodes).issubset(node_names(model)) or len(set(nodes)) != len(nodes):
        raise ValueError("Intervention nodes must be unique members of the node universe")
    if not torch.isfinite(torch.tensor(scale)):
        raise ValueError("scale must be finite")
    handles: list[Any] = []
    with _evaluation(model), torch.no_grad():
        with NodeTaps(model) as taps:
            _forward(model, corrupt, visible, selected)
            corrupt_acts = {key: value.clone() for key, value in taps.acts.items()}
        baseline = _metric_forward(model, clean, clean, visible, selected, negative)

        def patch_attention(layer: int, heads: list[int]) -> Any:
            def hook(_module: nn.Module, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
                output = inputs[0].clone()
                shape = (*output.shape[:-1], model.config.num_attention_heads, -1)
                view = output.reshape(shape)
                other = corrupt_acts[f"layer.{layer}.attn"].reshape(shape)
                view[..., heads, :] += scale * (other[..., heads, :] - view[..., heads, :])
                return (output, *inputs[1:])

            return hook

        def patch_mlp(layer: int) -> Any:
            def hook(_module: nn.Module, _inputs: Any, output: Tensor) -> Tensor:
                return output + scale * (corrupt_acts[f"layer.{layer}.mlp"] - output)

            return hook

        try:
            for i, layer in enumerate(model.model.layers):
                heads = [
                    h
                    for h in range(model.config.num_attention_heads)
                    if f"layer.{i}.attn.{h}" in nodes
                ]
                if heads:
                    handles.append(
                        layer.self_attn.o_proj.register_forward_pre_hook(patch_attention(i, heads))
                    )
                if f"layer.{i}.mlp" in nodes:
                    handles.append(
                        layer.post_feedforward_layernorm.register_forward_hook(patch_mlp(i))
                    )
            patched = _metric_forward(model, clean, clean, visible, selected, negative)
        finally:
            for handle in handles:
                handle.remove()
    return {
        "clean_metric": baseline.item(),
        "patched_metric": patched.item(),
        "effect": (patched - baseline).item(),
        "scale": scale,
    }


def extract_residual_features(
    model: nn.Module,
    input_ids: Sequence[int] | Tensor,
    *,
    device: str | torch.device | None = None,
    attention_mask: Sequence[int] | Tensor | None = None,
    position: int | None = None,
) -> Tensor:
    """Return CPU [embedding+layers, hidden] features at one visible position.

    Layer outputs precede the model's final norm. Callers supply prompt-only
    inputs for semantic probes or prompt+candidate for answer validity. Never
    include a gold rationale as context. This function neither appends nor
    generates any answer tokens.
    """
    node_names(model)
    device = device if device is not None else next(model.parameters()).device
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    visible = (
        torch.ones_like(ids)
        if attention_mask is None
        else torch.as_tensor(attention_mask, device=device)
    )
    if ids.ndim != 1 or not ids.numel() or visible.shape != ids.shape:
        raise ValueError("Expected a nonempty 1D sequence with an aligned attention mask")
    if not ((visible == 0) | (visible == 1)).all() or not visible.any():
        raise ValueError("attention_mask must be binary with at least one visible token")
    position = int(visible.nonzero()[-1]) if position is None else position
    if not 0 <= position < len(ids) or not visible[position]:
        raise ValueError("Feature position must select a visible token")
    start, stop = int(visible.nonzero()[0]), int(visible.nonzero()[-1]) + 1
    if not visible[start:stop].all():
        raise ValueError("Only boundary padding is supported")
    ids, visible, position = ids[start:stop], visible[start:stop], position - start
    features: list[Tensor] = []
    handles: list[Any] = []

    def save(_module: nn.Module, _inputs: Any, output: Tensor | tuple[Tensor, ...]) -> None:
        activation = output[0] if isinstance(output, tuple) else output
        features.append(activation[0, position].detach().float().cpu().clone())

    try:
        handles.append(model.get_input_embeddings().register_forward_hook(save))
        handles.extend(layer.register_forward_hook(save) for layer in model.model.layers)
        with _evaluation(model), torch.no_grad():
            # The language-model head is unnecessary for residual probes and
            # would allocate a sequence-by-vocabulary logits tensor.
            model.model(
                input_ids=ids[None],
                attention_mask=visible.long()[None],
                position_ids=torch.arange(len(ids), device=ids.device)[None],
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(features)
