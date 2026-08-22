"""Node- and edge-level mechanistic primitives for the ts38mt Tier-2/3 drivers
(test 4 cross-model patching, test 2 circuit Jaccard, test 3 node-vs-edge ΔS,
test 5 DCM — decisions.md 2026-08-21 night "ts38mt pre-registration"). Shared
foundation only: no argparse ``main``, no output table — the three drivers
that import this module own their own CLI/table/print_summary, matching
``mech_lib.py``'s own role for the Phase-0 drivers.

**Nodes of a Llama-style block i.** Attention heads ``h = 0..H-1`` (``H =
model.config.num_attention_heads``, ``d_head = hidden_size // H`` or
``config.head_dim`` when the config carries one) and the MLP. A head's
activation is the ``[b, s, d_head]`` slice ``[..., h*d_head:(h+1)*d_head]``
of the INPUT to ``self_attn.o_proj`` (hooked via
``register_forward_pre_hook`` — GQA only widens/narrows K/V, never the
o_proj input, which stays exactly ``H*d_head`` wide; ``_num_heads_and_dhead``
asserts this at every call site that needs it). Its write to the residual
stream is ``o_proj.weight[:, h*d_head:(h+1)*d_head] @ head_act`` (LoRA-aware:
see ``_o_proj_weight_bias``). The MLP's activation IS its write: the output
of the ``mlp`` module, ``[b, s, d_model]`` (hooked via
``register_forward_hook``).

``LoRALinear`` (``geode/train/lora.py``) wraps ``o_proj``/``down_proj``/etc.
on target-run trees; hooks on the WRAPPER see the same input/output tensors
a plain ``nn.Linear`` would (``forward(x) = base(x) + scaling*B(A(x))``, the
same ``x`` feeds every term), so every hook in this module works unchanged
on a LoRA-wrapped tree. Only the WEIGHT-based computations (the head's
residual-write) need the LoRA/full-FT dispatch, via ``_o_proj_weight_bias``
(``adapters.lora_delta_w``'s scaling, the same convention ``weight_diff.py``
uses).

**Node ids.** ``NodeId(layer=i, kind="head"|"mlp", index=h|0)``, frozen +
orderable (sorts layer-major, then kind alphabetically — "head" < "mlp", so
a layer's heads sort before its MLP, execution order — then index).
``str(NodeId(i,"head",h)) == f"a{i}.h{h}"``, ``str(NodeId(i,"mlp",0)) ==
f"m{i}"``; ``NodeId.parse`` inverts both. A third kind, ``"attn"``
(``str -> f"a{i}"``, no ``.h{h}`` suffix), exists ONLY as an edge-attribution
SINK identity (the ``v`` side of an edge, "the whole block's attention
input point") — ``capture_nodes``/``patch_nodes`` never produce or accept
it; only ``edge_attribution_scores`` does. Don't confuse ``a{i}`` (attn
sink, whole block) with ``a{i}.h{h}`` (one head).

**Residual-hook layer convention.** This module never numbers "layers" the
way ``mech_lib.capture_residuals`` does (0=embed, i+1=block i); it always
addresses blocks by their own 0-based transformer index, matching
``weight_diff.py``'s convention, because every quantity here (nodes, LN
inputs) lives strictly INSIDE a block, never at the embed/final-norm
boundary the residual-hook convention exists to name. ``patch_residual`` is
the one exception (it patches the same points ``mech_lib.capture_residuals``
captures) and uses that module's 0=embed / i+1=block-i convention explicitly,
documented on the function itself.

**Metric.** Unless stated otherwise, "the metric" is log p(target) in nats
at a given position — the same quantity ``logit_lens.py`` calls
``first_answer``: position ``label_span[0] - 1``, target
``input_ids[label_span[0]]`` (the first answer token). ``answer_targets``
derives both from a batch of ``geode.arith.spans.SftExample``.

**Edge attribution (EAP-style).** ``score(u -> v) = Δout_u · ∂metric/∂in_v``
where ``in_v`` is the residual tensor entering ``v``'s first op (a block's
``input_layernorm`` for its attention, ``post_attention_layernorm`` for its
MLP, the final norm for the sentinel sink ``"out"``) and ``Δout_u`` is node
``u``'s residual-write delta under the ablation (head:
``W_eff[:, slice] @ Δa``, mlp: ``Δout`` directly — the same ``Δa =
a_ablate - a_clean`` ``attribution_scores`` uses). Approximation, stated
once here rather than at every call site: this is a FIRST-ORDER estimate
(gradient at the clean run times a finite ablation delta, not the exact
ablated-model metric), and it treats each downstream module's own LayerNorm
as part of that module (``in_v`` is measured BEFORE the LN, so the LN's
own nonlinearity is folded into "computing v", never separately
attributed) — the standard EAP approximation. The gradient used for
``∂metric/∂in_v`` is the DIRECT-path gradient at that exact point: the LN's
input tensor is CLONED before ``retain_grad()`` (``_direct_path_grads``),
because on an additive residual stream the SAME tensor object also feeds
the residual add carrying it to every later block — retaining grad on the
un-cloned tensor would give the TOTAL downstream gradient (sum over EVERY
future consumer), not the one path ``v`` alone contributes. This is what
makes the EAP consistency identity (``Σ_v score(u->v) ≈ node score(u)``,
see ``edge_attribution_scores``'s docstring) hold to floating-point
precision rather than only approximately — a point worth flagging because
naive residual-stream gradient hooking gets this wrong silently.
``capture_residual_inputs`` (the forward-only, no-grad sibling of that
capture) is unaffected by this distinction — its VALUES are identical
either way, only a gradient taken on the un-cloned tensor would be wrong.

Sibling-import convention matches ``mech_lib.py``/``weight_diff.py``: bare
``from mech_lib import ...`` / ``from adapters import ...``, the launcher
contract is ``cd analysis/ && python3 <script>.py``, and
``tests/_scriptloader.py`` puts this directory on ``sys.path`` for tests.

Device-agnostic throughout (no hardcoded ``"cuda"``); activations are cast
to fp32 on capture, final reductions (attribution/edge scores) are done in
fp64. ``_EPS`` (1e-30) guards every denominator this module divides by.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from adapters import lora_delta_w  # sibling import, same convention as weight_diff.py
from mech_lib import final_norm_and_head, pad_examples, residual_modules

from geode.arith.spans import SftExample
from geode.train.lora import LoRALinear

_EPS = 1e-30
_VALID_KINDS = ("head", "mlp", "attn")


# --------------------------------------------------------------------------
# NodeId
# --------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class NodeId:
    """A node of block ``layer``: ``kind="head"`` (``index`` = head 0..H-1),
    ``kind="mlp"`` (``index`` must be 0), or ``kind="attn"`` (edge-sink only,
    ``index`` must be 0 — see module docstring). Orderable (layer, kind,
    index) lexicographically, so a sorted list of a capture's keys reads
    layer-major with each layer's heads before its MLP.
    """

    layer: int
    kind: str
    index: int

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"NodeId: kind must be one of {_VALID_KINDS}, got {self.kind!r}")
        if self.layer < 0:
            raise ValueError(f"NodeId: layer must be >= 0, got {self.layer}")
        if self.index < 0:
            raise ValueError(f"NodeId: index must be >= 0, got {self.index}")
        if self.kind in ("mlp", "attn") and self.index != 0:
            raise ValueError(f"NodeId: kind={self.kind!r} requires index=0, got {self.index}")

    def __str__(self) -> str:
        if self.kind == "head":
            return f"a{self.layer}.h{self.index}"
        if self.kind == "mlp":
            return f"m{self.layer}"
        return f"a{self.layer}"  # kind == "attn"

    @staticmethod
    def parse(s: str) -> "NodeId":
        """Inverse of ``str()``. Raises ``ValueError`` on anything that
        doesn't match ``a{i}.h{h}`` (head), ``m{i}`` (mlp), or ``a{i}``
        (attn edge-sink)."""
        m = re.fullmatch(r"a(\d+)\.h(\d+)", s)
        if m:
            return NodeId(int(m.group(1)), "head", int(m.group(2)))
        m = re.fullmatch(r"m(\d+)", s)
        if m:
            return NodeId(int(m.group(1)), "mlp", 0)
        m = re.fullmatch(r"a(\d+)", s)
        if m:
            return NodeId(int(m.group(1)), "attn", 0)
        raise ValueError(
            f"NodeId.parse: {s!r} matches none of 'a{{i}}.h{{h}}' / 'm{{i}}' / 'a{{i}}'"
        )


def _node_pos(x: "NodeId | str", n_layers: int) -> int:
    """Total order over writer nodes + edge sinks: attention of block i is
    ``2i``, MLP of block i is ``2i+1``, ``"out"`` is ``2*n_layers`` (after
    everything). ``u`` is strictly upstream of ``v`` iff ``_node_pos(u) <
    _node_pos(v)`` — a head/mlp WRITER's position uses the same block-attn /
    block-mlp slot its write lands on."""
    if isinstance(x, str):
        if x != "out":
            raise ValueError(f"_node_pos: unknown sink string {x!r} (only 'out' is valid)")
        return 2 * n_layers
    if x.kind == "mlp":
        return 2 * x.layer + 1
    if x.kind in ("head", "attn"):
        return 2 * x.layer
    raise ValueError(f"_node_pos: unknown NodeId kind {x.kind!r}")


# --------------------------------------------------------------------------
# Llama-tree plumbing (mirrors mech_lib's asserts; refuses loudly off-tree)
# --------------------------------------------------------------------------


def _o_proj_module(block: nn.Module) -> nn.Module:
    attn = getattr(block, "self_attn", None)
    o_proj = getattr(attn, "o_proj", None) if attn is not None else None
    if o_proj is None:
        raise ValueError(
            "mech_nodes: expected a Llama-style decoder layer with "
            f"self_attn.o_proj; got {type(block).__name__}"
        )
    return o_proj


def _mlp_module(block: nn.Module) -> nn.Module:
    mlp = getattr(block, "mlp", None)
    if mlp is None:
        raise ValueError(
            f"mech_nodes: expected a Llama-style decoder layer with .mlp; got {type(block).__name__}"
        )
    return mlp


def _input_layernorm(block: nn.Module) -> nn.Module:
    ln = getattr(block, "input_layernorm", None)
    if ln is None:
        raise ValueError(
            "mech_nodes: expected a Llama-style decoder layer with .input_layernorm; "
            f"got {type(block).__name__}"
        )
    return ln


def _post_attn_layernorm(block: nn.Module) -> nn.Module:
    ln = getattr(block, "post_attention_layernorm", None)
    if ln is None:
        raise ValueError(
            "mech_nodes: expected a Llama-style decoder layer with .post_attention_layernorm; "
            f"got {type(block).__name__}"
        )
    return ln


def _num_heads_and_dhead(model: nn.Module) -> tuple[int, int]:
    """``(H, d_head)`` from ``model.config``, asserting the o_proj input is
    exactly ``H*d_head`` wide (module docstring: GQA only affects K/V)."""
    config = getattr(model, "config", None)
    n_heads = getattr(config, "num_attention_heads", None)
    hidden = getattr(config, "hidden_size", None)
    if config is None or n_heads is None or hidden is None:
        raise ValueError(
            "mech_nodes: expected a Llama-style causal LM with config.num_attention_heads "
            f"and config.hidden_size; got {type(model).__name__}"
        )
    d_head = getattr(config, "head_dim", None) or hidden // n_heads
    _, blocks = residual_modules(model)
    o_proj = _o_proj_module(blocks[0])
    in_features = o_proj.base.in_features if isinstance(o_proj, LoRALinear) else o_proj.in_features
    if in_features != n_heads * d_head:
        raise ValueError(
            f"mech_nodes: o_proj.in_features={in_features} != num_attention_heads*d_head="
            f"{n_heads}*{d_head}={n_heads * d_head} — GQA affects K/V width only, the o_proj "
            "input must stay num_attention_heads*d_head wide"
        )
    return n_heads, d_head


def _o_proj_weight_bias(o_proj: nn.Module) -> tuple[Tensor, Tensor | None]:
    """Effective ``(weight, bias)`` of an o_proj module: LoRA-aware (``W_eff =
    base.weight + scaling*B.weight@A.weight`` via ``adapters.lora_delta_w``,
    the same scaling ``weight_diff.py`` uses) or the plain ``nn.Linear``
    tensors. fp32. ``bias`` is ``None`` when the module has none (Llama's
    default) — callers must not assume every o_proj has a bias.
    """
    if isinstance(o_proj, LoRALinear):
        base = o_proj.base
        w = base.weight.detach().to(torch.float32) + lora_delta_w(
            o_proj.B.weight.detach().to(torch.float32),
            o_proj.A.weight.detach().to(torch.float32),
            o_proj.alpha,
            o_proj.rank,
        )
        bias = base.bias.detach().to(torch.float32) if base.bias is not None else None
        return w, bias
    bias = o_proj.bias.detach().to(torch.float32) if o_proj.bias is not None else None
    return o_proj.weight.detach().to(torch.float32), bias


# --------------------------------------------------------------------------
# Forward-only captures
# --------------------------------------------------------------------------


def capture_nodes(
    model: nn.Module, input_ids: Tensor, attention_mask: Tensor
) -> dict[NodeId, Tensor]:
    """Forward-only per-node activations: heads ``[b,s,d_head]`` (the
    o_proj-input slice), mlp ``[b,s,d_model]`` (the mlp output). fp32, no
    grad, hooks always removed. Keys are sorted (layer-major, heads before
    each layer's mlp)."""
    n_heads, d_head = _num_heads_and_dhead(model)
    _, blocks = residual_modules(model)
    captured: dict[NodeId, Tensor] = {}
    handles = []
    for i, block in enumerate(blocks):
        o_proj = _o_proj_module(block)
        mlp = _mlp_module(block)

        def make_o_hook(layer: int):
            def hook(_module, args):
                x = args[0].detach().to(torch.float32)
                for h in range(n_heads):
                    captured[NodeId(layer, "head", h)] = x[
                        ..., h * d_head : (h + 1) * d_head
                    ].clone()

            return hook

        def make_mlp_hook(layer: int):
            def hook(_module, _args, output):
                captured[NodeId(layer, "mlp", 0)] = output.detach().to(torch.float32).clone()

            return hook

        handles.append(o_proj.register_forward_pre_hook(make_o_hook(i)))
        handles.append(mlp.register_forward_hook(make_mlp_hook(i)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask.long())
    finally:
        for h in handles:
            h.remove()
    expected = len(blocks) * (n_heads + 1)
    if len(captured) != expected:
        raise RuntimeError(f"capture_nodes: hooks fired for {len(captured)}/{expected} nodes")
    return {nid: captured[nid] for nid in sorted(captured)}


def capture_residual_inputs(
    model: nn.Module, input_ids: Tensor, attention_mask: Tensor
) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    """Forward-only ``(pre_block, pre_mlp)``: ``pre_block[i]`` = input to
    block ``i``'s ``input_layernorm`` (the residual entering its attention),
    ``pre_mlp[i]`` = input to block ``i``'s ``post_attention_layernorm``
    (``pre_block[i] + attn_write_i``). fp32, no grad, hooks always removed.

    These are the node INPUT points ``edge_attribution_scores`` takes
    gradients at — but that function does NOT call this one: it re-hooks the
    same points with an extra ``.clone()`` before ``retain_grad()`` (module
    docstring's edge-attribution section), because the un-cloned tensor here
    is the same object the residual add also carries forward, so a gradient
    taken on it would total every downstream consumer instead of isolating
    one. The VALUES captured here are identical either way — only a gradient
    would differ — so this forward-only capture is safe to use on its own
    wherever no gradient is involved.
    """
    _, blocks = residual_modules(model)
    pre_block: dict[int, Tensor] = {}
    pre_mlp: dict[int, Tensor] = {}
    handles = []
    for i, block in enumerate(blocks):
        ln1 = _input_layernorm(block)
        ln2 = _post_attn_layernorm(block)

        def make_hook(store: dict[int, Tensor], layer: int):
            def hook(_module, args):
                store[layer] = args[0].detach().to(torch.float32).clone()

            return hook

        handles.append(ln1.register_forward_pre_hook(make_hook(pre_block, i)))
        handles.append(ln2.register_forward_pre_hook(make_hook(pre_mlp, i)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask.long())
    finally:
        for h in handles:
            h.remove()
    if len(pre_block) != len(blocks) or len(pre_mlp) != len(blocks):
        raise RuntimeError("capture_residual_inputs: hooks never fired for every block")
    return pre_block, pre_mlp


# --------------------------------------------------------------------------
# Patching
# --------------------------------------------------------------------------


def _normalize_positions(
    positions: Tensor | Sequence[int] | None, device: torch.device
) -> Tensor | None:
    if positions is None:
        return None
    t = (
        positions
        if isinstance(positions, Tensor)
        else torch.tensor(list(positions), dtype=torch.long)
    )
    return t.to(device=device, dtype=torch.long)


def _masked_merge(original: Tensor, replacement: Tensor, positions: Tensor | None) -> Tensor:
    """``replacement`` everywhere if ``positions is None``, else only at each
    example's own position (other positions keep ``original``). Both tensors
    must share ``original``'s exact shape."""
    if replacement.shape != original.shape:
        raise ValueError(
            f"_masked_merge: replacement shape {tuple(replacement.shape)} != "
            f"original shape {tuple(original.shape)}"
        )
    if positions is None:
        return replacement
    b, s = original.shape[0], original.shape[1]
    if positions.shape != (b,):
        raise ValueError(
            f"_masked_merge: positions must have shape ({b},), got {tuple(positions.shape)}"
        )
    mask = torch.zeros(b, s, dtype=torch.bool, device=original.device)
    mask[torch.arange(b, device=original.device), positions.to(original.device)] = True
    while mask.dim() < original.dim():
        mask = mask.unsqueeze(-1)
    return torch.where(mask, replacement, original)


@contextlib.contextmanager
def patch_nodes(
    model: nn.Module,
    replacements: dict[NodeId, Tensor | Callable[[Tensor], Tensor]],
    positions: Tensor | Sequence[int] | None = None,
):
    """Context manager: install hooks replacing every node in
    ``replacements`` (forward pre-hook on o_proj for heads, forward hook on
    mlp for the mlp node) for the duration of the ``with`` block, then remove
    them — ALWAYS, including when the block raises.

    Each value is either a full-shape tensor (``[b,s,d_head]`` for a head,
    ``[b,s,d_model]`` for the mlp — the same shape ``capture_nodes`` returns
    for that node) used directly, or a callable taking the node's current
    (clean, about-to-run) activation and returning a replacement of the same
    shape (e.g. ``lambda x: x + direction`` to plant a signal, or ``lambda x:
    torch.zeros_like(x)`` to ablate).

    ``positions`` is ``None`` (patch every position) or a per-example
    ``Tensor[b]`` of position indices (patch only that one position per
    example; every other position keeps its own clean value — see
    ``_masked_merge``). Only ``"head"``/``"mlp"`` nodes are accepted (an
    ``"attn"``-kind key raises — that kind is edge-sink-only, never a
    writable node).
    """
    for nid in replacements:
        if nid.kind not in ("head", "mlp"):
            raise ValueError(f"patch_nodes: only 'head'/'mlp' nodes are patchable, got {nid}")
    device = next(model.parameters()).device
    pos = _normalize_positions(positions, device)
    n_heads, d_head = _num_heads_and_dhead(model)
    _, blocks = residual_modules(model)

    by_layer_heads: dict[int, dict[int, Tensor | Callable[[Tensor], Tensor]]] = {}
    by_layer_mlp: dict[int, Tensor | Callable[[Tensor], Tensor]] = {}
    for nid, repl in replacements.items():
        if nid.layer >= len(blocks):
            raise ValueError(
                f"patch_nodes: {nid} names layer {nid.layer}, model has {len(blocks)} blocks"
            )
        if nid.kind == "head":
            by_layer_heads.setdefault(nid.layer, {})[nid.index] = repl
        else:
            by_layer_mlp[nid.layer] = repl

    handles = []

    def make_o_hook(layer: int, head_repls: dict[int, Tensor | Callable[[Tensor], Tensor]]):
        def hook(_module, args):
            # Rebuilt FUNCTIONALLY (per-head pieces read from the untouched
            # input, then one ``cat``), never by in-place slice writes into a
            # clone: an in-place write after a value-dependent callable has
            # read the same slice (``lambda a: m * a_T + (1 - m) * a``) makes
            # autograd raise "modified by an inplace operation" at backward —
            # the DCM driver's (test 5) mask optimisation is exactly that case.
            x0 = args[0]
            pieces = []
            for h in range(n_heads):
                orig_slice = x0[..., h * d_head : (h + 1) * d_head]
                if h not in head_repls:
                    pieces.append(orig_slice)
                    continue
                repl = head_repls[h]
                new_val = repl(orig_slice) if callable(repl) else repl
                pieces.append(_masked_merge(orig_slice, new_val, pos))
            return (torch.cat(pieces, dim=-1),) + args[1:]

        return hook

    def make_mlp_hook(repl: Tensor | Callable[[Tensor], Tensor]):
        def hook(_module, _args, output):
            new_val = repl(output) if callable(repl) else repl
            return _masked_merge(output, new_val, pos)

        return hook

    try:
        for layer, head_repls in by_layer_heads.items():
            handles.append(
                _o_proj_module(blocks[layer]).register_forward_pre_hook(
                    make_o_hook(layer, head_repls)
                )
            )
        for layer, repl in by_layer_mlp.items():
            handles.append(_mlp_module(blocks[layer]).register_forward_hook(make_mlp_hook(repl)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def patch_residual(
    model: nn.Module,
    layer: int,
    tensor: Tensor,
    positions: Tensor | Sequence[int] | None = None,
):
    """Context manager: replace the residual-stream output at ``mech_lib``'s
    layer convention (0 = embed output, i+1 = block i's output) for the
    duration of the ``with`` block, removing the hook always. ``tensor`` must
    have the full captured shape ``[b,s,d_model]`` (``mech_lib.capture_
    residuals``'s shape for that layer); ``positions`` restricts the
    replacement the same way ``patch_nodes`` does (``None`` = every
    position, else a per-example ``Tensor[b]``).

    Patching the FINAL layer (``layer == n_layers``) reproduces ``tensor``'s
    own model's final-layer LOGITS exactly only because the two models share
    the same ``model.model.norm``/``lm_head`` downstream of that point — this
    is a claim about the two models sharing everything after ``layer``, not
    a general property of residual patching.
    """
    embed, blocks = residual_modules(model)
    n_layers = len(blocks)
    if not (0 <= layer <= n_layers):
        raise ValueError(f"patch_residual: layer must be in [0, {n_layers}], got {layer}")
    device = next(model.parameters()).device
    pos = _normalize_positions(positions, device)

    def merge_output(output):
        if isinstance(output, tuple):
            return (_masked_merge(output[0], tensor, pos),) + output[1:]
        return _masked_merge(output, tensor, pos)

    def hook(_module, _args, output):
        return merge_output(output)

    module = embed if layer == 0 else blocks[layer - 1]
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------
# Metric + gradients
# --------------------------------------------------------------------------


def answer_logprob(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    positions: Tensor,
    targets: Tensor,
) -> Tensor:
    """log p(target_i) (nats, fp64) at ``positions[i]``, per example
    (``[b]``). Runs the forward pass WITHOUT disabling autograd — wrap in
    ``torch.no_grad()`` at the call site when no backward is wanted (as
    ``cross_patch.py`` does); left enabled here so
    ``answer_logprob_and_node_grads`` can reuse it directly for the backward
    it needs.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask.long()).logits
    b = input_ids.shape[0]
    idx = torch.arange(b, device=logits.device)
    sel = logits[idx, positions.to(logits.device), :]
    logp = torch.log_softmax(sel.to(torch.float64), dim=-1)
    return logp.gather(1, targets.to(device=logits.device)[:, None]).squeeze(1)


def answer_logprob_and_node_grads(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    positions: Tensor,
    targets: Tensor,
) -> tuple[Tensor, dict[NodeId, Tensor]]:
    """``(metric, grads)`` from ONE forward + backward: ``metric`` is
    ``answer_logprob`` (detached, fp64, ``[b]``); ``grads[node]`` is
    ``∂(Σ_i metric_i)/∂(node's own activation)`` for every head/mlp node,
    same shape ``capture_nodes`` returns.

    Params are frozen for the call (every ``requires_grad`` flag saved and
    restored, even on error) and the embedding output is re-rooted
    (``detach().requires_grad_(True)``, ``geode.probe.extract.extract_
    probe``'s pattern) so the graph never depends on any parameter's own
    ``requires_grad`` — this works identically whether the caller's model
    has grad-enabled params (a plain full-FT checkpoint) or frozen ones (a
    LoRA target tree). Node tensors are tapped with ``retain_grad()``
    in-place (the o_proj INPUT as a whole, then sliced per head after
    backward — valid because concatenation-then-slice is linear, so
    ``∂metric/∂head_h`` is exactly the ``h``-th slice of ``∂metric/∂(o_proj
    input)``).

    Per-example independence (why summing the batch before ``.backward()``
    is safe): causal attention + right-padding + no cross-example ops means
    example i's metric depends only on example i's own activations
    (``geode.probe.extract``'s V5.10 argument, reused here), so
    ``∂(Σ_j metric_j)/∂a_u[i] == ∂metric_i/∂a_u[i]`` exactly — the batched
    backward gives every example's own gradient in one pass, not a mixture.
    """
    n_heads, d_head = _num_heads_and_dhead(model)
    embed, blocks = residual_modules(model)
    saved_requires_grad = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved_requires_grad:
        p.requires_grad_(False)

    x_tensors: dict[int, Tensor] = {}
    y_tensors: dict[int, Tensor] = {}
    handles = []

    def embed_hook(_module, _inputs, output):
        return output.detach().requires_grad_(True)

    handles.append(embed.register_forward_hook(embed_hook))
    for i, block in enumerate(blocks):
        o_proj = _o_proj_module(block)
        mlp = _mlp_module(block)

        def make_o_hook(layer: int):
            def hook(_module, args):
                x = args[0]
                x.retain_grad()
                x_tensors[layer] = x

            return hook

        def make_mlp_hook(layer: int):
            def hook(_module, _args, output):
                output.retain_grad()
                y_tensors[layer] = output

            return hook

        handles.append(o_proj.register_forward_pre_hook(make_o_hook(i)))
        handles.append(mlp.register_forward_hook(make_mlp_hook(i)))

    try:
        with torch.enable_grad():
            metric = answer_logprob(model, input_ids, attention_mask, positions, targets)
            metric.sum().backward()
        grads: dict[NodeId, Tensor] = {}
        for i in range(len(blocks)):
            xg = x_tensors[i].grad
            if xg is None:
                raise RuntimeError(
                    f"answer_logprob_and_node_grads: no grad reached o_proj input, layer {i}"
                )
            for h in range(n_heads):
                grads[NodeId(i, "head", h)] = (
                    xg[..., h * d_head : (h + 1) * d_head].detach().to(torch.float32).clone()
                )
            yg = y_tensors[i].grad
            if yg is None:
                raise RuntimeError(
                    f"answer_logprob_and_node_grads: no grad reached mlp output, layer {i}"
                )
            grads[NodeId(i, "mlp", 0)] = yg.detach().to(torch.float32).clone()
    finally:
        for h in handles:
            h.remove()
        for p, rg in saved_requires_grad:
            p.requires_grad_(rg)
    return metric.detach(), {nid: grads[nid] for nid in sorted(grads)}


def _direct_path_grads(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    positions: Tensor,
    targets: Tensor,
) -> dict["NodeId | str", Tensor]:
    """``∂metric/∂in_v`` for every edge-attribution sink ``v``:
    ``NodeId(i,"attn",0)`` (input to block i's ``input_layernorm``),
    ``NodeId(i,"mlp",0)`` (input to block i's ``post_attention_layernorm``),
    ``"out"`` (input to the final norm). See the module docstring's
    edge-attribution section for why each LN's input is CLONED before
    ``retain_grad()`` — the clone's only consumer is that one LN, isolating
    the direct-path gradient ``v`` alone needs, instead of the total
    downstream gradient the un-cloned residual tensor would give (it also
    feeds the residual add carrying it to every later block). The embedding
    output is re-rooted (``answer_logprob_and_node_grads``'s pattern) so the
    graph builds regardless of the model's own parameter ``requires_grad``
    state — without it, block 0's ``input_layernorm`` input traces straight
    back to a frozen ``nn.Embedding`` weight with nothing upstream requiring
    grad, and ``retain_grad()`` on its clone would raise.
    """
    embed, blocks = residual_modules(model)
    norm, _head = final_norm_and_head(model)
    saved_requires_grad = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved_requires_grad:
        p.requires_grad_(False)

    clones: dict["NodeId | str", Tensor] = {}
    handles = []

    def embed_hook(_module, _inputs, output):
        return output.detach().requires_grad_(True)

    handles.append(embed.register_forward_hook(embed_hook))

    def make_clone_hook(key: "NodeId | str"):
        def hook(_module, args):
            c = args[0].clone()
            c.retain_grad()
            clones[key] = c
            return (c,) + args[1:]

        return hook

    for i, block in enumerate(blocks):
        handles.append(
            _input_layernorm(block).register_forward_pre_hook(make_clone_hook(NodeId(i, "attn", 0)))
        )
        handles.append(
            _post_attn_layernorm(block).register_forward_pre_hook(
                make_clone_hook(NodeId(i, "mlp", 0))
            )
        )
    handles.append(norm.register_forward_pre_hook(make_clone_hook("out")))

    try:
        with torch.enable_grad():
            metric = answer_logprob(model, input_ids, attention_mask, positions, targets)
            metric.sum().backward()
        grads: dict["NodeId | str", Tensor] = {}
        for key, t in clones.items():
            if t.grad is None:
                raise RuntimeError(f"_direct_path_grads: no grad reached sink {key!r}")
            grads[key] = t.grad.detach().to(torch.float32).clone()
    finally:
        for h in handles:
            h.remove()
        for p, rg in saved_requires_grad:
            p.requires_grad_(rg)
    return grads


# --------------------------------------------------------------------------
# Attribution patching (node + edge)
# --------------------------------------------------------------------------


def answer_targets(examples: Sequence[SftExample]) -> tuple[Tensor, Tensor]:
    """``(positions, targets)``: ``position[i] = label_span[0]-1``,
    ``target[i] = input_ids[label_span[0]]`` — identical to
    ``logit_lens.position_target_pairs(..., FIRST_ANSWER)``."""
    positions = torch.tensor([ex.label_span[0] - 1 for ex in examples], dtype=torch.long)
    targets = torch.tensor([ex.input_ids[ex.label_span[0]] for ex in examples], dtype=torch.long)
    return positions, targets


def _ablated_acts(
    clean_acts: dict[NodeId, Tensor],
    ablation: str,
    reference_acts: dict[NodeId, Tensor] | None,
) -> dict[NodeId, Tensor]:
    """``a_u^ablate`` per node: ``"zero"`` -> zeros; ``"mean"`` -> the
    per-position mean over ``reference_acts`` (``clean_acts`` itself if
    ``None``), broadcast back to every example in the batch."""
    if ablation == "zero":
        return {nid: torch.zeros_like(a) for nid, a in clean_acts.items()}
    if ablation == "mean":
        ref = reference_acts if reference_acts is not None else clean_acts
        return {
            nid: ref[nid].mean(dim=0, keepdim=True).expand_as(clean_acts[nid]) for nid in clean_acts
        }
    raise ValueError(f"ablation must be 'mean' or 'zero', got {ablation!r}")


def _restrict_positions(per_position: Tensor, positions: str, answer_pos: Tensor) -> Tensor:
    """``[b,s] -> [b]``: ``"answer"`` gathers the single answer position
    (``label_span[0]-1``), ``"all"`` sums every position."""
    if positions == "answer":
        return per_position.gather(1, answer_pos[:, None].to(per_position.device)).squeeze(1)
    if positions == "all":
        return per_position.sum(dim=1)
    raise ValueError(f"positions must be 'answer' or 'all', got {positions!r}")


def attribution_scores(
    model: nn.Module,
    examples_batch: Sequence[SftExample],
    *,
    ablation: str = "mean",
    reference_acts: dict[NodeId, Tensor] | None = None,
    positions: str = "answer",
) -> dict[NodeId, Tensor]:
    """Node attribution patching: ``score(u) = (a_u^ablate - a_u^clean) ·
    ∂metric/∂a_u``, per example (``Tensor[b]``), a first-order estimate of
    the metric change from ablating node ``u``. ``ablation="mean"`` uses the
    per-position mean over ``reference_acts`` (the same batch if omitted);
    ``"zero"`` uses 0. ``positions="answer"`` restricts the dot product's
    position-sum to the single answer position (``label_span[0]-1``);
    ``"all"`` sums over every position. fp64 final reduction.
    """
    if not examples_batch:
        raise ValueError("attribution_scores: empty examples_batch")
    device = next(model.parameters()).device
    input_ids, attention_mask = pad_examples(examples_batch, device=str(device))
    answer_pos, targets = answer_targets(examples_batch)
    answer_pos, targets = answer_pos.to(device), targets.to(device)

    clean_acts = capture_nodes(model, input_ids, attention_mask)
    _metric, grads = answer_logprob_and_node_grads(
        model, input_ids, attention_mask, answer_pos, targets
    )
    ablated = _ablated_acts(clean_acts, ablation, reference_acts)

    scores: dict[NodeId, Tensor] = {}
    for nid in clean_acts:
        delta = (ablated[nid] - clean_acts[nid]).to(torch.float64)
        grad = grads[nid].to(torch.float64)
        per_position = (delta * grad).sum(dim=-1)
        scores[nid] = _restrict_positions(per_position, positions, answer_pos)
    return scores


def _residual_write_deltas(
    model: nn.Module,
    clean_acts: dict[NodeId, Tensor],
    ablated_acts: dict[NodeId, Tensor],
) -> dict[NodeId, Tensor]:
    """``Δout_u`` per node: head -> ``W_eff[:, slice] @ Δa`` (LoRA-aware,
    ``_o_proj_weight_bias``; bias excluded — a constant contributes nothing
    to an ABLATION delta), mlp -> ``Δout`` directly (the mlp node's
    activation IS its write). ``[b,s,d_model]``, fp32."""
    n_heads, d_head = _num_heads_and_dhead(model)
    _, blocks = residual_modules(model)
    deltas: dict[NodeId, Tensor] = {}
    for i, block in enumerate(blocks):
        w, _bias = _o_proj_weight_bias(_o_proj_module(block))
        for h in range(n_heads):
            nid = NodeId(i, "head", h)
            delta_a = (ablated_acts[nid] - clean_acts[nid]).to(torch.float32)
            w_slice = w[:, h * d_head : (h + 1) * d_head]
            deltas[nid] = torch.einsum("od,bsd->bso", w_slice, delta_a)
        mlp_nid = NodeId(i, "mlp", 0)
        deltas[mlp_nid] = (ablated_acts[mlp_nid] - clean_acts[mlp_nid]).to(torch.float32)
    return deltas


def edge_attribution_scores(
    model: nn.Module,
    examples_batch: Sequence[SftExample],
    *,
    ablation: str = "mean",
    reference_acts: dict[NodeId, Tensor] | None = None,
    positions: str = "answer",
) -> dict[tuple[NodeId, "NodeId | str"], Tensor]:
    """EAP-style edge scores: ``score(u->v) = Δout_u · ∂metric/∂in_v`` for
    every ``u`` (head/mlp writer) strictly upstream of every ``v`` (an
    ``"attn"``/``"mlp"``-kind ``NodeId`` sink, or the string ``"out"``) —
    see the module docstring's "Edge attribution" section for the exact
    definition, the LN-as-part-of-``v`` approximation, and why the gradient
    is taken on a CLONED LN input (direct-path, not total-downstream). Only
    upstream pairs appear in the returned dict (module docstring's
    ``_node_pos`` order: attention of block i is upstream of that block's
    own MLP and of every later block; nothing is upstream of itself).

    First-order consistency identity (the key EAP correctness check, tested
    exactly — not just approximately — because of the clone above):
    ``Σ_v score(u->v)`` over every downstream ``v`` (including ``"out"``)
    equals ``attribution_scores(...)[u]`` to floating-point round-off, for
    the SAME ``ablation``/``reference_acts``/``positions`` arguments.
    """
    if not examples_batch:
        raise ValueError("edge_attribution_scores: empty examples_batch")
    device = next(model.parameters()).device
    input_ids, attention_mask = pad_examples(examples_batch, device=str(device))
    answer_pos, targets = answer_targets(examples_batch)
    answer_pos, targets = answer_pos.to(device), targets.to(device)
    n_layers = len(residual_modules(model)[1])

    clean_acts = capture_nodes(model, input_ids, attention_mask)
    ablated_acts = _ablated_acts(clean_acts, ablation, reference_acts)
    write_deltas = _residual_write_deltas(model, clean_acts, ablated_acts)
    sink_grads = _direct_path_grads(model, input_ids, attention_mask, answer_pos, targets)

    scores: dict[tuple[NodeId, "NodeId | str"], Tensor] = {}
    for u, delta in write_deltas.items():
        u_pos = _node_pos(u, n_layers)
        for v, grad in sink_grads.items():
            if u_pos >= _node_pos(v, n_layers):
                continue
            per_position = (delta.to(torch.float64) * grad.to(torch.float64)).sum(dim=-1)
            scores[(u, v)] = _restrict_positions(per_position, positions, answer_pos)
    return scores
