"""Edge-level patching and EAP-IG attribution for HF Llama models.

Graph (see experiments/eapig-circuit-check/PLAN.md, "Frozen decisions"):

- Upstream nodes U: ``embed``; per layer l the per-query-head writes
  ``a{l}.h{i}`` (z_i @ W_o[:, i*dh:(i+1)*dh].T) then ``m{l}`` (MLP output).
- Receivers R: per layer l ``a{l}.q{i}``, ``a{l}.k{g}``, ``a{l}.v{g}``,
  ``m{l}.in``; finally ``logits``.
- Edge u -> r is valid iff u is computed before r is read.

The residual stream at any point is exactly the sum of the upstream node
outputs computed so far (Llama has no attention/MLP biases), so patching an
edge u -> r means replacing u's contribution to r's input:

    in_r = sum_{u available to r} own_u + sum_u (1 - keep[u, r]) (corrupt_u - own_u)

Inputs are unpadded equal-length (B, T) batches; there is no attention mask
other than the causal one.

On/off masks (``keep`` bool, or float with every valid entry in {0, 1}, and not
requiring grad) take a sparse path in ``patched_forward``. Per receiver group
(a layer's q/k/v receivers, its ``m.in``, or ``logits``; read prefix u < n) the
group's base input is ``resid`` (mostly-kept masks) or the all-corrupt sum
``C_n = sum_{u<n} corrupt_u`` (= resid + S_n, mostly-patched masks), and only the
exception edges are added as ``coef[u, r] * D_u`` with D_u = corrupt_u - own_u
(coef = +1 on base resid, -1 on base C_n). Receivers with no exceptions share the
base input: it is layernormed and projected once with the ordinary batched q/k/v
matmuls; per-receiver inputs are built only for receivers with exceptions. A group
whose exceptions exceed ``_DENSE_FRAC`` of its read region uses the dense recipe
(base resid, every receiver, the whole prefix). Fractional or ``requires_grad``
masks keep the original dense path. Setting the environment variable
``GEODE_EAPIG_DENSE=1`` (read per call) forces the original dense path for every
non-all-ones mask, for timing/equivalence checks.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import torch
from torch import Tensor
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

__all__ = [
    "EdgeGraph",
    "answer_logprobs",
    "build_graph",
    "eap_ig_scores",
    "kl_2tok",
    "logit_diff_2tok",
    "node_outputs",
    "patched_forward",
]


@dataclass(frozen=True)
class EdgeGraph:
    """Node/receiver/edge bookkeeping for an L-layer GQA Llama."""

    n_layers: int
    n_heads: int
    n_kv_heads: int

    # ---- index arithmetic -------------------------------------------------
    @property
    def n_upstream(self) -> int:
        return 1 + self.n_layers * (self.n_heads + 1)

    @property
    def n_receivers(self) -> int:
        return self.n_layers * self.recv_per_layer + 1

    @property
    def recv_per_layer(self) -> int:
        return self.n_heads + 2 * self.n_kv_heads + 1

    def head_index(self, layer: int, head: int) -> int:
        return 1 + layer * (self.n_heads + 1) + head

    def mlp_index(self, layer: int) -> int:
        return 1 + layer * (self.n_heads + 1) + self.n_heads

    def q_index(self, layer: int, head: int) -> int:
        return layer * self.recv_per_layer + head

    def k_index(self, layer: int, kv: int) -> int:
        return layer * self.recv_per_layer + self.n_heads + kv

    def v_index(self, layer: int, kv: int) -> int:
        return layer * self.recv_per_layer + self.n_heads + self.n_kv_heads + kv

    def mlp_in_index(self, layer: int) -> int:
        return layer * self.recv_per_layer + self.n_heads + 2 * self.n_kv_heads

    @property
    def logits_index(self) -> int:
        return self.n_layers * self.recv_per_layer

    def n_avail_attn(self, layer: int) -> int:
        """Number of upstream nodes (a prefix of U) read by layer's q/k/v."""
        return 1 + layer * (self.n_heads + 1)

    def n_avail_mlp(self, layer: int) -> int:
        """Number of upstream nodes (a prefix of U) read by m{layer}.in."""
        return 1 + layer * (self.n_heads + 1) + self.n_heads

    # ---- names --------------------------------------------------------------
    @cached_property
    def upstream_names(self) -> list[str]:
        names = ["embed"]
        for layer in range(self.n_layers):
            names += [f"a{layer}.h{i}" for i in range(self.n_heads)]
            names.append(f"m{layer}")
        return names

    @cached_property
    def receiver_names(self) -> list[str]:
        names: list[str] = []
        for layer in range(self.n_layers):
            names += [f"a{layer}.q{i}" for i in range(self.n_heads)]
            names += [f"a{layer}.k{g}" for g in range(self.n_kv_heads)]
            names += [f"a{layer}.v{g}" for g in range(self.n_kv_heads)]
            names.append(f"m{layer}.in")
        names.append("logits")
        return names

    @cached_property
    def receiver_layer(self) -> list[int]:
        """Receiving layer per receiver (``logits`` = n_layers)."""
        out = [layer for layer in range(self.n_layers) for _ in range(self.recv_per_layer)]
        return out + [self.n_layers]

    @cached_property
    def receiver_node(self) -> list[str]:
        """Node a receiver belongs to: q -> a{l}.h{i}, k/v -> a{l}.kv{g}, m{l}, logits."""
        names: list[str] = []
        for layer in range(self.n_layers):
            names += [f"a{layer}.h{i}" for i in range(self.n_heads)]
            names += [f"a{layer}.kv{g}" for g in range(self.n_kv_heads)] * 2
            names.append(f"m{layer}")
        names.append("logits")
        return names

    # ---- edges --------------------------------------------------------------
    @cached_property
    def valid(self) -> Tensor:
        """Boolean (U, R) mask of valid edges."""
        mask = torch.zeros(self.n_upstream, self.n_receivers, dtype=torch.bool)
        for layer in range(self.n_layers):
            base = layer * self.recv_per_layer
            n_attn = self.n_heads + 2 * self.n_kv_heads
            mask[: self.n_avail_attn(layer), base : base + n_attn] = True
            mask[: self.n_avail_mlp(layer), self.mlp_in_index(layer)] = True
        mask[:, self.logits_index] = True
        return mask

    @cached_property
    def n_edges(self) -> int:
        return int(self.valid.sum())

    @cached_property
    def edge_ur(self) -> Tensor:
        """(n_edges, 2) long tensor of (u, r), row-major order of ``valid``."""
        return self.valid.nonzero()

    @cached_property
    def _flat_table(self) -> Tensor:
        table = torch.full((self.n_upstream, self.n_receivers), -1, dtype=torch.long)
        table[self.valid] = torch.arange(self.n_edges)
        return table

    def flat_to_ur(self, idx: Tensor) -> tuple[Tensor, Tensor]:
        """Flat edge indices -> (u, r) index tensors."""
        ur = self.edge_ur[torch.as_tensor(idx, dtype=torch.long)]
        return ur[..., 0], ur[..., 1]

    def ur_to_flat(self, u: Tensor, r: Tensor) -> Tensor:
        """(u, r) -> flat edge index; raises if any (u, r) is not a valid edge."""
        out = self._flat_table[
            torch.as_tensor(u, dtype=torch.long), torch.as_tensor(r, dtype=torch.long)
        ]
        if (out < 0).any():
            raise ValueError("ur_to_flat: some (u, r) pairs are not valid edges")
        return out

    def mask_from_flat(self, idx: Tensor) -> Tensor:
        """Flat edge indices -> boolean (U, R) mask with exactly those edges set."""
        mask = torch.zeros(self.n_upstream, self.n_receivers, dtype=torch.bool)
        u, r = self.flat_to_ur(idx)
        mask[u, r] = True
        return mask

    def edge_names(self) -> list[str]:
        up, rc = self.upstream_names, self.receiver_names
        return [f"{up[u]}->{rc[r]}" for u, r in self.edge_ur.tolist()]

    def edge_upstream_node(self) -> list[str]:
        up = self.upstream_names
        return [up[u] for u in self.edge_ur[:, 0].tolist()]

    def edge_receiver_node(self) -> list[str]:
        rn = self.receiver_node
        return [rn[r] for r in self.edge_ur[:, 1].tolist()]

    def edge_receiving_layer(self) -> Tensor:
        """(n_edges,) long: receiving layer per edge (logits = n_layers)."""
        return torch.tensor(self.receiver_layer, dtype=torch.long)[self.edge_ur[:, 1]]

    @classmethod
    def from_config(cls, config) -> EdgeGraph:
        return cls(
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
        )


def build_graph(model) -> EdgeGraph:
    """Edge graph for a ``LlamaForCausalLM`` (asserts the assumptions we rely on)."""
    cfg = model.config
    assert not getattr(cfg, "attention_bias", False), "attention bias breaks the node decomposition"
    assert not getattr(cfg, "mlp_bias", False), "mlp bias breaks the node decomposition"
    assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
    return EdgeGraph.from_config(cfg)


# ---------------------------------------------------------------------------
# Sparse plan for on/off masks
# ---------------------------------------------------------------------------

# A receiver group whose exceptions exceed this fraction of its read region
# (prefix length x receivers) uses the dense recipe.
_DENSE_FRAC = 0.25


def _force_dense() -> bool:
    return os.environ.get("GEODE_EAPIG_DENSE", "") == "1"


@dataclass(frozen=True)
class _Group:
    """Input recipe for one receiver group (a layer's q/k/v, its m.in, or logits).

    Receiver ``cols[j]``'s input is ``base + sum_i coef[i, j] * D[rows[i]]``; every
    other receiver of the group reads ``base`` itself. ``base`` is ``resid``, or
    ``C_n = sum_{u<n} corrupt_u`` when ``base_corrupt``.
    """

    dense: bool
    base_corrupt: bool
    rows: Tensor | slice  # D-buffer slots (dense: the whole prefix, slots == u)
    n_cols: int
    coef: Tensor  # (n_rows, n_cols)
    # attention groups only: per q/k/v (positions in cols, head ids, count)
    parts: tuple[tuple[Tensor, Tensor, int], ...] = ()


@dataclass(frozen=True)
class _Plan:
    groups: list[_Group]  # attn_0, mlp_0, ..., attn_{L-1}, mlp_{L-1}, logits
    n_slots: int  # D-buffer rows (upstream nodes some exception reads)
    # node-chunk start u -> (first slot, local node indices or None for all, count)
    chunks: dict[int, tuple[int, Tensor | None, int]]
    need_corrupt_sum: bool


def _binary_plan(graph: EdgeGraph, keep: Tensor, device, dtype) -> _Plan | None:
    """Exception lists per receiver group, or None if ``keep`` is not 0/1 on valid edges."""
    valid = graph.valid
    k = keep.detach().to("cpu")
    kv = k[valid]
    if not bool(((kv == 0) | (kv == 1)).all()):
        return None
    w = valid & (k == 0)  # edges carrying corrupt_u
    H, KV, U = graph.n_heads, graph.n_kv_heads, graph.n_upstream
    n_attn = H + 2 * KV
    specs = []  # (prefix n, r0, r1, is_attn)
    for layer in range(graph.n_layers):
        r0 = layer * graph.recv_per_layer
        specs.append((graph.n_avail_attn(layer), r0, r0 + n_attn, True))
        rm = graph.mlp_in_index(layer)
        specs.append((graph.n_avail_mlp(layer), rm, rm + 1, False))
    specs.append((U, graph.logits_index, graph.logits_index + 1, False))

    needed = torch.zeros(U, dtype=torch.bool)
    raw = []
    for n, r0, r1, is_attn in specs:
        region = w[:n, r0:r1]
        total, ones = region.numel(), int(region.sum())
        if min(ones, total - ones) > _DENSE_FRAC * total:
            needed[:n] = True
            cols = torch.arange(r1 - r0)
            raw.append((True, False, None, cols, region, is_attn))
            continue
        base_corrupt = 2 * ones > total
        exc = ~region if base_corrupt else region
        rows = exc.any(1).nonzero().squeeze(1)
        cols = exc.any(0).nonzero().squeeze(1)
        needed[rows] = True
        raw.append((False, base_corrupt, rows, cols, exc[rows][:, cols], is_attn))

    slot = needed.long().cumsum(0) - 1
    groups = []
    for dense, base_corrupt, rows, cols, sub, is_attn in raw:
        coef = sub.to(dtype)
        if base_corrupt:
            coef = -coef
        parts = ()
        if is_attn:
            parts = tuple(
                (
                    m.nonzero().squeeze(1).to(device),
                    (cols[m] - off).to(device),
                    int(m.sum()),
                )
                for off, cnt in ((0, H), (H, KV), (H + KV, KV))
                for m in [(cols >= off) & (cols < off + cnt)]
            )
        groups.append(
            _Group(
                dense=dense,
                base_corrupt=base_corrupt,
                rows=slice(0, sub.shape[0]) if dense else slot[rows].to(device),
                n_cols=len(cols),
                coef=coef.to(device),
                parts=parts,
            )
        )

    chunks: dict[int, tuple[int, Tensor | None, int]] = {}
    starts = [(0, 1)]
    for layer in range(graph.n_layers):
        starts += [(graph.head_index(layer, 0), H), (graph.mlp_index(layer), 1)]
    for s, c in starts:
        sel = needed[s : s + c].nonzero().squeeze(1)
        if len(sel):
            local = None if len(sel) == c else sel.to(device)
            chunks[s] = (int(slot[s + sel[0]]), local, len(sel))
    return _Plan(
        groups=groups,
        n_slots=int(needed.sum()),
        chunks=chunks,
        need_corrupt_sum=any(g.base_corrupt for g in groups),
    )


# ---------------------------------------------------------------------------
# Forward pass with per-receiver inputs
# ---------------------------------------------------------------------------


def _run(
    model,
    graph: EdgeGraph,
    embeds: Tensor,
    corrupt_acts: Tensor | None,
    keep: Tensor | None,
    record_nodes: bool = False,
    collect_inputs: bool = False,
    last_n: int | None = None,
    plan: _Plan | None = None,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """Core forward from input embeddings (B, T, d).

    Returns (logits, node outputs as chunks [(B,T,c,d)...] if record_nodes,
    receiver inputs [attn (B,T,H+2KV,d), mlp (B,T,d)] per layer + logits
    input (B,T,d) if collect_inputs). ``keep`` selects the dense patched path,
    ``plan`` (from ``_binary_plan``, with ``keep=None``) the sparse one. Logits
    cover the last ``last_n`` positions only when given.
    """
    inner = model.model
    B, T, d = embeds.shape
    H, KV, L = graph.n_heads, graph.n_kv_heads, graph.n_layers
    n_attn = H + 2 * KV
    patched = keep is not None
    sparse = plan is not None
    if patched or sparse:
        assert corrupt_acts is not None
        assert corrupt_acts.dim() == 4 and corrupt_acts.shape[2:] == (graph.n_upstream, d), (
            corrupt_acts.shape
        )
    if patched:
        assert not sparse
        assert keep.shape == (graph.n_upstream, graph.n_receivers), keep.shape
        w = 1.0 - keep.to(embeds.dtype)  # (U, R): weight on (corrupt - own)
    if sparse:
        assert not collect_inputs
        d_buf = embeds.new_empty(plan.n_slots, B, T, d)  # D rows by slot, (slot, B, T, d)
    csum: Tensor | None = None  # sparse: C_n = sum of corrupt_u over computed nodes
    if last_n is not None:
        assert last_n >= 1, last_n

    position_ids = torch.arange(T, device=embeds.device).unsqueeze(0).expand(B, T)
    cos, sin = inner.rotary_emb(embeds, position_ids)
    causal = torch.ones(T, T, dtype=torch.bool, device=embeds.device).triu(1)

    resid = embeds
    nodes: list[Tensor] = []
    inputs: list[Tensor] = []
    # D chunks (corrupt - own) with their upstream start index, for patched runs
    d_chunks: list[tuple[int, Tensor]] = []
    if patched:
        d_chunks.append((0, corrupt_acts[:, :, 0:1] - embeds.unsqueeze(2)))
    if record_nodes:
        nodes.append(embeds.unsqueeze(2))

    def patch_delta(n_avail: int, r_slice: slice) -> Tensor:
        total = None
        for start, dc in d_chunks:
            c = dc.shape[2]
            if start >= n_avail:
                break
            c = min(c, n_avail - start)
            term = torch.einsum("btcd,cr->btrd", dc[:, :, :c], w[start : start + c, r_slice])
            total = term if total is None else total + term
        return total

    def store(start: int, own: Tensor) -> None:
        """Sparse: add node chunk [start, start+c) to C_n and its needed D rows to d_buf."""
        nonlocal csum
        cor = corrupt_acts[:, :, start : start + own.shape[2]]
        if plan.need_corrupt_sum:
            s = cor.sum(2)
            csum = s if csum is None else csum + s
        spec = plan.chunks.get(start)
        if spec is None:
            return
        a, local, cnt = spec
        if local is not None:
            cor, own = cor[:, :, local], own[:, :, local]
        torch.sub(cor, own, out=d_buf[a : a + cnt].permute(1, 2, 0, 3))

    def group_base(g: _Group, t0: int = 0) -> Tensor:
        return csum.expand(B, T, d)[:, t0:] if g.base_corrupt else resid[:, t0:]

    def group_delta(g: _Group, t0: int = 0) -> Tensor:
        """(B, T - t0, n_cols, d): sum_i coef[i, j] D[rows[i]] at positions >= t0."""
        buf = d_buf if t0 == 0 else d_buf[:, :, t0:]
        return torch.einsum("ubtd,uc->btcd", buf[g.rows], g.coef)

    if sparse:
        store(0, embeds.unsqueeze(2))

    for layer_idx, layer in enumerate(inner.layers[:L]):
        attn = layer.self_attn
        dh = attn.head_dim
        base = layer_idx * graph.recv_per_layer

        # ---- attention receivers: one input per q/k/v head ----
        if sparse:
            g = plan.groups[2 * layer_idx]
            xs = group_base(g)
            xen = layer.input_layernorm(xs.unsqueeze(2) + group_delta(g)) if g.n_cols else None
            xsn = layer.input_layernorm(xs) if g.n_cols < n_attn else None
            qkv = []
            for proj, cnt, (pos, heads, n_e) in zip(
                (attn.q_proj, attn.k_proj, attn.v_proj), (H, KV, KV), g.parts
            ):
                if n_e < cnt:  # shared input for receivers without exceptions
                    y = proj(xsn).view(B, T, cnt, dh)
                else:
                    y = xen.new_empty(B, T, cnt, dh)
                if n_e:
                    wx = proj.weight.view(cnt, dh, d)[heads]
                    y[:, :, heads] = torch.einsum("btnd,ned->btne", xen[:, :, pos], wx)
                qkv.append(y.transpose(1, 2))
            q, k, v = qkv
        else:
            x = resid.unsqueeze(2).expand(B, T, n_attn, d)
            if patched:
                x = x + patch_delta(graph.n_avail_attn(layer_idx), slice(base, base + n_attn))
            if collect_inputs:
                inputs.append(x)
            xn = layer.input_layernorm(x)
            wq = attn.q_proj.weight.view(H, dh, d)
            wk = attn.k_proj.weight.view(KV, dh, d)
            wv = attn.v_proj.weight.view(KV, dh, d)
            q = torch.einsum("bthd,hed->bhte", xn[:, :, :H], wq)
            k = torch.einsum("bthd,hed->bhte", xn[:, :, H : H + KV], wk)
            v = torch.einsum("bthd,hed->bhte", xn[:, :, H + KV :], wv)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        groups = H // KV
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * attn.scaling
        scores = scores.masked_fill(causal, float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        z = torch.matmul(probs, v)  # (B, H, T, dh)
        wo = attn.o_proj.weight.view(d, H, dh)
        heads = torch.einsum("bhte,dhe->bthd", z, wo)  # (B, T, H, d)
        resid = resid + heads.sum(2)
        if record_nodes:
            nodes.append(heads)
        if patched:
            start = graph.head_index(layer_idx, 0)
            d_chunks.append((start, corrupt_acts[:, :, start : start + H] - heads))
        if sparse:
            store(graph.head_index(layer_idx, 0), heads)

        # ---- MLP receiver ----
        r_mlp = graph.mlp_in_index(layer_idx)
        if sparse:
            g = plan.groups[2 * layer_idx + 1]
            xm = group_base(g)
            if g.n_cols:
                xm = xm + group_delta(g).squeeze(2)
        elif patched:
            xm = resid + patch_delta(graph.n_avail_mlp(layer_idx), slice(r_mlp, r_mlp + 1)).squeeze(
                2
            )
        else:
            # distinct autograd node: grad wrt xm must exclude resid's skip path
            xm = resid.view_as(resid)
        if collect_inputs:
            inputs.append(xm)
        mlp_out = layer.mlp(layer.post_attention_layernorm(xm))
        resid = resid + mlp_out
        if record_nodes:
            nodes.append(mlp_out.unsqueeze(2))
        if patched:
            m = graph.mlp_index(layer_idx)
            d_chunks.append((m, corrupt_acts[:, :, m : m + 1] - mlp_out.unsqueeze(2)))
        if sparse:
            store(graph.mlp_index(layer_idx), mlp_out.unsqueeze(2))

    # ---- logits receiver ----
    if sparse:
        g = plan.groups[-1]
        t0 = 0 if last_n is None else max(T - last_n, 0)
        xl = group_base(g, t0)
        if g.n_cols:
            xl = xl + group_delta(g, t0).squeeze(2)
    else:
        xl = resid
        if patched:
            r = graph.logits_index
            xl = xl + patch_delta(graph.n_upstream, slice(r, r + 1)).squeeze(2)
        if collect_inputs:
            inputs.append(xl)
        if last_n is not None:
            xl = xl[:, -last_n:]
    logits = model.lm_head(inner.norm(xl))
    return logits, nodes, inputs


def _embed(model, input_ids: Tensor) -> Tensor:
    assert input_ids.dim() == 2, "input_ids must be an unpadded (B, T) batch"
    return model.model.embed_tokens(input_ids)


def patched_forward(
    model,
    graph: EdgeGraph,
    input_ids: Tensor,
    corrupt_acts: Tensor,
    keep: Tensor,
    *,
    last_n: int | None = None,
) -> Tensor:
    """Logits (B, T, V) of a run where edge u->r carries corrupt_u where keep[u, r] == 0.

    ``keep``: float or bool (U, R); invalid-edge entries are ignored. 0/1 masks not
    requiring grad take the sparse path (see module docstring).
    ``corrupt_acts``: broadcastable to (B, T, U, d) -- ``node_outputs`` of a
    counterfactual run, or (1, 1, U, d) / (1, T, U, d) means.
    ``last_n``: if given, logits (B, last_n, V) for the last ``last_n`` positions only.
    """
    embeds = _embed(model, input_ids)
    fast = not keep.requires_grad and bool((keep[graph.valid.to(keep.device)].float() == 1).all())
    if fast:
        logits, _, _ = _run(model, graph, embeds, None, None, last_n=last_n)
        return logits
    corrupt = corrupt_acts.to(embeds.dtype)
    plan = None
    if not keep.requires_grad and not _force_dense():
        plan = _binary_plan(graph, keep, embeds.device, embeds.dtype)
    if plan is None:
        logits, _, _ = _run(model, graph, embeds, corrupt, keep, last_n=last_n)
    else:
        logits, _, _ = _run(model, graph, embeds, corrupt, None, last_n=last_n, plan=plan)
    return logits


def node_outputs(model, graph: EdgeGraph, input_ids: Tensor) -> Tensor:
    """Per-node writes (B, T, U, d) of an unpatched run, in ``graph.upstream_names`` order."""
    _, nodes, _ = _run(model, graph, _embed(model, input_ids), None, None, record_nodes=True)
    return torch.cat(nodes, dim=2)


# ---------------------------------------------------------------------------
# EAP-IG
# ---------------------------------------------------------------------------


def logit_diff_2tok(
    logits: Tensor, c1: Tensor, k1: Tensor, c2: Tensor, k2: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Two-token teacher-forced logit difference; sequence = prompt + first answer token.

    term1 = logits[:, -2, c1] - logits[:, -2, k1]; term2 = logits[:, -1, c2] - logits[:, -1, k2].
    Returns (term1 + term2, term1, term2), each (B,).
    """
    rows = torch.arange(logits.shape[0], device=logits.device)
    a, b = logits[:, -2], logits[:, -1]
    term1 = a[rows, c1] - a[rows, k1]
    term2 = b[rows, c2] - b[rows, k2]
    return term1 + term2, term1, term2


def answer_logprobs(logits: Tensor) -> Tensor:
    """(B, T, V) logits -> (B, 2, V) float32 log-softmax at positions -2 and -1 (the two answer predictions)."""
    return torch.log_softmax(logits[:, -2:].float(), dim=-1)


def kl_2tok(logits: Tensor, clean_logp: Tensor) -> Tensor:
    """Per-example KL(p_clean || p_x) in nats over the full vocabulary, summed over the two answer positions.

    ``logits`` (B, T, V) are the patched/any run; ``clean_logp`` (B, 2, V) comes from
    ``answer_logprobs`` of the clean full-model logits. Returns (B,) float32.
    Differentiable in ``logits``; no clamping. Use ``-kl_2tok(...)`` as the EAP-IG metric.
    """
    logp_x = answer_logprobs(logits)
    clean_logp = clean_logp.float()
    return (clean_logp.exp() * (clean_logp - logp_x)).sum(-1).sum(-1)


def eap_ig_scores(
    model,
    graph: EdgeGraph,
    clean_ids: Tensor,
    corrupt_ids: Tensor,
    metric: Callable[[Tensor], Tensor],
    steps: int = 5,
) -> Tensor:
    """Per-example EAP-IG (inputs variant) edge scores, (B, n_edges) float32.

    score[b, u->r] = sum_t (clean_u - corrupt_u)[b, t] . mean_k dm/d in_r at
    input embeddings emb(corrupt) + (k/steps) (emb(clean) - emb(corrupt)),
    k = 1..steps. Positive = edge supports the metric.
    """
    assert clean_ids.shape == corrupt_ids.shape and clean_ids.dim() == 2
    assert steps >= 1
    B = clean_ids.shape[0]
    with torch.no_grad():
        delta = node_outputs(model, graph, clean_ids) - node_outputs(model, graph, corrupt_ids)
        emb_clean = _embed(model, clean_ids)
        emb_corrupt = _embed(model, corrupt_ids)
    U, R = graph.n_upstream, graph.n_receivers
    scores = torch.zeros(B, U, R, dtype=torch.float32, device=delta.device)
    H, KV = graph.n_heads, graph.n_kv_heads
    n_attn = H + 2 * KV

    for k in range(1, steps + 1):
        alpha = k / steps
        x = (emb_corrupt + alpha * (emb_clean - emb_corrupt)).detach().requires_grad_(True)
        with torch.enable_grad():
            logits, _, inputs = _run(model, graph, x, None, None, collect_inputs=True)
            grads = torch.autograd.grad(metric(logits).sum(), inputs)
        with torch.no_grad():
            for layer in range(graph.n_layers):
                g_attn, g_mlp = grads[2 * layer], grads[2 * layer + 1]
                base = layer * graph.recv_per_layer
                n = graph.n_avail_attn(layer)
                scores[:, :n, base : base + n_attn] += torch.einsum(
                    "btud,btrd->bur", delta[:, :, :n], g_attn
                ).float()
                n = graph.n_avail_mlp(layer)
                scores[:, :n, graph.mlp_in_index(layer)] += torch.einsum(
                    "btud,btd->bu", delta[:, :, :n], g_mlp
                ).float()
            scores[:, :, graph.logits_index] += torch.einsum(
                "btud,btd->bu", delta, grads[-1]
            ).float()
        del grads, inputs, logits

    scores /= steps
    return scores[:, graph.valid.to(scores.device)]
