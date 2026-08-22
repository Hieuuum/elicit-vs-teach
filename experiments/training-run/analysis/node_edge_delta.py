"""Node-vs-edge ΔS (ts38mt Tier-3 test 3, gated on Tier 1 finding a latent
sum — decisions.md 2026-08-21 night "ts38mt pre-registration"): did training
change the attribution structure by RE-WEIGHTING the nodes (which
components matter changed) or RE-WIRING the edges between them (how the
same components connect changed), or both?

Given θ0 (``--model-a``) and θ_T (``--model-b``) on a prompt set, this
computes ``mech_nodes.attribution_scores``/``edge_attribution_scores`` per
example (mean-ablation default; ``--ablation``/``--positions``/
``--batch-size`` all pass straight through to ``mech_nodes``), averages over
examples to get vectors ``s0``/``sT`` over nodes and over edges
(``attribution_means``), then reports:

- ``delta_node_l1`` = Σ_u |sT(u) − s0(u)|, ``delta_edge_l1`` likewise over
  edges (``l1_delta``); normalised ``delta_node_rel``/``delta_edge_rel`` =
  the L1 delta divided by Σ(|s0|+|sT|)/2 (``rel_delta``) so the two
  granularities are comparable in units, though NOT in scale — see the
  "node vs. edge scale" note below.
- ``node_rank_spearman``/``edge_rank_spearman`` (``spearman``, ties by
  average rank, torch-only — a local implementation, not shared with
  ``circuit_jaccard.py``'s own spearman: same math, kept independent so
  neither Tier-3 driver depends on the other's file).
- ``node_sign_flip_frac`` = the fraction of nodes, weighted by |sT(u)|,
  whose score's sign flipped between θ0 and θ_T (``sign_flip_frac``; a
  node going to/from exactly zero does not count as a flip — see its
  docstring).
- The structural decomposition (this module's core contribution): for
  every node ``u``, ``edge_change_within_node(u) = Σ_v |sT(u→v) −
  s0(u→v)|`` (the L1 mass its edges moved by) vs ``|Δs_node(u)| = |sT(u) −
  s0(u)|``. By ``mech_nodes.edge_attribution_scores``'s own first-order EAP
  identity (Σ_v score(u→v) = score(u), holding SEPARATELY for θ0 and for
  θ_T), ``Σ_v (sT(u→v) − s0(u→v)) = Δs_node(u)`` exactly (to floating-point
  round-off) — the SIGNED sum of edge deltas always equals the node delta;
  it is only the UNSIGNED (L1) sum that can exceed it, by the triangle
  inequality. ``rewiring_index(u) = 1 − |Δs_node(u)| / Σ_v|Δs_edge(u→v)| ∈
  [0, 1]`` (``rewiring_index``, NaN when the edge-l1 denominator is ≈0):
  0 means every one of ``u``'s edges moved together in the same net
  direction (a PURE RE-WEIGHTING of ``u``'s existing role — it writes
  more/less but reads out through the same downstream paths in the same
  proportions), 1 means ``u``'s edges redistributed with zero net effect on
  its own total score (PURE REWIRING — the same total "amount" of ``u``
  gets routed to different downstream consumers). ``rewiring_index_mean``
  (``rewiring_index_mean``) is the |Δs_edge|-mass-weighted mean over every
  node — nodes that barely changed contribute ~0 weight rather than
  polluting the average with an ill-defined ratio.

**Node/edge/layer conventions**: identical to ``mech_nodes.py`` — nodes are
``NodeId(layer=i, kind="head"|"mlp", index=h|0)``, ``layer`` is the 0-based
transformer BLOCK index (never the residual-hook ``mech_lib`` convention:
this module never touches the residual stream directly, only attribution
scores at node/edge granularity), edges are ``(u, v)`` pairs with ``v`` an
``"attn"``/``"mlp"``-kind ``NodeId`` sink or the string ``"out"``.

**Node vs. edge scale — a caveat, not a bug.** ``delta_node_rel`` and
``delta_edge_rel`` are both dimensionless (each is its own L1 delta over
its own L1 scale), but they are NOT on a comparable absolute scale: a
model has one score per node but H+1 (heads + mlp) times more numbers per
UPSTREAM node's outgoing edges, and every node's score is itself the SUM
of its own outgoing edges (the EAP identity above) — so ``delta_edge_rel
> delta_node_rel`` is the GENERIC expectation from the edge vector simply
having more, and more partially-cancelling, entries, not evidence of
rewiring on its own. The comparison this module is actually built for is
the STRUCTURAL one — ``rewiring_index`` — which asks, PER NODE, whether
ITS OWN edges moved together or apart; that comparison controls for scale
by construction (both the numerator and denominator are that one node's
own numbers). Also note ``rel_delta``'s range is ``[0, 2]``, not ``[0,
1]``: Σ|sT−s0| ≤ Σ(|s0|+|sT|) = 2·denominator, with equality (rel_delta =
2) when θ0 and θ_T scores are equal in magnitude but opposite in sign
everywhere.

**Batching over a length-heterogeneous prompt set.** Real task prompts are
NOT fixed-length (``D_algo_eval_bare.parquet``'s ``full_text`` char lengths
already span 34–51 chars / ~17–24 tokens with this repo's tokenizer,
checked directly against the file) — but ``ablation="mean"`` needs ONE
FIXED per-position reference activation per node, shared by every batch a
given example lands in, or splitting the same examples into different
``--batch-size`` values would silently change which activation "mean"
ablates toward (each batch's own within-batch mean would differ). This
module resolves both constraints together: examples are grouped into
buckets by their own ``len(input_ids)`` (``_bucket_by_length`` — bucket
membership depends only on an example's own length, never on
``--batch-size``), the mean-ablation reference is accumulated once per
bucket over ALL of that bucket's examples (``_reference_acts_mean``,
streamed in ``--batch-size`` chunks so memory stays O(batch), not
O(len(examples)) — mathematically identical to passing every raw
activation as ``reference_acts``, since ``mech_nodes._ablated_acts``'s
``mean(dim=0, keepdim=True)`` is the identity on an already-averaged
``[1,s,d]`` tensor), and then every batch of that bucket reuses that same
fixed reference. The result: ``attribution_means``'s output does not
depend on ``--batch-size`` (tested directly), only on the example set and
which bucket each example falls in.

Separately, note ``--positions="all"`` (``mech_nodes``'s own
``_restrict_positions``) sums a score over EVERY position in an example,
including the trailing positions right-padding adds within a batch — so on
a length-heterogeneous prompt set, examples from a SHORT bucket and a LONG
bucket contribute position-sums over different numbers of real positions
before they get averaged into the same mean. This is inherited from
``mech_nodes``, not a bug introduced by the bucketing above (bucketing
itself stays exactly batch-size invariant under ``"all"`` too, tested
directly) — it is a reason to prefer the default ``--positions="answer"``
(one position per example, always, on any length) on this repo's
length-heterogeneous task data; the pre-registered metric in the module
docstring above is stated at the answer position for this reason.

**Candidate readouts (documented, NOT pre-registered — decisions.md
2026-08-21 night, Tier 3: "no owner signature pre-registered").** Neither
of these metrics has a calibrated absolute scale, so read them as a
CROSS-ARM comparison — pp (elicit) vs. fmt (teach) vs. base (the
non-pretaught control for "what does training on this LoRA data alone do
from scratch"), all under the SAME ``--ablation``/``--positions``/
``--batch-size`` — not as absolute numbers for one arm alone.

- **elicit-as-readout-unlock** (θ0 = ``evt-ts38pp-parent``): if the sum is
  already computed by θ0's internals and target-training only needs to
  teach the model to EXPOSE that existing computation under a bare-NL
  prompt, the components that do the computing should stay the same
  components before and after — LOW ``delta_node_rel``, HIGH
  ``node_rank_spearman`` (the ranking of "which nodes matter" barely
  reshuffles), and change CONCENTRATED in a few LATE nodes (the ones
  responsible for reading a representation out to the vocabulary) rather
  than spread across depth. Those few late nodes should show a HIGH
  ``rewiring_index`` specifically — their total contribution needn't grow
  much (the representation was already being computed), but WHERE they
  route it changes (new downstream consumer reads it now, an old one
  reads it less) — routing changes, not new mass. Early/mid "compute the
  sum" nodes should show low delta AND low rewiring (their role is
  literally unchanged).
- **teach-as-new-construction** (θ0 = ``evt-ts38mt-fmt-parent``): if there
  is no latent computation to redirect, target-training must build fresh
  machinery — HIGH ``delta_node_rel`` (previously-unimportant nodes become
  important, not just a reshuffle among already-important ones), LOW
  ``node_rank_spearman`` (importance ranking is substantially reshuffled,
  not just perturbed at the edges of a fixed core set), change SPREAD
  across MANY layers (``delta_l1`` roughly comparable layer-to-layer
  rather than concentrated near the top), and LOWER ``rewiring_index`` on
  the nodes that DO change (their total contribution genuinely grows —
  new mass, not redistributed mass — since the computation itself is
  being built, not just re-routed). ``node_sign_flip_frac`` may also read
  higher under this story: a node repurposed for a new computation can
  flip which direction it net-pushes the metric, where a node merely
  read out differently keeps its own sign.

Both are stated as hypotheses to check against the pp/fmt/base runs, not
results.

Usage:
    python3 node_edge_delta.py --model-a dir:$GEODE_STORE/runs/evt-ts38mt-fmt-parent/model \\
        --model-b run:evt-ts38mt-fmt-n21544 \\
        --prompt-parquet ../data/full/D_algo_eval_bare.parquet \\
        --out node_edge_delta_fmt.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import Tensor, nn

from mech_lib import load_any_model, load_task_examples, pad_examples, write_table
from mech_nodes import (
    NodeId,
    attribution_scores,
    capture_nodes,
    edge_attribution_scores,
)

from geode.arith.spans import SftExample

DEFAULT_TOKENIZER = Path(__file__).resolve().parent.parent / "tokenizer"
_EPS = 1e-30

EdgeKey = tuple[NodeId, "NodeId | str"]


# --------------------------------------------------------------------------
# Pure metric functions
# --------------------------------------------------------------------------


def l1_delta(s0: Tensor, sT: Tensor) -> float:
    """Σ|sT − s0|, fp64. ``s0``/``sT`` are any two aligned 1-D score
    vectors (used for both node and edge vectors — this function has no
    node/edge-specific logic)."""
    if s0.shape != sT.shape:
        raise ValueError(f"l1_delta: shape mismatch {tuple(s0.shape)} vs {tuple(sT.shape)}")
    return float((sT.to(torch.float64) - s0.to(torch.float64)).abs().sum())


def rel_delta(s0: Tensor, sT: Tensor) -> float:
    """``l1_delta(s0, sT) / (Σ(|s0|+|sT|)/2)`` — normalises the L1 delta by
    the vectors' own average L1 mass, so it stays comparable across score
    vectors of different overall magnitude. Range ``[0, 2]`` (NOT ``[0,
    1]``: equality at 2 is a same-magnitude, opposite-sign flip everywhere
    — Σ|sT−s0| ≤ Σ(|s0|+|sT|) = 2·denominator by the triangle inequality).
    Denominator clamped at ``_EPS``: returns 0.0 (not NaN) when both
    vectors are ~zero everywhere — the numerator is then 0 too, so this is
    the identical-models degenerate case, not a division-by-zero error.
    """
    numer = l1_delta(s0, sT)
    denom = float(((s0.to(torch.float64).abs() + sT.to(torch.float64).abs()) / 2).sum())
    return numer / max(denom, _EPS)


def _average_rank(x: Tensor) -> Tensor:
    """Ranks (1-based, ties get the average of their tied positions'
    ranks — ``scipy.stats.rankdata``'s ``method="average"``), fp64."""
    n = x.shape[0]
    sorted_vals, sorted_idx = torch.sort(x)
    ranks = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0  # 1-based inclusive range [i+1, j+1]
        ranks[sorted_idx[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman(s0: Tensor, sT: Tensor) -> float:
    """Spearman rank correlation between ``s0`` and ``sT`` (ties by average
    rank, computed locally — NOT imported from ``circuit_jaccard.py``'s own
    spearman, to keep this Tier-3 driver independent of that one; same
    math, duplicated deliberately). ``n<=1`` or both vectors ranking as
    constant returns 1.0 (trivially "in the same order" — nothing to
    disagree about); exactly one vector constant while the other varies
    returns NaN (undefined correlation, not silently 0 or 1).
    """
    if s0.shape != sT.shape:
        raise ValueError(f"spearman: shape mismatch {tuple(s0.shape)} vs {tuple(sT.shape)}")
    n = s0.shape[0]
    if n == 0:
        raise ValueError("spearman: empty vectors")
    if n == 1:
        return 1.0
    r0 = _average_rank(s0.to(torch.float64))
    rT = _average_rank(sT.to(torch.float64))
    r0c = r0 - r0.mean()
    rTc = rT - rT.mean()
    n0 = float(r0c.norm())
    nT = float(rTc.norm())
    if n0 < _EPS and nT < _EPS:
        return 1.0
    if n0 < _EPS or nT < _EPS:
        return math.nan
    return float((r0c @ rTc) / (n0 * nT))


def sign_flip_frac(s0: Tensor, sT: Tensor) -> float:
    """Fraction of entries, weighted by ``|sT|``, whose sign flipped
    between ``s0`` and ``sT``: ``sign(s0)·sign(sT) < 0`` (strictly opposite
    NONZERO signs — an entry moving to/from exactly zero is not counted as
    a flip, since "which sign it flipped to" is undefined there). Weight
    denominator clamped at ``_EPS``: returns 0.0 when ``sT`` is ~zero
    everywhere (no weight to decide "which" entries flipped)."""
    if s0.shape != sT.shape:
        raise ValueError(f"sign_flip_frac: shape mismatch {tuple(s0.shape)} vs {tuple(sT.shape)}")
    s0d = s0.to(torch.float64)
    sTd = sT.to(torch.float64)
    flipped = (torch.sign(s0d) * torch.sign(sTd)) < 0
    weights = sTd.abs()
    denom = float(weights.sum())
    if denom < _EPS:
        return 0.0
    return float(weights[flipped].sum() / denom)


def rewiring_index(delta_node_abs: Tensor, edge_delta_l1: Tensor) -> Tensor:
    """Elementwise ``1 − delta_node_abs / edge_delta_l1``, both same-shape
    tensors (one entry per node): 0 = pure re-weighting (all of a node's
    edges moved in the same net direction), 1 = pure rewiring (edges
    redistributed with zero net effect on the node's own total score). NaN
    wherever ``edge_delta_l1 < _EPS`` (the node's edges barely moved at
    all — the ratio is undefined, not "0 rewiring"). By the EAP identity
    (module docstring), ``delta_node_abs <= edge_delta_l1`` up to
    floating-point slack, so the index is confined to ``[0, 1]``."""
    if delta_node_abs.shape != edge_delta_l1.shape:
        raise ValueError(
            f"rewiring_index: shape mismatch {tuple(delta_node_abs.shape)} vs "
            f"{tuple(edge_delta_l1.shape)}"
        )
    d = delta_node_abs.to(torch.float64)
    e = edge_delta_l1.to(torch.float64)
    idx = 1.0 - d / e.clamp_min(_EPS)
    return torch.where(e < _EPS, torch.full_like(idx, math.nan), idx)


def rewiring_index_mean(idx: Tensor, weights: Tensor) -> float:
    """``weights``-mass-weighted mean of a ``rewiring_index`` vector,
    skipping entries with ``weights < _EPS`` (their index is NaN by
    construction — including them would poison the mean with ``0·NaN``
    instead of correctly contributing zero information). NaN if every
    weight is ~0 (nothing to average)."""
    if idx.shape != weights.shape:
        raise ValueError(
            f"rewiring_index_mean: shape mismatch {tuple(idx.shape)} vs {tuple(weights.shape)}"
        )
    w = weights.to(torch.float64)
    valid = w > _EPS
    if not bool(valid.any()):
        return math.nan
    wv = w[valid]
    iv = idx.to(torch.float64)[valid]
    return float((iv * wv).sum() / wv.sum())


def aggregate_by_layer(
    layers: Sequence[int],
    s0: Tensor,
    sT: Tensor,
    edge_delta_l1: Tensor,
    rewiring_idx: Tensor,
) -> list[dict]:
    """Roll up per-node ``(s0, sT, edge_delta_l1, rewiring_idx)`` vectors
    (all aligned to ``layers``, one entry per node) into one row per
    distinct layer value: ``s0_sum``/``sT_sum`` (signed sums — CAN
    partially cancel across nodes, unlike the deltas below), ``delta_l1``
    = Σ_u|sT(u)−s0(u)| within the layer (does NOT cancel — the layer's
    total re-weighting mass), ``edge_delta_l1_sum`` = Σ_u
    edge_change_within_node(u) within the layer, ``rewiring_index_mean``
    (``rewiring_index_mean`` over just this layer's own nodes)."""
    if not (
        len(layers) == s0.shape[0] == sT.shape[0] == edge_delta_l1.shape[0] == rewiring_idx.shape[0]
    ):
        raise ValueError(
            "aggregate_by_layer: layers/s0/sT/edge_delta_l1/rewiring_idx length mismatch"
        )
    layers_t = torch.as_tensor(list(layers), dtype=torch.long)
    rows: list[dict] = []
    for layer in sorted(set(int(x) for x in layers)):
        mask = layers_t == layer
        s0_l, sT_l = s0[mask], sT[mask]
        edge_l1_l = edge_delta_l1[mask]
        rw_l = rewiring_idx[mask]
        rows.append(
            {
                "level": "layer",
                "layer": layer,
                "s0_sum": float(s0_l.sum()),
                "sT_sum": float(sT_l.sum()),
                "delta_l1": float((sT_l - s0_l).abs().sum()),
                "edge_delta_l1_sum": float(edge_l1_l.sum()),
                "rewiring_index_mean": rewiring_index_mean(rw_l, edge_l1_l),
                "n_nodes": int(mask.sum()),
            }
        )
    return rows


def edge_l1_and_signed_by_node(
    node_keys: Sequence[NodeId],
    edge_keys: Sequence[EdgeKey],
    s0_edge: Tensor,
    sT_edge: Tensor,
) -> tuple[Tensor, Tensor]:
    """Group per-edge deltas by their writer node ``u``: returns
    ``(edge_delta_l1_per_node, edge_delta_signed_sum_per_node)``, both
    aligned to ``node_keys``. The FIRST is ``rewiring_index``'s
    denominator. The SECOND is the quantity the module docstring's EAP
    identity equates to ``sT_node(u) − s0_node(u)`` exactly — surfaced so
    that identity is checkable directly through THIS grouping code (not
    just through ``mech_nodes``'s own, already-tested, per-model
    identity), and reported as a column so a reader can eyeball it holding
    on real data too."""
    if len(edge_keys) != s0_edge.shape[0] or len(edge_keys) != sT_edge.shape[0]:
        raise ValueError("edge_l1_and_signed_by_node: edge_keys/s0_edge/sT_edge length mismatch")
    index = {nid: i for i, nid in enumerate(node_keys)}
    l1 = torch.zeros(len(node_keys), dtype=torch.float64)
    signed = torch.zeros(len(node_keys), dtype=torch.float64)
    s0v = s0_edge.to(torch.float64).tolist()
    sTv = sT_edge.to(torch.float64).tolist()
    for (u, _v), a, b in zip(edge_keys, s0v, sTv):
        if u not in index:
            raise ValueError(f"edge_l1_and_signed_by_node: edge writer {u} not in node_keys")
        i = index[u]
        d = b - a
        l1[i] += abs(d)
        signed[i] += d
    return l1, signed


# --------------------------------------------------------------------------
# Model plumbing
# --------------------------------------------------------------------------


def _edge_sort_key(edge: EdgeKey) -> tuple:
    """Total order over edge keys built from PRIMITIVE tuples only — never
    compares a ``NodeId`` to the string ``"out"`` directly (which would
    raise, since ``NodeId``'s ``order=True`` only knows how to compare
    against another ``NodeId``)."""
    u, v = edge
    u_key = (u.layer, u.kind, u.index)
    v_key = (v.layer, v.kind, v.index) if isinstance(v, NodeId) else (2**62, "zzz_out", 0)
    return (*u_key, *v_key)


def _bucket_by_length(examples: Sequence[SftExample]) -> dict[int, list[int]]:
    """Example indices grouped by each example's own ``len(input_ids)`` —
    see the module docstring's "batching over a length-heterogeneous
    prompt set" section for why this, not a single global length, is the
    right unit for the mean-ablation reference."""
    buckets: dict[int, list[int]] = {}
    for i, ex in enumerate(examples):
        buckets.setdefault(len(ex.input_ids), []).append(i)
    return buckets


def _reference_acts_mean(
    model: nn.Module, examples: Sequence[SftExample], *, batch_size: int
) -> dict[NodeId, Tensor]:
    """Per-position mean node activation over ``examples`` (all ONE
    length — callers pass one length-bucket at a time), accumulated in
    ``batch_size`` chunks so memory stays O(batch_size) rather than
    O(len(examples)). The returned ``[1, s, d]`` tensor per node is
    mathematically identical, as a ``reference_acts`` argument to
    ``mech_nodes.attribution_scores``/``edge_attribution_scores``, to
    passing every raw activation (``_ablated_acts``'s own
    ``mean(dim=0, keepdim=True)`` is the identity on an already-averaged
    ``[1,s,d]`` tensor) — purely a memory optimization."""
    device = next(model.parameters()).device
    total: dict[NodeId, Tensor] = {}
    n = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        ids, mask = pad_examples(batch, device=str(device))
        acts = capture_nodes(model, ids, mask)
        for nid, a in acts.items():
            s = a.sum(dim=0)
            total[nid] = total.get(nid, torch.zeros_like(s)) + s
        n += len(batch)
    if n == 0:
        raise ValueError("_reference_acts_mean: empty examples")
    return {nid: (t / n).unsqueeze(0) for nid, t in total.items()}


def attribution_means(
    model: nn.Module,
    examples: Sequence[SftExample],
    *,
    ablation: str = "mean",
    positions: str = "answer",
    batch_size: int = 32,
) -> tuple[dict[NodeId, float], dict[EdgeKey, float]]:
    """``(node_means, edge_means)``: mean ``attribution_scores``/``edge_
    attribution_scores`` over ``examples``, processed in ``batch_size``
    chunks. Does NOT depend on ``batch_size`` (tested directly) — see the
    module docstring's batching section: examples are grouped into
    length-buckets first, each bucket's ``ablation="mean"`` reference is
    computed once over ALL of that bucket's own examples before any
    per-batch attribution call, so splitting the same examples into
    different-sized batches cannot change the reference each one ablates
    toward. ``ablation="zero"`` needs no reference (fixed regardless of
    batching) but goes through the same bucketed-batch loop for a single
    code path.
    """
    if not examples:
        raise ValueError("attribution_means: empty examples")
    buckets = _bucket_by_length(examples)

    reference_by_len: dict[int, dict[NodeId, Tensor]] = {}
    if ablation == "mean":
        for length, idxs in buckets.items():
            reference_by_len[length] = _reference_acts_mean(
                model, [examples[i] for i in idxs], batch_size=batch_size
            )

    node_sum: dict[NodeId, float] = {}
    edge_sum: dict[EdgeKey, float] = {}
    n = 0
    for length, idxs in buckets.items():
        ref = reference_by_len.get(length)
        for start in range(0, len(idxs), batch_size):
            batch = [examples[i] for i in idxs[start : start + batch_size]]
            node_scores = attribution_scores(
                model, batch, ablation=ablation, reference_acts=ref, positions=positions
            )
            edge_scores = edge_attribution_scores(
                model, batch, ablation=ablation, reference_acts=ref, positions=positions
            )
            for nid, v in node_scores.items():
                node_sum[nid] = node_sum.get(nid, 0.0) + float(v.sum())
            for key, v in edge_scores.items():
                edge_sum[key] = edge_sum.get(key, 0.0) + float(v.sum())
            n += len(batch)

    node_means = {
        nid: node_sum[nid] / n for nid in sorted(node_sum, key=lambda k: (k.layer, k.kind, k.index))
    }
    edge_means = {key: edge_sum[key] / n for key in sorted(edge_sum, key=_edge_sort_key)}
    return node_means, edge_means


def node_edge_delta_rows(
    model_a: nn.Module,
    model_b: nn.Module,
    examples: Sequence[SftExample],
    *,
    ablation: str = "mean",
    positions: str = "answer",
    batch_size: int = 32,
) -> list[dict]:
    """Per-node rows (``level="node"``), per-layer aggregates
    (``level="layer"``), and one ``level="summary"`` row — the module
    docstring's full metric set, computed by comparing ``attribution_
    means(model_a, ...)`` against ``attribution_means(model_b, ...)``."""
    if not examples:
        raise ValueError("node_edge_delta_rows: empty examples")
    node_a, edge_a = attribution_means(
        model_a, examples, ablation=ablation, positions=positions, batch_size=batch_size
    )
    node_b, edge_b = attribution_means(
        model_b, examples, ablation=ablation, positions=positions, batch_size=batch_size
    )
    if set(node_a) != set(node_b):
        raise ValueError(
            "node_edge_delta_rows: model_a/model_b produced different node sets "
            "(architecture mismatch?)"
        )
    if set(edge_a) != set(edge_b):
        raise ValueError(
            "node_edge_delta_rows: model_a/model_b produced different edge sets "
            "(architecture mismatch?)"
        )

    node_keys = sorted(node_a, key=lambda nid: (nid.layer, nid.kind, nid.index))
    s0_node = torch.tensor([node_a[k] for k in node_keys], dtype=torch.float64)
    sT_node = torch.tensor([node_b[k] for k in node_keys], dtype=torch.float64)

    edge_keys = sorted(edge_a, key=_edge_sort_key)
    s0_edge = torch.tensor([edge_a[k] for k in edge_keys], dtype=torch.float64)
    sT_edge = torch.tensor([edge_b[k] for k in edge_keys], dtype=torch.float64)

    delta_node_l1 = l1_delta(s0_node, sT_node)
    delta_node_rel = rel_delta(s0_node, sT_node)
    node_rank_spearman = spearman(s0_node, sT_node)
    node_flip_frac = sign_flip_frac(s0_node, sT_node)

    delta_edge_l1 = l1_delta(s0_edge, sT_edge)
    delta_edge_rel = rel_delta(s0_edge, sT_edge)
    edge_rank_spearman = spearman(s0_edge, sT_edge)

    edge_delta_l1_per_node, edge_delta_signed_per_node = edge_l1_and_signed_by_node(
        node_keys, edge_keys, s0_edge, sT_edge
    )
    delta_node_signed = sT_node - s0_node
    delta_node_abs = delta_node_signed.abs()
    rewiring = rewiring_index(delta_node_abs, edge_delta_l1_per_node)
    rw_mean = rewiring_index_mean(rewiring, edge_delta_l1_per_node)

    n = len(examples)
    rows: list[dict] = []
    for i, nid in enumerate(node_keys):
        rows.append(
            {
                "level": "node",
                "u": str(nid),
                "layer": nid.layer,
                "kind": nid.kind,
                "index": nid.index,
                "s0": float(s0_node[i]),
                "sT": float(sT_node[i]),
                "delta": float(delta_node_signed[i]),
                "edge_delta_l1": float(edge_delta_l1_per_node[i]),
                "edge_delta_signed_sum": float(edge_delta_signed_per_node[i]),
                "rewiring_index": float(rewiring[i]),
                "n": n,
            }
        )

    layers = [nid.layer for nid in node_keys]
    for row in aggregate_by_layer(layers, s0_node, sT_node, edge_delta_l1_per_node, rewiring):
        row["n"] = n
        rows.append(row)

    rows.append(
        {
            "level": "summary",
            "layer": -1,
            "delta_node_l1": delta_node_l1,
            "delta_node_rel": delta_node_rel,
            "delta_edge_l1": delta_edge_l1,
            "delta_edge_rel": delta_edge_rel,
            "node_rank_spearman": node_rank_spearman,
            "edge_rank_spearman": edge_rank_spearman,
            "node_sign_flip_frac": node_flip_frac,
            "rewiring_index_mean": rw_mean,
            "ablation": ablation,
            "positions": positions,
            "n_nodes": len(node_keys),
            "n_edges": len(edge_keys),
            "n": n,
        }
    )
    return rows


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    summary = df[df["level"] == "summary"].iloc[0]
    print(
        f"[evt] delta_node_rel={summary['delta_node_rel']:.4f} "
        f"delta_edge_rel={summary['delta_edge_rel']:.4f} "
        f"node_rank_spearman={summary['node_rank_spearman']:.4f} "
        f"edge_rank_spearman={summary['edge_rank_spearman']:.4f} "
        f"node_sign_flip_frac={summary['node_sign_flip_frac']:.4f} "
        f"rewiring_index_mean={summary['rewiring_index_mean']:.4f}"
    )
    nodes = df[df["level"] == "node"].copy()
    if not nodes.empty:
        nodes["abs_delta"] = nodes["delta"].abs()
        top = nodes.sort_values("abs_delta", ascending=False).head(5)
        print("[evt] top-5 nodes by |delta|:")
        print(top[["u", "layer", "delta", "rewiring_index"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-a", required=True, help="theta0: run:<run_id> or dir:<path>")
    ap.add_argument("--model-b", required=True, help="theta_T: run:<run_id> or dir:<path>")
    ap.add_argument("--prompt-parquet", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--store", type=Path, default=None, help="override $GEODE_STORE for run: specs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N prompt rows")
    ap.add_argument("--ablation", default="mean", choices=["mean", "zero"])
    ap.add_argument("--positions", default="answer", choices=["answer", "all"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "node_edge_delta.csv"
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model_a = load_any_model(args.model_a, device=args.device, store=args.store)
    model_b = load_any_model(args.model_b, device=args.device, store=args.store)
    df = pd.read_parquet(args.prompt_parquet)
    examples = load_task_examples(df, tokenizer, limit=args.limit)

    rows = node_edge_delta_rows(
        model_a,
        model_b,
        examples,
        ablation=args.ablation,
        positions=args.positions,
        batch_size=args.batch_size,
    )
    write_table(pd.DataFrame(rows), args.out)
    print(f"[evt] wrote {args.out} ({len(rows)} rows)")
    print_summary(rows)


if __name__ == "__main__":
    main()
