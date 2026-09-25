"""Circuit Jaccard (ts38mt Tier-3 test 2, gated on Tier 1 finding a latent
sum): do θ0 and θ_T (or two different arms' θ_T) route the answer through
the SAME small set of nodes/edges, or different ones?

For every model given (``--model name=spec``, repeatable, at least two), on
the SAME prompt set, computes the "circuit" at budget ``k``: the top-``k``
nodes by mean |attribution score| over examples (``mech_nodes.
attribution_scores``, default mean-ablation, ``positions="answer"`` — see
``--ablation``/``--positions``), and likewise the top-``k`` EDGES by mean
|edge score| (``mech_nodes.edge_attribution_scores``). "Node"/"edge" here are
``mech_nodes``'s own conventions verbatim: a node is ``NodeId(layer, "head",
h)`` or ``NodeId(layer, "mlp", 0)``, block index 0-based (mech_nodes' layer
convention, NOT ``mech_lib``'s residual-hook one — see mech_nodes' module
docstring); an edge is ``(u, v)`` with ``u`` a node and ``v`` a node, an
``"attn"``-kind sink, or the string ``"out"``. Both serialise to strings for
every readout below: a node as ``str(NodeId)`` (``"a3.h1"``, ``"m5"``), an
edge as ``edge_key_to_str(u, v)`` (``"a0.h1->m2"``, ``"m3->out"``,
``edge_key_from_str`` inverts it).

For every model: per-budget model rows (``level="model"``) carry the
circuit's ``concentration`` (fraction of the model's total |score| mass held
by its own top-``k``) and the top-``k`` circuit itself, joined
``",".join`` in ranked (``-|score|``, then key) order, e.g. ``"a3.h1,m3,
a0.h2"``.

For every UNORDERED model pair (``itertools.combinations`` of the sorted
model names — ``model_a < model_b`` always): per-budget pair rows
(``level="pair"``) carry ``jaccard`` (``jaccard(top_k(A), top_k(B))``) and
the pair's ``mass_overlap`` both directions — ``mass_overlap_ab`` = the
fraction of A's OWN top-``k`` mass that also lies in B's top-``k``,
``mass_overlap_ba`` the mirror (asymmetric: high `_ab`/low `_ba` means A's
circuit is a near-subset of a much broader B). Every pair row also carries
``spearman`` — Spearman rank correlation of the FULL |score| vectors (every
node/edge in common between the two models, not just the top-``k``) — this
one number is budget-free and repeats across every ``k`` row of the same
(pair, kind), because it doesn't depend on ``k``.

**Node budgets and edge budgets are separate lists** (``--node-budgets``,
default 5,10,20,40; ``--edge-budgets``, default 10,25,50,100) — edges vastly
outnumber nodes at the same depth (``mech_nodes.edge_attribution_scores``'s
combinatorics), so a shared budget scale would either starve the node
circuit or drown the edge one.

**Not pre-registered** (decisions.md 2026-08-21 night, Tier 3: "not
pre-registered" for test 2). Candidate readouts, stated as hypotheses to
check, not results: **elicit** (θ0 = ``evt-ts38pp-parent``, θ_T =
``evt-ts38mt-pp-n<N>``) — θ_T's circuit is close to θ0's own (high
``jaccard``/``spearman`` between ``pp0`` and ``ppT``): training reused
pre-existing machinery rather than building new. **teach** (θ0 = ``evt-
ts38mt-fmt-parent``, θ_T = ``evt-ts38mt-fmt-n<N>``) — low overlap between
``fmt0`` and ``fmtT``: training built/rewired new components, so the circuit
moved. **Cross-arm**: ``ppT`` vs ``fmtT`` overlap speaks to whether the two
arms converge on the same final circuit regardless of how they got there,
or stay on genuinely different solutions.

**Operating rule for ``--ablation mean`` (the default).** ``circuit_scores``
batches over examples for memory only; ``attribution_scores``'s "mean"
ablation reference is always THIS CALL's own batch (no ``reference_acts``
passed), so with ``batch_size < len(examples)`` the ablation baseline is
batch-LOCAL and the reported mean(|score|) is NOT invariant to how the
example set was chunked (verified negatively in the test suite -- it
genuinely differs, not just numerically noisy). To get a reproducible,
chunking-independent circuit under ``--ablation mean``, pass ``--batch-size``
at least as large as the prompt count (one batch = the whole set = a single
global reference) or trim with ``--limit`` to fit memory; otherwise use
``--ablation zero`` (no reference at all, exactly batch-size-invariant).

Usage:
    python3 circuit_jaccard.py \\
        --model pp0=dir:$GEODE_STORE/runs/evt-ts38pp-parent/model \\
        --model ppT=run:evt-ts38mt-pp-n21544 \\
        --model fmt0=dir:$GEODE_STORE/runs/evt-ts38mt-fmt-parent/model \\
        --model fmtT=run:evt-ts38mt-fmt-n21544 \\
        --prompt-parquet ../data/full/D_algo_eval_bare.parquet \\
        --batch-size 512 \\
        --out circuit_jaccard.csv
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch import Tensor, nn

from mech_lib import load_any_model, load_task_examples, write_table
from mech_nodes import NodeId, attribution_scores, edge_attribution_scores

from geode.arith.spans import SftExample

DEFAULT_TOKENIZER = Path(__file__).resolve().parent.parent / "tokenizer"
DEFAULT_NODE_BUDGETS: tuple[int, ...] = (5, 10, 20, 40)
DEFAULT_EDGE_BUDGETS: tuple[int, ...] = (10, 25, 50, 100)
# The "default k" print_summary reports: the 2nd entry of each default list
# (a mid-size, less top-1-noisy budget than the smallest) -- falls back to
# whatever k values are actually present when a caller used custom budgets.
_SUMMARY_K = {"node": 10, "edge": 25}
_EPS = 1e-30

Edge = tuple[NodeId, "NodeId | str"]
_EDGE_STR_RE = re.compile(r"^(.+)->(.+)$")


# --------------------------------------------------------------------------
# Edge key <-> string
# --------------------------------------------------------------------------


def edge_key_to_str(u: NodeId, v: "NodeId | str") -> str:
    """``(u, v) -> "a0.h1->m2"`` / ``"m3->out"``. ``v`` is either a ``NodeId``
    (an "attn"-kind sink or an "mlp"-kind sink) or the literal string
    ``"out"``."""
    v_str = str(v) if isinstance(v, NodeId) else str(v)
    return f"{u}->{v_str}"


def edge_key_from_str(s: str) -> Edge:
    """Inverse of ``edge_key_to_str``. Raises ``ValueError`` on anything not
    matching ``"<u>-><v>"`` where both sides parse via ``NodeId.parse``
    (``v`` may also be the literal ``"out"``)."""
    m = _EDGE_STR_RE.match(s)
    if not m:
        raise ValueError(f"edge_key_from_str: {s!r} does not match '<u>-><v>'")
    u_str, v_str = m.group(1), m.group(2)
    u = NodeId.parse(u_str)
    v: "NodeId | str" = v_str if v_str == "out" else NodeId.parse(v_str)
    return u, v


# --------------------------------------------------------------------------
# Pure functions: top-k, jaccard, spearman, mass_overlap, concentration
# --------------------------------------------------------------------------


def top_k(scores: dict[Any, float], k: int) -> set[Any]:
    """The ``k`` keys with the largest ``|score|``. Deterministic tie-break:
    sorted by ``(-|score|, str(key))``, so equal-magnitude ties resolve in
    ascending string order of the key, not insertion/hash order. ``k``
    larger than ``len(scores)`` is silently clipped to ``len(scores)`` (the
    whole set, no crash); ``k`` must be ``>= 0``.
    """
    if k < 0:
        raise ValueError(f"top_k: k must be >= 0, got {k}")
    ordered = sorted(scores.items(), key=lambda kv: (-abs(kv[1]), str(kv[0])))
    return {key for key, _ in ordered[:k]}


def jaccard(a: set[Any], b: set[Any]) -> float:
    """``|a∩b| / |a∪b|``. ``nan`` when BOTH are empty (undefined, not 0 --
    two empty circuits agree on nothing to compare); ``0.0`` when exactly one
    is empty (a defined, meaningful "no overlap")."""
    if not a and not b:
        return math.nan
    return len(a & b) / len(a | b)


def _rankdata(x: Tensor) -> Tensor:
    """1-based average ranks (ties share the mean rank of their tied block),
    fp64. ``x`` must be 1-D."""
    n = x.shape[0]
    order = torch.argsort(x)
    ranks = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, torch-only (no scipy): Pearson correlation
    of average ranks (``_rankdata``), the standard tie-corrected formula.
    ``nan`` when either input has zero rank-variance (every value tied --
    most commonly, a constant vector) since the correlation is then
    undefined, not 0. Raises ``ValueError`` on empty or mismatched-length
    input.
    """
    xt = torch.as_tensor(list(x), dtype=torch.float64)
    yt = torch.as_tensor(list(y), dtype=torch.float64)
    if xt.shape != yt.shape:
        raise ValueError(f"spearman: length mismatch {xt.shape[0]} vs {yt.shape[0]}")
    if xt.numel() == 0:
        raise ValueError("spearman: empty input")
    rx = _rankdata(xt)
    ry = _rankdata(yt)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.sqrt((rx * rx).sum() * (ry * ry).sum())
    if float(denom) < _EPS:
        return math.nan
    return float((rx * ry).sum() / denom)


def mass_overlap(scores_a: dict[Any, float], top_a: set[Any], top_b: set[Any]) -> float:
    """``Σ_{u∈top_a∩top_b} |scores_a[u]| / Σ_{u∈top_a} |scores_a[u]|`` -- the
    fraction of A's OWN top-k |score| mass that also lies in B's top-k.
    Asymmetric by construction (swap ``scores_a``/``top_a`` with
    ``scores_b``/``top_b`` for the other direction). ``nan`` when
    ``top_a``'s own mass is ~0 (``top_a`` empty, or every score in it is
    exactly 0).
    """
    total = sum(abs(scores_a[key]) for key in top_a)
    if total < _EPS:
        return math.nan
    inter = top_a & top_b
    return sum(abs(scores_a[key]) for key in inter) / total


def concentration(scores: dict[Any, float], top: set[Any]) -> float:
    """Fraction of ``scores``'s total |score| mass held by ``top`` (a
    ``top_k(scores, k)`` result, though any subset works). ``nan`` when the
    total mass is ~0."""
    total = sum(abs(v) for v in scores.values())
    if total < _EPS:
        return math.nan
    return sum(abs(scores[key]) for key in top) / total


# --------------------------------------------------------------------------
# Model plumbing
# --------------------------------------------------------------------------


def circuit_scores(
    model: nn.Module,
    examples: Sequence[SftExample],
    *,
    ablation: str = "mean",
    positions: str = "answer",
    batch_size: int = 32,
) -> tuple[dict[NodeId, float], dict[Edge, float]]:
    """``(node_means, edge_means)``: mean |attribution score| / mean |edge
    score| over every example in ``examples``, batched (``batch_size``) to
    keep memory bounded -- each batch runs one ``attribution_scores`` call
    and one ``edge_attribution_scores`` call (each its own forward+backward,
    ``mech_nodes``'s own per-call cost), and per-example |score| is summed
    across batches then divided by the total example count, which is exactly
    ``mean_i |score_i|`` over the full set regardless of how it was chunked.

    Caveat for ``ablation="mean"`` (the default) -- see the module
    docstring's "Operating rule for --ablation mean" section: ``attribution_
    scores``'s own "mean" ablation reference is THIS CALL's batch
    (``reference_acts`` omitted -- see its docstring), so the ablation
    baseline itself is batch-LOCAL, not a single global mean over every
    example passed here. The mean(|score|) this function reports is
    therefore only batch-size-invariant for ``ablation="zero"`` (no
    reference, no batch-relative baseline); with ``ablation="mean"`` results
    genuinely DIFFER across ``batch_size`` (not just numerically noisy --
    pinned by ``test_circuit_scores_batching_NON_invariance_with_mean_
    ablation``), by design, not by bug: a global mean-ablation reference
    isn't reachable here without re-implementing ``attribution_scores``'
    padding, which this module does not do (mech_nodes.py is read-only for
    these drivers). Pass ``batch_size >= len(examples)`` for a reproducible
    ``"mean"``-ablation circuit.
    """
    if not examples:
        raise ValueError("circuit_scores: empty examples")
    if batch_size <= 0:
        raise ValueError(f"circuit_scores: batch_size must be > 0, got {batch_size}")

    node_sum: dict[NodeId, float] = {}
    edge_sum: dict[Edge, float] = {}
    n = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        node_scores = attribution_scores(model, batch, ablation=ablation, positions=positions)
        edge_scores = edge_attribution_scores(model, batch, ablation=ablation, positions=positions)
        for nid, s in node_scores.items():
            node_sum[nid] = node_sum.get(nid, 0.0) + float(s.abs().sum())
        for key, s in edge_scores.items():
            edge_sum[key] = edge_sum.get(key, 0.0) + float(s.abs().sum())
        n += len(batch)

    node_means = {nid: v / n for nid, v in node_sum.items()}
    edge_means = {key: v / n for key, v in edge_sum.items()}
    return node_means, edge_means


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


def _circuit_string(scores: dict[str, float], top: set[str]) -> str:
    ranked = sorted(top, key=lambda key: (-abs(scores[key]), key))
    return ",".join(ranked)


def _model_rows(
    name: str, scores: dict[str, float], budgets: Sequence[int], kind: str, n: int
) -> list[dict]:
    rows = []
    for k in budgets:
        top = top_k(scores, k)
        rows.append(
            {
                "level": "model",
                "model": name,
                "kind": kind,
                "k": k,
                "k_effective": len(top),
                "concentration": concentration(scores, top),
                "circuit": _circuit_string(scores, top),
                "n": n,
            }
        )
    return rows


def _pair_rows(
    name_a: str,
    name_b: str,
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    budgets: Sequence[int],
    kind: str,
    n: int,
) -> list[dict]:
    common = sorted(set(scores_a) & set(scores_b))
    if not common:
        raise ValueError(
            f"_pair_rows: {name_a!r} and {name_b!r} share no common {kind} keys "
            "(different model architectures?)"
        )
    rho = spearman([abs(scores_a[key]) for key in common], [abs(scores_b[key]) for key in common])

    rows = []
    for k in budgets:
        top_a = top_k(scores_a, k)
        top_b = top_k(scores_b, k)
        rows.append(
            {
                "level": "pair",
                "model_a": name_a,
                "model_b": name_b,
                "kind": kind,
                "k": k,
                "jaccard": jaccard(top_a, top_b),
                "mass_overlap_ab": mass_overlap(scores_a, top_a, top_b),
                "mass_overlap_ba": mass_overlap(scores_b, top_b, top_a),
                "spearman": rho,
                "n": n,
            }
        )
    return rows


def circuit_jaccard_rows(
    models: dict[str, nn.Module],
    examples: Sequence[SftExample],
    *,
    node_budgets: Sequence[int] = DEFAULT_NODE_BUDGETS,
    edge_budgets: Sequence[int] = DEFAULT_EDGE_BUDGETS,
    ablation: str = "mean",
    positions: str = "answer",
    batch_size: int = 32,
) -> list[dict]:
    """One row per (model, kind, k) [``level="model"``] and per (unordered
    pair, kind, k) [``level="pair"``] -- see the module docstring's "For
    every model" / "For every UNORDERED model pair" sections for the exact
    column list. Requires >= 2 models and a non-empty example set.
    """
    if len(models) < 2:
        raise ValueError(f"circuit_jaccard_rows: need >= 2 models, got {len(models)}")
    if not examples:
        raise ValueError("circuit_jaccard_rows: empty examples")

    node_scores: dict[str, dict[str, float]] = {}
    edge_scores: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        nmeans, emeans = circuit_scores(
            model, examples, ablation=ablation, positions=positions, batch_size=batch_size
        )
        node_scores[name] = {str(nid): v for nid, v in nmeans.items()}
        edge_scores[name] = {edge_key_to_str(u, v): val for (u, v), val in emeans.items()}

    n = len(examples)
    names = sorted(models)
    rows: list[dict] = []
    for name in names:
        rows.extend(_model_rows(name, node_scores[name], node_budgets, "node", n))
        rows.extend(_model_rows(name, edge_scores[name], edge_budgets, "edge", n))
    for name_a, name_b in itertools.combinations(names, 2):
        rows.extend(
            _pair_rows(
                name_a, name_b, node_scores[name_a], node_scores[name_b], node_budgets, "node", n
            )
        )
        rows.extend(
            _pair_rows(
                name_a, name_b, edge_scores[name_a], edge_scores[name_b], edge_budgets, "edge", n
            )
        )
    return rows


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    pair_df = df[df["level"] == "pair"]
    for kind in ("node", "edge"):
        sub = pair_df[pair_df["kind"] == kind]
        if sub.empty:
            continue
        avail_ks = sorted(sub["k"].unique())
        k = _SUMMARY_K[kind] if _SUMMARY_K[kind] in avail_ks else avail_ks[len(avail_ks) // 2]
        at_k = sub[sub["k"] == k].sort_values(["model_a", "model_b"])
        print(f"[evt] kind={kind} k={k}")
        print(
            at_k[
                ["model_a", "model_b", "jaccard", "mass_overlap_ab", "mass_overlap_ba", "spearman"]
            ].to_string(index=False)
        )


def _parse_model_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"--model must be 'name=spec' (e.g. 'pp0=dir:/path'), got {spec!r}")
    name, model_spec = spec.split("=", 1)
    return name, model_spec


def _parse_budgets(s: str) -> tuple[int, ...]:
    return tuple(int(part) for part in s.split(","))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        dest="models",
        metavar="name=spec",
        help="name=spec, repeatable, at least twice (spec is run:<run_id> or dir:<path>)",
    )
    ap.add_argument("--prompt-parquet", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--store", type=Path, default=None, help="override $GEODE_STORE for run: specs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N prompt rows")
    ap.add_argument("--ablation", default="mean", choices=("mean", "zero"))
    ap.add_argument("--positions", default="answer", choices=("answer", "all"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--node-budgets", default="5,10,20,40", help="comma-separated top-k budgets for nodes"
    )
    ap.add_argument(
        "--edge-budgets", default="10,25,50,100", help="comma-separated top-k budgets for edges"
    )
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "circuit_jaccard.csv"
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    models: dict[str, nn.Module] = {}
    for spec in args.models:
        name, model_spec = _parse_model_arg(spec)
        models[name] = load_any_model(model_spec, device=args.device, store=args.store)
    if len(models) < 2:
        raise ValueError(f"--model must be given at least twice, got {len(models)}")

    df = pd.read_parquet(args.prompt_parquet)
    examples = load_task_examples(df, tokenizer, limit=args.limit)

    rows = circuit_jaccard_rows(
        models,
        examples,
        node_budgets=_parse_budgets(args.node_budgets),
        edge_budgets=_parse_budgets(args.edge_budgets),
        ablation=args.ablation,
        positions=args.positions,
        batch_size=args.batch_size,
    )
    write_table(pd.DataFrame(rows), args.out)
    print(f"[evt] wrote {args.out} ({len(rows)} rows)")
    print_summary(rows)


if __name__ == "__main__":
    main()
