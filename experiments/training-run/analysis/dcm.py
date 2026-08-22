"""DCM — Desiderata-based Component Masking (Davies et al. 2023), adapted
cross-model (Phase-0/Tier-3 test 5, gated on Tier 1 finding a latent sum —
decisions.md 2026-08-21 night "ts38mt pre-registration"): what is the
SMALLEST set of θ0 (``--model-a``) components whose activations, if
replaced by θ_T's (``--model-b``) own, make θ0 predict the correct FIRST
answer token — the desideratum "θ0 now behaves like θ_T at the answer",
adapted from Davies et al.'s single-model circuit-discovery objective to a
cross-model recovery objective?

**Nodes.** Every head/mlp node of ``mech_nodes.py`` (block-index NodeId
convention documented there, NOT ``mech_lib``'s embed/residual layer
numbering — this module never touches the residual-hook points directly).

**The relaxation.** A learnable scalar per node, ``logit_u``, with
``m_u = sigmoid(logit_u) in (0,1)``. For every node, θ0's forward pass
replaces its clean activation with the convex combination ``m_u * a_T[u] +
(1 - m_u) * a_clean`` (``a_T`` = θ_T's own captured activation on the SAME
prompts, ``a_clean`` = θ0's own clean activation at that node — captured
implicitly, live, during the patched forward). ``m_u -> 1`` means "node u is
selected: use θ_T's value there"; ``m_u -> 0`` means "leave θ0 alone here".
Loss, summed over the full batch (no minibatching):

    L(logit) = -mean_i log p_theta0[patched](target_i) + lambda * sum_u m_u

the first term rewards the patched forward getting the answer right, the
second (L1 on the CONTINUOUS mask value) pushes as many ``m_u`` toward 0 as
the first term tolerates — the "desiderata satisfaction at minimum
component cost" objective, continuous-relaxed for gradient descent (Adam).
Only ``logit`` trains; every model parameter of both θ0 and θ_T is frozen
for the whole fit (``fit_mask`` saves + restores θ0's own ``requires_grad``
flags even on error — θ_T's forward never enters a grad-enabled graph at
all in this module, so it needs no such save/restore to keep the same
guarantee).

**Patching under autograd.** Both the relaxed fit (``fit_mask``) and the
discrete evaluation (``evaluate_mask``) go through ``mech_nodes.patch_nodes``.
Its per-head hook rebuilds the o_proj input FUNCTIONALLY (per-head pieces
read from the never-mutated input, one ``torch.cat``) precisely so that a
callable whose output depends on the VALUE of its clean input — the
``(1-m)*a_clean`` term here — backprops cleanly. The first draft of this
driver (2026-08-21) found the original in-place implementation raised
``RuntimeError: ... modified by an inplace operation`` at ``.backward()``
for exactly that case; the fix landed in ``mech_nodes.py`` the same day,
pinned by ``test_mech_nodes.py::TestPatchNodes::
test_value_dependent_callable_backprops_to_mask_param``, and
``test_dcm.py`` additionally checks the analytic mask gradient against a
finite difference on a live model.

**Evaluation (the actual readout).** After ``--steps`` of Adam, the
CONTINUOUS mask is binarised at ``--threshold`` (default 0.5, ``m_u >=
threshold`` selected) and re-evaluated with a REAL forward pass patching
EXACTLY the selected nodes' θ_T tensors into θ0 (``evaluate_mask``) —
``masked_logprob``/``masked_top1_acc``, plus clean θ0/θ_T references
(computed once, shared across every λ) and:

    recovery_frac = (masked - clean_a) / (clean_b - clean_a)

(NaN when ``|clean_b - clean_a| < 1e-6`` — same guard, same reading,
as ``cross_patch.py``'s own ``recovery_frac``, reimplemented locally here
per this module's build brief: only ``mech_lib``/``mech_nodes`` are shared
across the ts38mt drivers, each driver otherwise owns its small pure
helpers). ``n_selected``/``frac_selected``/``selected_nodes`` (a
comma-joined node-id string)/``per_layer_counts`` (``"0:2,1:0,2:1"``, every
layer present including zero) summarize the discrete mask itself.

**The relaxation gap.** The CONTINUOUS training loss (soft interpolation at
every node, every step) does not exactly predict the BINARIZED mask's true
recovery — a well-known property of any L0/L1-relaxed subset-selection
objective (Davies et al.'s own DCM formulation has the same gap; this
module's ``evaluate_mask`` step exists precisely because the relaxed loss
alone cannot be trusted as the final readout). This driver is also a
GREEDY first pass, not a reproduction of the paper's exact desiderata
formulation: one joint sigmoid relaxation trained end-to-end with a fixed
L1 weight and no annealing/warm-up schedule, no projected-gradient or
hard-concrete gate, and masking whole (head, mlp) NODES rather than
individual weights/edges.

**λ-sweep (``--lambdas``, the real readout).** One "fit" row per λ — the
whole point is the size/recovery trade-off CURVE, not any single λ's
verdict. No owner signature is pre-registered for this test (decisions.md,
Tier 3 gated tests). Candidate readouts, stated as hypotheses to check, not
results: **elicit** (θ0 = ``evt-ts38pp-parent``) — a SMALL mask (few
components) already gets HIGH recovery (the rest of θ0's own computation
already knows what to do with the right representation once it's there,
consistent with Tier 1's "latent capability" reading if it holds) — the
size/recovery curve should bend sharply, reaching most of its plateau at a
small ``n_selected``. **teach** (θ0 = ``evt-ts38mt-fmt-parent``) — recovery
climbs only gradually with mask size (a LARGE/diffuse mask, many nodes
needed, no small subset carries most of the gain) — the curve stays closer
to linear across ``n_selected``, or never reaches a comparable plateau at
the same total mask budget.

**Row levels.** ``level="fit"``: one row per λ (steps/lr/seed/init_logit/
threshold/positions in every row for audit — this module never mutates its
own config mid-sweep). ``level="node"``: one row per (λ, node) — final
CONTINUOUS mask value and its ``selected`` bool.

Determinism: ``torch.manual_seed(seed)`` at the top of every fit; a single
full-batch (no minibatching/shuffling) over the SAME padded batch reused
for every λ in a sweep; fp32 model forward, fp64 metric/loss reduction
(``mech_nodes.answer_logprob``'s own convention). ``--limit`` caps the
example count via ``mech_lib.load_task_examples``.

Usage:
    python3 dcm.py --model-a dir:$GEODE_STORE/runs/evt-ts38pp-parent/model \\
        --model-b run:evt-ts38mt-pp-n21544 \\
        --prompt-parquet ../data/full/D_algo_eval_bare.parquet \\
        --lambdas 0.001,0.01,0.1 --steps 200 --out dcm_ts38mt_pp_n21544.csv
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import torch
from torch import Tensor, nn

from mech_lib import (
    load_any_model,
    load_task_examples,
    pad_examples,
    residual_modules,
    write_table,
)
from mech_nodes import (
    NodeId,
    _num_heads_and_dhead,
    answer_logprob,
    answer_targets,
    capture_nodes,
    patch_nodes,
)

from geode.arith.spans import SftExample

DEFAULT_TOKENIZER = Path(__file__).resolve().parent.parent / "tokenizer"
_DENOM_EPS = 1e-6


# --------------------------------------------------------------------------
# Pure metric helpers
# --------------------------------------------------------------------------


def recovery_frac(metric_masked: float, metric_clean_a: float, metric_clean_b: float) -> float:
    """``(masked - clean_a) / (clean_b - clean_a)`` -- 0.0 means "masked θ0
    behaves like its own clean self" (the selected nodes carried nothing
    useful), 1.0 means "masked θ0 fully reproduces θ_T's own clean answer".
    NaN when ``|clean_b - clean_a| < _DENOM_EPS`` (the two models give
    indistinguishable clean metrics -- division by ~zero)."""
    denom = metric_clean_b - metric_clean_a
    if abs(denom) < _DENOM_EPS:
        return math.nan
    return (metric_masked - metric_clean_a) / denom


def _forward_metrics(
    model: nn.Module, input_ids: Tensor, attention_mask: Tensor, positions: Tensor, targets: Tensor
) -> tuple[Tensor, Tensor]:
    """``(logprob[b] fp64 nats, top1_correct[b] bool)`` from ONE no-grad
    forward pass -- the same fp64 ``log_softmax`` convention
    ``cross_patch.py``/``logit_lens.py`` use, kept local (this module's
    build brief: each Tier-2/3 driver owns its own small pure helpers;
    only ``mech_lib``/``mech_nodes`` are shared)."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask.long()).logits
    b = input_ids.shape[0]
    idx = torch.arange(b, device=logits.device)
    sel = logits[idx, positions.to(logits.device), :]
    logp = torch.log_softmax(sel.to(torch.float64), dim=-1)
    target_logp = logp.gather(1, targets.to(device=logits.device)[:, None]).squeeze(1)
    top1 = sel.argmax(dim=-1) == targets.to(sel.device)
    return target_logp, top1


# --------------------------------------------------------------------------
# Batch preparation (shared across every lambda in a sweep)
# --------------------------------------------------------------------------


@dataclass
class DcmBatch:
    """The frozen inputs + captures every λ in a sweep reuses: one padded
    batch, θ_T's per-node activations (``a_T``), and both models' CLEAN
    (unpatched) reference metrics -- computed ONCE so ``recovery_frac``'s
    denominator and every mask's ``a_T`` are identical across the whole
    sweep."""

    input_ids: Tensor
    attention_mask: Tensor
    answer_pos: Tensor
    targets: Tensor
    nodes: list[NodeId]
    a_T: dict[NodeId, Tensor]
    clean_logprob_a: float
    clean_top1_a: float
    clean_logprob_b: float
    clean_top1_b: float
    n: int


def _assert_matching_architecture(model_a: nn.Module, model_b: nn.Module) -> None:
    """DCM patches θ_T's OWN node tensors directly into θ0's forward pass,
    so every node id must resolve to the SAME shape on both models -- a
    mismatched ``n_heads``/``d_head`` would otherwise silently broadcast
    (or crash deep inside a matmul with a confusing message) instead of
    raising here, at the one place that knows what went wrong. A
    layer-count mismatch is already caught loudly by ``patch_nodes``
    itself (``{nid} names layer {nid.layer}, model
    has {n} blocks``), so this only adds the head-geometry check that has
    no other guard."""
    ha, da = _num_heads_and_dhead(model_a)
    hb, db = _num_heads_and_dhead(model_b)
    if (ha, da) != (hb, db):
        raise ValueError(
            f"dcm: model_a head geometry (H={ha}, d_head={da}) != model_b's "
            f"(H={hb}, d_head={db}) -- DCM requires matched architectures for node-level patching"
        )
    n_layers_a = len(residual_modules(model_a)[1])
    n_layers_b = len(residual_modules(model_b)[1])
    if n_layers_a != n_layers_b:
        raise ValueError(
            f"dcm: model_a has {n_layers_a} blocks, model_b has {n_layers_b} blocks -- "
            "DCM requires matched architectures"
        )


def prepare_batch(
    model_a: nn.Module, model_b: nn.Module, examples: Sequence[SftExample]
) -> DcmBatch:
    """Pad ``examples`` once, capture θ_T's node activations, and record
    both models' clean (unpatched) answer metrics -- everything a λ-sweep
    needs that does NOT depend on λ itself."""
    if not examples:
        raise ValueError("prepare_batch: empty examples")
    _assert_matching_architecture(model_a, model_b)
    device = next(model_a.parameters()).device
    input_ids, attention_mask = pad_examples(examples, device=str(device))
    answer_pos, targets = answer_targets(examples)
    answer_pos, targets = answer_pos.to(device), targets.to(device)

    clean_logp_a, clean_top1_a = _forward_metrics(
        model_a, input_ids, attention_mask, answer_pos, targets
    )
    clean_logp_b, clean_top1_b = _forward_metrics(
        model_b, input_ids, attention_mask, answer_pos, targets
    )
    a_T = capture_nodes(model_b, input_ids, attention_mask)
    nodes = sorted(a_T.keys())

    return DcmBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        answer_pos=answer_pos,
        targets=targets,
        nodes=nodes,
        a_T=a_T,
        clean_logprob_a=float(clean_logp_a.mean()),
        clean_top1_a=float(clean_top1_a.to(torch.float64).mean()),
        clean_logprob_b=float(clean_logp_b.mean()),
        clean_top1_b=float(clean_top1_b.to(torch.float64).mean()),
        n=len(examples),
    )


def _make_relaxed_repl(logits_param: Tensor, i: int, a_t: Tensor) -> Callable[[Tensor], Tensor]:
    """``a_clean -> m*a_T + (1-m)*a_clean``, ``m = sigmoid(logits_param[i])``
    -- the continuous relaxation each node's replacement callable computes,
    closing over its OWN slice of the shared mask-logit tensor (index
    ``i``) and θ_T's captured activation ``a_t`` (``patch_nodes``'s
    callables receive only the clean activation -- θ_T's value is captured
    by closure)."""

    def repl(a_clean: Tensor) -> Tensor:
        m = torch.sigmoid(logits_param[i])
        return m * a_t + (1.0 - m) * a_clean

    return repl


# --------------------------------------------------------------------------
# Fitting the mask
# --------------------------------------------------------------------------


@dataclass
class MaskFit:
    """``mask_values``: final ``sigmoid(logit_u)`` per node (continuous,
    NOT yet binarised). ``loss_initial``/``loss_final``: the relaxed
    objective at step 0 and at the LAST iteration (logged BEFORE that
    iteration's own parameter update -- the standard per-step
    training-loop logging convention, so ``loss_final`` reflects the
    second-to-last mask state, not the just-updated one)."""

    mask_values: dict[NodeId, float]
    loss_initial: float
    loss_final: float


def fit_mask(
    model_a: nn.Module,
    batch: DcmBatch,
    *,
    lam: float,
    steps: int,
    lr: float,
    seed: int,
    init_logit: float,
    positions: str,
) -> MaskFit:
    """Adam-optimise one mask-logit scalar per node in ``batch.nodes``
    against the relaxed DCM objective (module docstring), full-batch over
    ``batch``'s examples, ``steps`` iterations. Only ``model_a``'s
    parameters are frozen/restored here (``model_b`` never enters a
    grad-enabled forward anywhere in this module -- its activations are
    already captured, no-grad, in ``batch.a_T``)."""
    if positions not in ("all", "answer"):
        raise ValueError(f"fit_mask: positions must be 'all' or 'answer', got {positions!r}")
    if steps < 1:
        raise ValueError(f"fit_mask: steps must be >= 1, got {steps}")
    torch.manual_seed(seed)
    device = batch.input_ids.device
    pos = batch.answer_pos if positions == "answer" else None

    logits_param = torch.full(
        (len(batch.nodes),),
        float(init_logit),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([logits_param], lr=lr)
    replacements = {
        nid: _make_relaxed_repl(logits_param, i, batch.a_T[nid])
        for i, nid in enumerate(batch.nodes)
    }

    saved_requires_grad = [(p, p.requires_grad) for p in model_a.parameters()]
    for p, _ in saved_requires_grad:
        p.requires_grad_(False)

    loss_initial = math.nan
    loss_final = math.nan
    try:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with patch_nodes(model_a, replacements, pos), torch.enable_grad():
                metric = answer_logprob(
                    model_a, batch.input_ids, batch.attention_mask, batch.answer_pos, batch.targets
                )
            mask_vals = torch.sigmoid(logits_param)
            loss = -metric.mean() + lam * mask_vals.sum().to(torch.float64)
            loss_val = float(loss.item())
            if step == 0:
                loss_initial = loss_val
            loss_final = loss_val
            loss.backward()
            optimizer.step()
    finally:
        for p, rg in saved_requires_grad:
            p.requires_grad_(rg)

    with torch.no_grad():
        mask_values = {
            nid: float(torch.sigmoid(logits_param[i])) for i, nid in enumerate(batch.nodes)
        }
    return MaskFit(mask_values=mask_values, loss_initial=loss_initial, loss_final=loss_final)


# --------------------------------------------------------------------------
# Binarising + evaluating the discrete mask
# --------------------------------------------------------------------------


def binarize_mask(mask_values: dict[NodeId, float], threshold: float) -> dict[NodeId, bool]:
    """``m_u >= threshold`` per node -- the DISCRETE mask ``evaluate_mask``
    actually patches with. ``>=`` (not ``>``) so a mask value landing
    EXACTLY on the threshold counts as selected, a well-defined boundary
    rather than an implementation accident."""
    return {nid: v >= threshold for nid, v in mask_values.items()}


def selected_node_list_str(selected: dict[NodeId, bool]) -> str:
    """Comma-joined ``str(NodeId)`` of every selected node, in ``NodeId``'s
    own sort order (layer-major, heads before that layer's mlp) -- a
    compact summary for the "fit" row (full per-node detail lives in the
    "node" rows)."""
    return ",".join(str(nid) for nid in sorted(nid for nid, sel in selected.items() if sel))


def per_layer_selected_counts(selected: dict[NodeId, bool]) -> dict[int, int]:
    """``{layer: n_selected_nodes_in_that_layer}``, only layers with >= 1
    selected node present -- ``per_layer_counts_str`` fills the zero-count
    layers back in from the caller's own full layer list."""
    counts: dict[int, int] = {}
    for nid, sel in selected.items():
        if sel:
            counts[nid.layer] = counts.get(nid.layer, 0) + 1
    return counts


def per_layer_counts_str(counts: dict[int, int]) -> str:
    """``"0:2,1:0,2:1"`` -- every layer key present, ascending order,
    including zero-count layers (the caller passes a dict already filled
    in for every layer, e.g. ``{l: counts.get(l, 0) for l in all_layers}``)."""
    return ",".join(f"{layer}:{counts[layer]}" for layer in sorted(counts))


def evaluate_mask(
    model_a: nn.Module,
    batch: DcmBatch,
    selected: dict[NodeId, bool],
    *,
    positions: str,
) -> dict:
    """Run θ0 with EXACTLY the selected nodes' clean activations replaced
    by θ_T's own (``batch.a_T``) -- a real full forward pass with the
    BINARIZED (discrete 0/1) mask, not the continuous relaxation
    ``fit_mask`` trains. Uses ``mech_nodes.patch_nodes`` with tensor-valued
    replacements (no gradient needed here). Returns every field
    ``dcm_rows``'s "fit" row needs:
    masked metrics, ``recovery_frac``, node counts, and the selected-node
    summary strings."""
    if positions not in ("all", "answer"):
        raise ValueError(f"evaluate_mask: positions must be 'all' or 'answer', got {positions!r}")
    replacements = {nid: batch.a_T[nid] for nid in batch.nodes if selected[nid]}
    pos_arg = batch.answer_pos if positions == "answer" else None
    with patch_nodes(model_a, replacements, positions=pos_arg):
        masked_logp, masked_top1 = _forward_metrics(
            model_a, batch.input_ids, batch.attention_mask, batch.answer_pos, batch.targets
        )
    masked_logprob = float(masked_logp.mean())
    masked_top1_acc = float(masked_top1.to(torch.float64).mean())
    n_selected = int(sum(selected.values()))
    n_nodes = len(batch.nodes)
    counts = per_layer_selected_counts(selected)
    all_layers = sorted({nid.layer for nid in batch.nodes})
    return {
        "masked_logprob": masked_logprob,
        "masked_top1_acc": masked_top1_acc,
        "recovery_frac": recovery_frac(
            masked_logprob, batch.clean_logprob_a, batch.clean_logprob_b
        ),
        "n_selected": n_selected,
        "n_nodes": n_nodes,
        "frac_selected": n_selected / n_nodes,
        "selected_nodes": selected_node_list_str(selected),
        "per_layer_counts": per_layer_counts_str(
            {layer: counts.get(layer, 0) for layer in all_layers}
        ),
    }


# --------------------------------------------------------------------------
# Lambda sweep -- the real readout
# --------------------------------------------------------------------------


def dcm_rows(
    model_a: nn.Module,
    model_b: nn.Module,
    examples: Sequence[SftExample],
    lambdas: Sequence[float],
    *,
    steps: int,
    lr: float,
    seed: int,
    init_logit: float,
    threshold: float,
    positions: str,
) -> list[dict]:
    """One "fit" row + one "node" row per node, per λ in ``lambdas`` -- the
    λ-sweep IS the readout (module docstring): the size
    (``n_selected``/``frac_selected``) vs. recovery (``recovery_frac``)
    trade-off curve. θ_T's captures and both models' clean references
    (``prepare_batch``) are computed ONCE and reused across the whole
    sweep, so every λ's ``recovery_frac`` shares the same denominator and
    every mask's ``a_T`` is bit-identical."""
    if not examples:
        raise ValueError("dcm_rows: empty examples")
    if not lambdas:
        raise ValueError("dcm_rows: empty lambdas")
    batch = prepare_batch(model_a, model_b, examples)

    rows: list[dict] = []
    for lam in lambdas:
        fit = fit_mask(
            model_a,
            batch,
            lam=lam,
            steps=steps,
            lr=lr,
            seed=seed,
            init_logit=init_logit,
            positions=positions,
        )
        selected = binarize_mask(fit.mask_values, threshold)
        ev = evaluate_mask(model_a, batch, selected, positions=positions)
        rows.append(
            {
                "level": "fit",
                "lambda": lam,
                "steps": steps,
                "lr": lr,
                "seed": seed,
                "init_logit": init_logit,
                "threshold": threshold,
                "positions": positions,
                "n": batch.n,
                "n_nodes": ev["n_nodes"],
                "n_selected": ev["n_selected"],
                "frac_selected": ev["frac_selected"],
                "clean_logprob_a": batch.clean_logprob_a,
                "clean_logprob_b": batch.clean_logprob_b,
                "clean_top1_a": batch.clean_top1_a,
                "clean_top1_b": batch.clean_top1_b,
                "masked_logprob": ev["masked_logprob"],
                "masked_top1_acc": ev["masked_top1_acc"],
                "recovery_frac": ev["recovery_frac"],
                "loss_initial": fit.loss_initial,
                "loss_final": fit.loss_final,
                "selected_nodes": ev["selected_nodes"],
                "per_layer_counts": ev["per_layer_counts"],
            }
        )
        for nid in batch.nodes:
            rows.append(
                {
                    "level": "node",
                    "lambda": lam,
                    "node": str(nid),
                    "layer": nid.layer,
                    "kind": nid.kind,
                    "index": nid.index,
                    "mask_value": fit.mask_values[nid],
                    "selected": selected[nid],
                }
            )
    return rows


def print_summary(rows: list[dict]) -> None:
    fit_rows = sorted((r for r in rows if r["level"] == "fit"), key=lambda r: r["lambda"])
    print("[evt] DCM lambda sweep (size/recovery trade-off):")
    for r in fit_rows:
        print(
            f"[evt]   lambda={r['lambda']:<8g} "
            f"n_selected={r['n_selected']:>3d}/{r['n_nodes']:<3d}  "
            f"frac_selected={r['frac_selected']:.3f}  "
            f"recovery_frac={r['recovery_frac']:.4f}  "
            f"masked_top1={r['masked_top1_acc']:.4f}"
        )
        print(f"[evt]     selected: {r['selected_nodes'] or '(none)'}")


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
    ap.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=0.01,
        help="L1 sparsity weight (ignored if --lambdas is given)",
    )
    ap.add_argument(
        "--lambdas", default=None, help="comma-separated lambda sweep, overrides --lambda"
    )
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1, help="Adam learning rate for the mask logits")
    ap.add_argument("--init-logit", type=float, default=-3.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--positions", choices=("all", "answer"), default="all")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "dcm.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model_a = load_any_model(args.model_a, device=args.device, store=args.store)
    model_b = load_any_model(args.model_b, device=args.device, store=args.store)
    df = pd.read_parquet(args.prompt_parquet)
    examples = load_task_examples(df, tokenizer, limit=args.limit)

    lambdas = [float(x) for x in args.lambdas.split(",")] if args.lambdas else [args.lam]

    rows = dcm_rows(
        model_a,
        model_b,
        examples,
        lambdas,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        init_logit=args.init_logit,
        threshold=args.threshold,
        positions=args.positions,
    )
    write_table(pd.DataFrame(rows), args.out)
    print(f"[evt] wrote {args.out} ({len(rows)} rows)")
    print_summary(rows)


if __name__ == "__main__":
    main()
