"""J-lens (Phase 0 test 7, Tier 2): how sensitive is the answer log-prob to a
perturbation of the residual stream, and does that sensitivity explain the
training-induced residual shift?

**Naming assumption (no owner definition on record).** "J-lens" is quoted
verbatim in the ts38mt pre-registration table (decisions.md 2026-08-21
night) with no signature for either arm and no worked-out metric — it is
listed only as a Tier-2 test name. This module reads it as a JACOBIAN LENS:
the gradient of the model's own output wrt each residual-stream point,
by analogy with ``logit_lens.py``'s "project the residual through the head"
but differentiated instead of merely projected. That reading, the metric
definitions below, and the elicit/teach candidate readout are this module's
own construction, not a pre-registered spec.

For each example ``i`` and residual layer ``ℓ`` (``mech_lib``'s convention:
``ℓ=0`` is ``hook_embed``, ``ℓ=i+1`` is ``blocks.{i}.hook_resid_post``):

    J_{i,ℓ} = d/d(h_ℓ[i, p_i - 1])  log p(correct first answer token | prompt_i)

the gradient (a ``[d_model]`` vector) of the FIRST answer token's log-prob
(nats — the same target as ``logit_lens.py``'s ``first_answer``, position
``p_i - 1``, ``position_target_pairs`` reused verbatim from ``logit_lens.py``
so the position convention cannot drift between the two scripts) at the
position that generates it, wrt the residual vector at that layer and
position. Per (model, set, layer) rows:

- ``mean_jac_norm`` = mean_i ‖J_{i,ℓ}‖ — the layer's raw sensitivity.
- ``jac_norm_rel`` = mean_i ‖J_{i,ℓ}‖·‖h_{i,ℓ}‖ — scale-free across layers
  whose own activation norm differs (deep-layer residuals are typically
  much larger than the embedding's): the first-order log-prob change from a
  "double the residual"-sized perturbation IF it were applied exactly along
  ``J_{i,ℓ}``'s own direction. It is an upper bound on an arbitrary
  unit-relative perturbation's effect, not a typical-case estimate — read
  it as a per-layer sensitivity CEILING.
- Direction consistency of ``{J_{i,ℓ}}`` across examples: ``mean_cos_to_mean``
  and ``top_pc_evr``, computed by ``resid_shift.shift_consistency`` UNVERBATIM
  reused on the ``[n, d_model]`` Jacobian matrix (same metric as test 10's,
  same UNCENTERED-top-PC choice and the same "0.0 not NaN" all-zero
  convention — see that module's docstring for why).

**Bridge to test 10 (optional — only when both ``--model-a`` (θ0) and
``--model-b`` (θ_T) are given).** At each layer, the per-example residual
shift ``d_i = h_{T,i} − h_{0,i}`` (``resid_shift``'s own quantity, at the
task position) is compared against θ0's OWN Jacobian ``J^{θ0}_{i,ℓ}``:

- ``mean_cos_shift_vs_jac0`` = mean_i cos(d_i, J^{θ0}_{i,ℓ}).
- ``mean_pred_gain_nats`` = mean_i d_i · J^{θ0}_{i,ℓ} — the first-order
  Taylor prediction of the answer log-prob gain from applying exactly the
  observed shift ``d_i`` at this layer, using only θ0's own local geometry.
- ``actual_gain_nats`` = mean_i (logp_{T,i} − logp_{0,i}) — the REAL gain
  (layer-invariant; repeated on every layer's row to sit next to that
  layer's prediction).
- ``pred_gain_ratio`` = ``mean_pred_gain_nats / actual_gain_nats``.

These four columns are attached ONLY to θ0's rows (they are defined
relative to θ0's Jacobian, not θ_T's) and are absent — read back as NaN
once concatenated — from θ_T's own standard-column rows. Candidate readout
(documented, NOT pre-registered, matching the pre-registration's own Tier-2
caveat): **elicit** → the training update moves the residual mostly along
the direction θ0's readout was already sensitive to (high
``mean_cos_shift_vs_jac0``, ``pred_gain_ratio`` near 1 at some layer) — the
gain is a small nudge along an existing sensitive direction. **teach** →
low cosine, ``pred_gain_ratio`` far from 1 or undefined-in-sign — the gain
is not linearly explainable from θ0's own Jacobian, consistent with the
update creating sensitivity that was not already there.

**Convention exception.** Every OTHER 0.0-not-NaN column in this table
follows the sibling scripts' rule (division guarded by a norm clamp, so an
all-zero numerator reads 0.0). ``pred_gain_ratio`` does NOT: on an
identical-models row both the numerator (``mean_pred_gain_nats``, itself
0.0 for the reason above) and the denominator (``actual_gain_nats``) are
EXACTLY zero — a genuine 0/0, not a near-zero-denominator numerical
pathology — so it is NaN there, the same "NaN where undefined" branch
``weight_diff.module_metrics`` takes for ``rel_fro`` when ``fro_w0 == 0``.

**Memory note.** This module keeps every hooked layer's activations AND
their gradients alive at once (roughly double ``mech_lib.capture_residuals``'s
footprint) for the full batch passed in — ``--limit`` is the memory dial on
a large prompt set, there is no internal chunking.

Usage:
    python3 jacobian_lens.py --model-a run:evt-ts38pp-parent \\
        --prompt-parquet ../data/full/D_algo_eval_bare.parquet --set-name task \\
        --out jacobian_lens_ts38pp_task.csv

    # with the bridge to test 10 (theta0 -> theta_T):
    python3 jacobian_lens.py \\
        --model-a dir:$GEODE_STORE/runs/evt-ts38pp-parent/model \\
        --model-b run:evt-ts38mt-pp-n21544 \\
        --prompt-parquet ../data/full/D_algo_eval_bare.parquet --set-name task \\
        --out jacobian_lens_ts38mt_pp_n21544.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import nn

from logit_lens import FIRST_ANSWER, position_target_pairs
from mech_lib import (
    load_any_model,
    load_task_examples,
    pad_examples,
    residual_modules,
    write_table,
)
from resid_shift import shift_consistency

from geode.arith.spans import SftExample
from geode.probe import residual_hook_names

DEFAULT_TOKENIZER = Path(__file__).resolve().parent.parent / "tokenizer"
_EPS = 1e-30


def freeze_params(model: nn.Module) -> None:
    """Set ``requires_grad=False`` on every parameter of ``model``, in
    place. Pure memory/compute economy, NOT a correctness requirement: every
    hooked activation this module differentiates through is rebuilt as a
    detached leaf (``hook_embed``, via ``capture_residuals_grad``) or a
    ``retain_grad()``'d descendant of that leaf (every later layer), so the
    Jacobian values are identical whether or not the model's own weights
    require grad — freezing just avoids ALSO accumulating an unused
    ``.grad`` on every weight tensor during ``.backward()``.
    """
    for p in model.parameters():
        p.requires_grad_(False)


def capture_residuals_grad(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Grad-enabled twin of ``mech_lib.capture_residuals``: same hook names
    and shapes, but built so a SINGLE ``.backward()`` from any scalar built
    out of the returned ``logits`` populates ``.grad`` on every captured
    tensor.

    ``hook_embed`` is computed directly (``embed(input_ids)``, identical to
    what ``LlamaModel.forward`` would compute internally when given
    ``input_ids``) and turned into a fresh leaf via
    ``.detach().clone().requires_grad_(True)`` — the ONE place this
    module's convention needs a genuine graph root, since every model
    parameter may be frozen (``freeze_params``) and nothing upstream of
    ``hook_embed`` is ever differentiated wrt here. The model is then run
    on ``inputs_embeds=leaf`` (never ``input_ids`` — passing both raises)
    instead of hooking the embedding module and substituting its output:
    fewer moving parts than relying on forward-hook return-value semantics,
    and numerically identical, since Llama applies no scaling or dropout
    between ``embed_tokens`` and the leaf's first use (verified against the
    installed transformers' own ``LlamaModel.forward``).

    Every later ``blocks.{i}.hook_resid_post`` is captured by a forward
    hook that just ``.retain_grad()``s the SAME tensor object used
    downstream (returns ``None``, no substitution) — one connected graph
    rooted at ``hook_embed`` carries gradient back to every captured point
    after one ``.backward()``. Hooks are removed before returning, even on
    error; ``hook_embed`` needs no hook at all (built directly, not via
    ``embed``'s forward), so there is nothing to remove for it.
    """
    embed, blocks = residual_modules(model)
    names = residual_hook_names(len(blocks))
    leaf = embed(input_ids).detach().clone().requires_grad_(True)
    captured: dict[str, torch.Tensor] = {"hook_embed": leaf}
    handles = []

    def make_block_hook(name: str):
        def hook(_module, _inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            t.retain_grad()
            captured[name] = t

        return hook

    for i, block in enumerate(blocks):
        handles.append(block.register_forward_hook(make_block_hook(names[i + 1])))
    try:
        logits = model(inputs_embeds=leaf, attention_mask=attention_mask.long()).logits
    finally:
        for h in handles:
            h.remove()
    missing = [n for n in names if n not in captured]
    if missing:
        raise RuntimeError(f"capture_residuals_grad: hooks never fired for {missing}")
    return {n: captured[n] for n in names}, logits


def jacobian_and_logprob(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ex_idx: torch.Tensor,
    pos: torch.Tensor,
    target: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """One forward + one backward. Returns ``(jac, resid, logp)``:

    - ``jac[name]``: ``[n, d_model]`` float64, ``J_{i,ℓ} = d logp_i / d
      h_ℓ[i, pos_i]``.
    - ``resid[name]``: ``[n, d_model]`` float64, ``h_ℓ[i, pos_i]`` itself
      (detached).
    - ``logp``: ``[n]`` float64 nats, log p(target_i | prompt_i) at
      ``pos_i`` (detached).

    ``pos`` must be non-negative: ``label_span[0] == 0`` for some example
    would silently WRAP a ``pos_i = -1`` to the last sequence position
    (Python/PyTorch negative indexing) rather than crash — the exact class
    of silent corruption this module's testing policy exists to rule out —
    so it is checked and raised on explicitly instead.

    Examples don't interact: a Llama-style stack has no cross-example
    computation anywhere in its forward pass (RMSNorm normalizes per token,
    attention is masked to within one example's own sequence via
    ``attention_mask`` + the causal mask, and every ``nn.Linear``/embedding
    op is pointwise across the batch dimension), so
    ``d(logp_j)/d(h_ℓ[i, ...])`` is STRUCTURALLY zero for ``j != i`` — not
    merely small. Summing ``target_logp`` over the whole batch before ONE
    ``.backward()`` therefore gives, at every hooked tensor's ``[i, pos_i,
    :]`` slot, EXACTLY ``d(logp_i)/d(h_ℓ[i, pos_i])`` with no cross terms —
    cheaper than one backward per example and no less exact. Cross-checked
    against a genuinely per-example backward (a batch of varying-length,
    right-padded examples vs. each run alone) in
    ``test_batching_invariance_matches_single_example_backward``.
    """
    if (pos < 0).any():
        raise ValueError(
            "jacobian_and_logprob: a negative position (label_span[0] == 0 for "
            "some example) would silently wrap to the LAST sequence position; refusing"
        )
    captured, logits = capture_residuals_grad(model, input_ids, attention_mask)
    sel = logits[ex_idx, pos, :].to(
        torch.float64
    )  # [n, vocab] — never cast the full [b,s,v] tensor
    logp = torch.log_softmax(sel, dim=-1)
    target_logp = logp.gather(1, target[:, None]).squeeze(1)  # [n], nats
    target_logp.sum().backward()

    jac: dict[str, torch.Tensor] = {}
    resid: dict[str, torch.Tensor] = {}
    for name, t in captured.items():
        if t.grad is None:
            raise RuntimeError(f"jacobian_and_logprob: no gradient captured at {name!r}")
        jac[name] = t.grad[ex_idx, pos, :].detach().to(torch.float64).clone()
        resid[name] = t[ex_idx, pos, :].detach().to(torch.float64).clone()
    return jac, resid, target_logp.detach()


def jacobian_norm_stats(jac: torch.Tensor, h: torch.Tensor) -> tuple[float, float]:
    """``(mean_jac_norm, jac_norm_rel)`` for one layer's ``[n, d_model]``
    Jacobian ``jac`` and the matching residual vectors ``h`` at the same
    (example, position) — module docstring for the ``jac_norm_rel``
    ceiling-not-average-case caveat."""
    jac64 = jac.to(torch.float64)
    h64 = h.to(torch.float64)
    jac_norms = jac64.norm(dim=1)
    h_norms = h64.norm(dim=1)
    return float(jac_norms.mean().item()), float((jac_norms * h_norms).mean().item())


def bridge_layer_stats(
    d: torch.Tensor, jac0: torch.Tensor, logp0: torch.Tensor, logp_t: torch.Tensor
) -> dict[str, float]:
    """The four bridge-to-test-10 columns for one layer (module docstring
    for the full definitions and the ``pred_gain_ratio`` NaN exception):
    ``d`` = ``h_T - h_0`` and ``jac0`` = ``J^{θ0}`` are both ``[n,
    d_model]``; ``logp0``/``logp_t`` are ``[n]``, the SAME ``target_logp``
    ``jacobian_and_logprob`` returns for θ0 and θ_T respectively."""
    d64 = d.to(torch.float64)
    jac64 = jac0.to(torch.float64)
    d_norms = d64.norm(dim=1).clamp_min(_EPS)
    jac_norms = jac64.norm(dim=1).clamp_min(_EPS)
    cos = (d64 * jac64).sum(dim=1) / (d_norms * jac_norms)
    pred_gain = (d64 * jac64).sum(dim=1)
    mean_pred_gain = float(pred_gain.mean().item())
    actual_gain = float((logp_t.to(torch.float64) - logp0.to(torch.float64)).mean().item())
    ratio = mean_pred_gain / actual_gain if abs(actual_gain) > _EPS else math.nan
    return {
        "mean_cos_shift_vs_jac0": float(cos.mean().item()),
        "mean_pred_gain_nats": mean_pred_gain,
        "actual_gain_nats": actual_gain,
        "pred_gain_ratio": ratio,
    }


def jacobian_lens_rows(
    model_a: nn.Module,
    examples: Sequence[SftExample],
    model_a_label: str,
    set_label: str,
    *,
    model_b: nn.Module | None = None,
    model_b_label: str | None = None,
) -> list[dict]:
    """One row per (model, layer): θ0's (``model_a``) standard columns
    always; θ_T's (``model_b``, optional) own standard columns too, for a
    direct elicit-vs-teach comparison of post-training sensitivity; and,
    only when ``model_b`` is given, the four bridge columns merged onto
    θ0's rows (module docstring)."""
    if not examples:
        raise ValueError("jacobian_lens_rows: empty example list")
    device = next(model_a.parameters()).device
    input_ids, attention_mask = pad_examples(examples, device=str(device))
    ex_idx_l, pos_l, target_l = position_target_pairs(examples, FIRST_ANSWER)
    ex_idx = torch.tensor(ex_idx_l, device=device)
    pos = torch.tensor(pos_l, device=device)
    target = torch.tensor(target_l, device=device)

    freeze_params(model_a)
    jac_a, resid_a, logp_a = jacobian_and_logprob(
        model_a, input_ids, attention_mask, ex_idx, pos, target
    )
    names = list(jac_a.keys())

    rows_a: list[dict] = []
    for layer, name in enumerate(names):
        mean_jn, jn_rel = jacobian_norm_stats(jac_a[name], resid_a[name])
        mean_cos, evr = shift_consistency(jac_a[name])
        rows_a.append(
            {
                "model": model_a_label,
                "set": set_label,
                "layer": layer,
                "n": len(examples),
                "mean_jac_norm": mean_jn,
                "jac_norm_rel": jn_rel,
                "mean_cos_to_mean": mean_cos,
                "top_pc_evr": evr,
            }
        )

    rows_b: list[dict] = []
    if model_b is not None:
        freeze_params(model_b)
        jac_b, resid_b, logp_b = jacobian_and_logprob(
            model_b, input_ids, attention_mask, ex_idx, pos, target
        )
        b_label = model_b_label or "model_b"
        for layer, name in enumerate(names):
            mean_jn, jn_rel = jacobian_norm_stats(jac_b[name], resid_b[name])
            mean_cos, evr = shift_consistency(jac_b[name])
            rows_b.append(
                {
                    "model": b_label,
                    "set": set_label,
                    "layer": layer,
                    "n": len(examples),
                    "mean_jac_norm": mean_jn,
                    "jac_norm_rel": jn_rel,
                    "mean_cos_to_mean": mean_cos,
                    "top_pc_evr": evr,
                }
            )
        for layer, name in enumerate(names):
            d = resid_b[name] - resid_a[name]
            rows_a[layer].update(bridge_layer_stats(d, jac_a[name], logp_a, logp_b))

    return rows_a + rows_b


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    for (model_label, set_label), sub in df.groupby(["model", "set"]):
        sub = sub.sort_values("layer")
        peak = sub.loc[sub["jac_norm_rel"].idxmax()]
        print(
            f"[evt] {model_label} / {set_label}: max jac_norm_rel at layer "
            f"{int(peak['layer'])} ({peak['jac_norm_rel']:.6g})"
        )
        if "mean_cos_shift_vs_jac0" in sub.columns and sub["mean_cos_shift_vs_jac0"].notna().any():
            align = sub[
                [
                    "layer",
                    "mean_cos_shift_vs_jac0",
                    "mean_pred_gain_nats",
                    "actual_gain_nats",
                    "pred_gain_ratio",
                ]
            ].dropna(subset=["mean_cos_shift_vs_jac0"])
            print(
                f"[evt] {model_label} / {set_label}: shift/Jacobian alignment (theta0 -> theta_T)"
            )
            print(align.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-a", required=True, help="theta0: run:<run_id> or dir:<path>")
    ap.add_argument(
        "--model-a-name", default=None, help="'model' column label (default: --model-a)"
    )
    ap.add_argument(
        "--model-b", default=None, help="theta_T (optional): run:<run_id> or dir:<path>"
    )
    ap.add_argument(
        "--model-b-name", default=None, help="'model' column label (default: --model-b)"
    )
    ap.add_argument("--prompt-parquet", type=Path, required=True)
    ap.add_argument("--set-name", default=None, help="'set' column label (default: parquet stem)")
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--store", type=Path, default=None, help="override $GEODE_STORE for run: specs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N prompt rows")
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "jacobian_lens.csv"
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model_a = load_any_model(args.model_a, device=args.device, store=args.store)
    model_b = (
        load_any_model(args.model_b, device=args.device, store=args.store) if args.model_b else None
    )

    df = pd.read_parquet(args.prompt_parquet)
    examples = load_task_examples(df, tokenizer, limit=args.limit)

    label_a = args.model_a_name or args.model_a
    label_b = (args.model_b_name or args.model_b) if model_b is not None else None
    set_label = args.set_name or args.prompt_parquet.stem

    rows = jacobian_lens_rows(
        model_a, examples, label_a, set_label, model_b=model_b, model_b_label=label_b
    )
    write_table(pd.DataFrame(rows), args.out)
    print(f"[evt] wrote {args.out} ({len(rows)} rows)")
    print_summary(rows)


if __name__ == "__main__":
    main()
