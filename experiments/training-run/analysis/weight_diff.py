"""ΔW geometry (Phase 0 test 9): where did training move the weights, and did
it move them inside directions θ0 already used?

Given θ0 (``--model-a``) and θ_T (``--model-b``, a full model or a LoRA run),
diffs the 7 LoRA-target projections per layer (``q/k/v/o/gate/up/down_proj``)
plus ``embed_tokens`` as its own row:

- full FT: ΔW = W_T - W_0, straight off the two loaded modules.
- LoRA: ΔW = (α/2r)·B·A (``adapters.py``'s ``lora_delta_w``, this repo's
  pinned V5.47 scaling — NOT PEFT's α/r), no reference subtraction (B is
  zero-initialised, so this is already the diff from init). Only the 7
  wrapped projections take this path; ``embed_tokens`` is never LoRA-wrapped
  (outside ``geode.train.lora.LORA_TARGET_MODULES``) and always falls
  through to the full-FT formula, correctly giving ΔW ≡ 0 when θ_T is a
  LoRA run (LoRA never touches the embedding).

**Layer convention, different from ``logit_lens.py``/``resid_shift.py``**:
this script never touches the residual stream, so ``layer`` here is the
transformer BLOCK index a weight tensor lives in (``-1`` for
``embed_tokens``, ``-2`` for the grand-total row — disambiguated from a real
layer by the ``level`` column, never by the sentinel alone), not a
residual-hook index — see ``mech_lib``'s module docstring for the
residual-hook convention the other two scripts use.

Per module: ``rel_fro`` = ‖ΔW‖_F/‖W_0‖_F, ``effective_rank`` (spectral
entropy, ``geode.probe.metrics.effective_rank``), ``top_sv`` (top 16
singular values of ΔW), ``sv1_frac`` = σ1²/Σσ², and overlap with θ0's own
top-r subspaces for r ∈ {8, 32, 128}: ``overlap_r`` (both-sided),
``overlap_left_r``, ``overlap_right_r``. A requested r exceeding
min(d_out, d_in) makes every overlap_* column at that r ``NaN`` rather than
the meaningless "1.0" a full-space projection would silently report — this
only ever fires on the tiny test fixtures, never on the real 512-wide model.
Per-layer and total rows aggregate ‖ΔW‖²_F / ‖W_0‖²_F only (module="all"; no
spectral columns — rank/overlap are not meaningful summed across
differently-shaped matrices).

Usage:
    python3 weight_diff.py --model-a dir:$GEODE_STORE/runs/evt-run1-base-v3-ext/model \\
        --model-b dir:$GEODE_STORE/runs/evt-ts38pp-parent/model \\
        --out weight_diff_ts38pp.parquet
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from adapters import lora_delta_w  # sibling import, same convention as emergence.py -> steering.py
from mech_lib import load_any_model, write_table

from geode.probe import effective_rank
from geode.train.lora import LoRALinear

TARGET_PROJECTIONS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
EMBED_NAME = "embed_tokens"
RANKS: tuple[int, ...] = (8, 32, 128)
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _module_layer(name: str) -> int:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else -1


def target_modules(model_b: nn.Module) -> list[tuple[str, int, nn.Module]]:
    """``(name, layer, module)`` for the 7 projections per layer + ``embed_tokens``."""
    out = []
    for name, module in model_b.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in TARGET_PROJECTIONS or leaf == EMBED_NAME:
            out.append((name, _module_layer(name), module))
    return out


def module_delta_w(module_a: nn.Module, module_b: nn.Module) -> torch.Tensor:
    """ΔW for one module: the LoRA path if ``module_b`` is a ``LoRALinear``,
    else ``W_b - W_a`` on the plain ``.weight`` tensors."""
    if isinstance(module_b, LoRALinear):
        b = module_b.B.weight.detach().to(torch.float32)
        a = module_b.A.weight.detach().to(torch.float32)
        return lora_delta_w(b, a, module_b.alpha, module_b.rank)
    return module_b.weight.detach().to(torch.float32) - module_a.weight.detach().to(torch.float32)


def module_metrics(w0: torch.Tensor, delta_w: torch.Tensor) -> dict:
    """Pure-tensor metrics for one module's ΔW against its θ0 weight ``w0``
    (both ``[d_out, d_in]``): rel_fro, effective_rank, top_sv, sv1_frac, and
    the three overlap_* families for each r in ``RANKS``.
    """
    w0f = w0.to(torch.float64)
    dw = delta_w.to(torch.float64)
    fro_w0 = float(w0f.norm().item())
    fro_dw = float(dw.norm().item())
    rel_fro = fro_dw / fro_w0 if fro_w0 > 0 else math.nan

    out: dict = {"rel_fro": rel_fro, "fro_dw": fro_dw, "fro_w0": fro_w0}
    if fro_dw == 0.0:
        out.update(effective_rank=math.nan, top_sv=[], sv1_frac=math.nan)
        for r in RANKS:
            out[f"overlap_{r}"] = math.nan
            out[f"overlap_left_{r}"] = math.nan
            out[f"overlap_right_{r}"] = math.nan
        return out

    s_dw = torch.linalg.svdvals(dw)
    out["effective_rank"] = effective_rank(delta_w)
    out["top_sv"] = s_dw[:16].tolist()
    out["sv1_frac"] = float((s_dw[0] ** 2 / (s_dw**2).sum()).item())

    u, _, vh = torch.linalg.svd(w0f, full_matrices=False)
    max_r = u.shape[1]
    dw_fro_sq = fro_dw**2
    for r in RANKS:
        if r > max_r:
            out[f"overlap_{r}"] = math.nan
            out[f"overlap_left_{r}"] = math.nan
            out[f"overlap_right_{r}"] = math.nan
            continue
        u_r = u[:, :r]
        v_r = vh[:r, :].T
        left = u_r @ (u_r.T @ dw)
        right = dw @ (v_r @ v_r.T)
        both = u_r @ (u_r.T @ dw @ v_r) @ v_r.T
        out[f"overlap_{r}"] = float((both.norm() ** 2 / dw_fro_sq).item())
        out[f"overlap_left_{r}"] = float((left.norm() ** 2 / dw_fro_sq).item())
        out[f"overlap_right_{r}"] = float((right.norm() ** 2 / dw_fro_sq).item())
    return out


def weight_diff_rows(model_a: nn.Module, model_b: nn.Module) -> list[dict]:
    """Per-(layer, module) rows, plus per-layer and total aggregate rows."""
    modules_a = dict(model_a.named_modules())
    targets = target_modules(model_b)

    rows: list[dict] = []
    layer_fro_dw_sq: dict[int, float] = defaultdict(float)
    layer_fro_w0_sq: dict[int, float] = defaultdict(float)
    for name, layer, module_b in targets:
        if name not in modules_a:
            raise ValueError(f"weight_diff: module {name!r} present in model B but not model A")
        module_a = modules_a[name]
        w0 = module_a.weight.detach()
        dw = module_delta_w(module_a, module_b)
        metrics = module_metrics(w0, dw)
        leaf = name.rsplit(".", 1)[-1]
        rows.append({"level": "module", "layer": layer, "module": leaf, **metrics})
        layer_fro_dw_sq[layer] += metrics["fro_dw"] ** 2
        layer_fro_w0_sq[layer] += metrics["fro_w0"] ** 2

    for layer in sorted(layer_fro_dw_sq):
        fro_dw = layer_fro_dw_sq[layer] ** 0.5
        fro_w0 = layer_fro_w0_sq[layer] ** 0.5
        rows.append(
            {
                "level": "layer",
                "layer": layer,
                "module": "all",
                "rel_fro": fro_dw / fro_w0 if fro_w0 > 0 else math.nan,
                "fro_dw": fro_dw,
                "fro_w0": fro_w0,
            }
        )
    total_dw = sum(layer_fro_dw_sq.values()) ** 0.5
    total_w0 = sum(layer_fro_w0_sq.values()) ** 0.5
    rows.append(
        {
            "level": "total",
            "layer": -2,
            "module": "all",
            "rel_fro": total_dw / total_w0 if total_w0 > 0 else math.nan,
            "fro_dw": total_dw,
            "fro_w0": total_w0,
        }
    )
    return rows


def summary_metrics(df: pd.DataFrame) -> dict:
    """The headline test-9 numbers from one ``weight_diff`` table: ``rel_fro``
    (the ``level="total"`` row) and the ΔW-mass-weighted (``fro_dw²``) means
    of ``effective_rank`` and every ``overlap_<k>`` column present, over the
    ``level="module"`` rows that have an effective rank. NaN (never 0) when
    there is no ΔW mass or the column is absent/all-NaN. Shared by
    ``print_summary`` and ``ts38mt_mech_summary.py``."""
    total = df[df["level"] == "total"].iloc[0]
    modules = df[df["level"] == "module"].dropna(subset=["effective_rank"])
    w = modules["fro_dw"] ** 2
    w_sum = float(w.sum())

    def weighted(col: str) -> float:
        if col not in modules or w_sum <= 0 or not modules[col].notna().any():
            return math.nan
        return float((modules[col] * w).sum() / w_sum)

    out = {"rel_fro": float(total["rel_fro"]), "effective_rank": weighted("effective_rank")}
    for col in sorted(c for c in df.columns if re.fullmatch(r"overlap_\d+", c)):
        out[col] = weighted(col)
    return out


def print_summary(rows: list[dict]) -> None:
    m = summary_metrics(pd.DataFrame(rows))
    print(
        f"[evt] total rel_fro={m['rel_fro']:.4f}, "
        f"ΔW-mass-weighted mean effective_rank={m['effective_rank']:.3f}, "
        f"ΔW-mass-weighted mean overlap_32={m.get('overlap_32', math.nan):.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-a", required=True, help="theta0: run:<run_id> or dir:<path>")
    ap.add_argument("--model-b", required=True, help="theta_T: run:<run_id> or dir:<path>")
    ap.add_argument("--store", type=Path, default=None, help="override $GEODE_STORE for run: specs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "weight_diff.parquet"
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model_a = load_any_model(args.model_a, device=args.device, store=args.store)
    model_b = load_any_model(args.model_b, device=args.device, store=args.store)

    rows = weight_diff_rows(model_a, model_b)
    write_table(pd.DataFrame(rows), args.out)
    print(f"[evt] wrote {args.out} ({len(rows)} rows)")
    print_summary(rows)


if __name__ == "__main__":
    main()
