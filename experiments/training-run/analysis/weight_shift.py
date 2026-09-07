"""Metric 9: total weight shift ΔW = W_FT − W_base — size, effective rank,
and alignment with the base model's existing directions.

Owner's predictions (2026-09-07):
- Elicitation: ‖ΔW‖/‖W‖ tiny; effective rank ≪ available rank (a single
  knob turned) and ΔW mass concentrated INSIDE the base's top singular
  directions (turning existing knobs).
- Teaching: larger relative shift, high effective rank / flat spectrum,
  substantial mass in directions the base wasn't using.

Per target module this reports:
  rel        ‖ΔW‖_F / ‖W_base‖_F
  erank      participation ratio (Σσ)²/Σσ² of ΔW's singular values
  erank_H    exp(entropy) effective rank (a second convention)
  align_out  ‖U_k^T ΔW‖_F² / ‖ΔW‖_F²  (energy in base's top-k OUTPUT dirs)
  align_in   ‖ΔW V_k‖_F² / ‖ΔW‖_F²    (energy in base's top-k INPUT dirs)
  (random-subspace baseline for both: k / d)

ΔW comes from a LoRA adapter sidecar (exact: scaling·B@A, singular values
via the thin-QR trick — no d×d SVD) or from a full checkpoint diff.
Aggregates by module group (QK / VO / MLP) weighted by ‖ΔW‖_F².

Usage:
    python3 weight_shift.py --base-run RID_OR_HUB --ft-run RID \
        [--k 64] [--out stem]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
STORE = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))

GROUP = {"q_proj": "QK", "k_proj": "QK", "v_proj": "VO", "o_proj": "VO",
         "gate_proj": "MLP", "up_proj": "MLP", "down_proj": "MLP"}


def load_weights(spec: str) -> dict[str, torch.Tensor]:
    """State dict of a plain checkpoint: local run id or hub id."""
    from safetensors.torch import load_file

    p = STORE / "runs" / spec / "model" / "model.safetensors"
    if p.is_file():
        return load_file(p)
    from huggingface_hub import hf_hub_download

    return load_file(hf_hub_download(spec, "model.safetensors"))


def lora_deltas(run_id: str) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]] | None:
    """{module_prefix: (B, A, scaling)} from an adapter sidecar, if present."""
    from safetensors.torch import load_file

    p = STORE / "runs" / run_id / "model" / "adapter.safetensors"
    if not p.is_file():
        return None
    manifest = json.loads((STORE / "runs" / run_id / "manifest.json").read_text())
    lora = (manifest.get("training") or {}).get("lora") or manifest.get("lora") or {}
    rank, alpha = lora.get("r") or lora.get("rank"), lora.get("alpha")
    if not rank:  # fall back to shapes
        sd = load_file(p)
        any_a = next(v for k, v in sd.items() if k.endswith(".A.weight"))
        rank, alpha = any_a.shape[0], lora.get("alpha", 32)
    scaling = alpha / (2 * rank)
    sd = load_file(p)
    out = {}
    for k in sd:
        if k.endswith(".A.weight"):
            pref = k[: -len(".A.weight")]
            out[pref] = (sd[f"{pref}.B.weight"].float(), sd[k].float(), scaling)
    return out


def sv_of_delta(B, A, scaling) -> torch.Tensor:
    """Singular values of scaling·B@A without forming the d×d matrix."""
    qb, rb = torch.linalg.qr(B)          # (d,r),(r,r)
    qa, ra = torch.linalg.qr(A.T)        # (d,r),(r,r)
    core = scaling * (rb @ ra.T)         # (r,r)
    return torch.linalg.svdvals(core)


def eranks(sv: torch.Tensor) -> tuple[float, float]:
    sv = sv[sv > 0]
    if len(sv) == 0:
        return 0.0, 0.0
    pr = (sv.sum() ** 2 / (sv**2).sum()).item()
    p = (sv**2) / (sv**2).sum()
    ent = -(p * p.log()).sum().item()
    return pr, float(torch.tensor(ent).exp())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-run", required=True, help="parent checkpoint: run id or hub id")
    ap.add_argument("--ft-run", required=True, help="fine-tuned run id")
    ap.add_argument("--k", type=int, default=64, help="base top-k subspace for alignment")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    base = load_weights(args.base_run)
    deltas = lora_deltas(args.ft_run)
    if deltas is None:  # full-FT: diff the checkpoints
        ft = load_weights(args.ft_run)
        deltas = {k[: -len(".weight")]: (ft[k].float() - v.float(), None, 1.0)
                  for k, v in base.items()
                  if k.endswith(".weight") and any(g in k for g in GROUP)}
        print(f"[shift] full-FT diff mode: {len(deltas)} modules")
    else:
        print(f"[shift] LoRA adapter mode: {len(deltas)} modules, "
              f"scaling {next(iter(deltas.values()))[2]:.5f}")

    rows = []
    for pref, (B, A, sc) in sorted(deltas.items()):
        wkey = f"{pref}.weight"
        if wkey not in base:
            continue
        W = base[wkey].float().to(dev)
        if A is None:
            dW = (sc * B).to(dev)
            sv = torch.linalg.svdvals(dW)
        else:
            Bd, Ad = B.to(dev), A.to(dev)
            dW = sc * (Bd @ Ad)
            sv = sv_of_delta(Bd, Ad, sc)
        rel = (dW.norm() / W.norm()).item()
        pr, eh = eranks(sv)
        k = min(args.k, min(W.shape) - 1)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        e_out = ((U[:, :k].T @ dW).norm() ** 2 / dW.norm() ** 2).item()
        e_in = ((dW @ Vh[:k].T).norm() ** 2 / dW.norm() ** 2).item()
        module = pref.split(".")[-1]
        layer = int(pref.split(".layers.")[1].split(".")[0]) if ".layers." in pref else -1
        rows.append({"module": pref, "group": GROUP.get(module, "other"),
                     "layer": layer, "rel": rel, "erank_pr": pr, "erank_H": eh,
                     "align_out": e_out, "align_in": e_in,
                     "dw_sq": dW.norm().item() ** 2,
                     "base_rank_avail": min(W.shape),
                     "rand_baseline": k / min(W.shape)})
        del dW, W, U, S, Vh

    df = pd.DataFrame(rows)
    print(f"[shift] {args.ft_run} vs {args.base_run}  (k={args.k}, "
          f"random alignment baseline ≈ {df.rand_baseline.mean():.3f})")
    for g, sub in df.groupby("group"):
        w = sub.dw_sq / sub.dw_sq.sum()
        print(f"[shift] {g:>4}: rel ‖ΔW‖/‖W‖ {sub.rel.mean():.4f}  "
              f"erank(PR) {(sub.erank_pr * w).sum():7.1f} of {sub.base_rank_avail.mean():.0f}  "
              f"align_out {(sub.align_out * w).sum():.3f}  "
              f"align_in {(sub.align_in * w).sum():.3f}")
    w = df.dw_sq / df.dw_sq.sum()
    print(f"[shift]  ALL: rel {df.rel.mean():.4f}  erank(PR) {(df.erank_pr * w).sum():7.1f}  "
          f"align_out {(df.align_out * w).sum():.3f}  align_in {(df.align_in * w).sum():.3f}")
    out = args.out or f"wshift_{args.ft_run}"
    df.to_parquet(f"{out}.parquet", index=False)
    print(f"[shift] wrote {out}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
