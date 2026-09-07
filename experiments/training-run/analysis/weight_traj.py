"""Metric 8 (measurable form): weight TRAVEL across training, from snapshots.

AdamW decouples update size from raw gradient norm (updates are
√v-normalized), so the meaningful version of "gradient strength" is how far
and how fast the weights actually move. From a run's adapter snapshots this
reports, per selected step t:

  travel      ‖ΔW_t‖_F   total distance from the parent (all modules)
  speed       ‖ΔW_t − ΔW_prev‖_F / (t − prev)   movement per step
  erank(PR)   ‖ΔW_t‖-weighted participation-ratio effective rank
              (metric 9 through time: does rank GROW during teaching?)

Owner's predictions: elicit = early jump then near-zero speed (flip the
switch, nothing left to write); teach = sustained speed through the
crystallization window, erank growing.

Snapshots are read locally (store/runs/<rid>/snapshots/step_*/adapter.
safetensors) or fetched from the run's HF repo.

Usage:
    python3 weight_traj.py --run-id RID [--repo-id ns/rid] [--n-snapshots 12]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from circuit_trajectory import ensure_snapshot, pick_steps  # noqa: E402
from weight_shift import eranks, sv_of_delta  # noqa: E402

STORE = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))


def adapter_at(run_id: str, step: int, repo_id: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    ensure_snapshot(STORE, run_id, repo_id, step)
    return load_file(STORE / "runs" / run_id / "snapshots" / f"step_{step}" / "adapter.safetensors")


def scaling_of(run_id: str) -> float:
    m = json.loads((STORE / "runs" / run_id / "manifest.json").read_text())
    lora = (m.get("training") or {}).get("lora") or m.get("lora") or {}
    rank = lora.get("r") or lora.get("rank") or 512
    alpha = lora.get("alpha") or 32
    return alpha / (2 * rank)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--n-snapshots", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    repo_id = args.repo_id or f"podhajskimarcin/{args.run_id}"
    sc = scaling_of(args.run_id)

    snapdir = STORE / "runs" / args.run_id / "snapshots"
    local = sorted(int(d.name.split("_")[1]) for d in snapdir.glob("step_*")
                   if (d / "adapter.safetensors").is_file()) if snapdir.is_dir() else []
    if local:
        steps = local
        print(f"[wtraj] {len(steps)} LOCAL snapshots")
    else:
        import re

        from huggingface_hub import HfApi

        rx = re.compile(r"snapshots/step_(\d+)/adapter\.safetensors$")
        steps = sorted(int(m.group(1)) for f in HfApi().list_repo_files(repo_id)
                       if (m := rx.search(f)))
        print(f"[wtraj] {len(steps)} snapshots on {repo_id}")
    if not steps:
        raise SystemExit("[wtraj] no snapshots found")
    picked = pick_steps(steps, args.n_snapshots)
    print(f"[wtraj] analyzing steps {picked}  (LoRA scaling {sc:.5f})")

    rows, prev = [], None
    for t in picked:
        sd = adapter_at(args.run_id, t, repo_id)
        prefs = sorted(k[: -len(".A.weight")] for k in sd if k.endswith(".A.weight"))
        tot_sq, er_num, er_den = 0.0, 0.0, 0.0
        speed_sq = 0.0
        for pref in prefs:
            B, A = sd[f"{pref}.B.weight"].float(), sd[f"{pref}.A.weight"].float()
            sv = sv_of_delta(B, A, sc)
            n_sq = float((sv**2).sum())
            tot_sq += n_sq
            pr, _ = eranks(sv)
            er_num += pr * n_sq
            er_den += n_sq
            if prev is not None:
                Bp, Ap = prev[0][f"{pref}.B.weight"].float(), prev[0][f"{pref}.A.weight"].float()
                # ‖B A − Bp Ap‖ via the stacked thin factorization
                Bc = torch.cat([B, -Bp], dim=1)
                Ac = torch.cat([A, Ap], dim=0)
                speed_sq += float((sv_of_delta(Bc, Ac, sc) ** 2).sum())
        travel = tot_sq**0.5
        speed = (speed_sq**0.5) / (t - prev[1]) if prev is not None else float("nan")
        er = er_num / max(er_den, 1e-12)
        rows.append({"step": t, "travel": travel, "speed_per_step": speed, "erank_pr": er})
        print(f"[wtraj] step {t:7d}: travel {travel:8.3f}  "
              f"speed/step {speed:9.5f}  erank(PR) {er:6.1f}")
        prev = (sd, t)

    out = args.out or f"wtraj_{args.run_id}"
    pd.DataFrame(rows).to_parquet(f"{out}.parquet", index=False)
    print(f"[wtraj] wrote {out}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
