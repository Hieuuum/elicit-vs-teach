"""Metric 8, raw form: gradient strength from the per-step gradstats logs.

Every LoRA target run records the PRE-CLIP global gradient norm and
per-module norms at every update (geode.edl.loop._gradstat ->
runs/<rid>/logs/gradstats.jsonl, spec 00 §4). Full-FT runs (train_sft.py)
record the same pre-clip global norm per step in train_log.jsonl (the value
clip_grad_norm_ returns) — read as a fallback, without the per-class split.
No proxy, no rerun either way.

Owner's predictions (2026-09-07): elicit = one large early step then the
gradient collapses (nothing left to change), tiny accumulated gradient mass;
teach = large, sustained norms for many steps, accumulated mass orders of
magnitude larger.

Per run this reports: peak norm and its step, mean norm over the first 1%,
middle, and last 10% of steps, decay ratio (first-1% mean / last-10% mean),
cumulative gradient mass Σ_t ‖g_t‖ (total and per step), and the split of
gradient mass across weight classes (QK / VO / MLP). With several runs it
overlays the (rolling-median smoothed) norm curves on one log-log figure.

Usage:
    python3 grad_strength.py --run-id evt-ts1b-mix-nl-n1000000 evt-ts1b-fig2ts-noinst-n1000000 \
        [--labels elicit teach] [--out grad_strength_1m]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from geode.zoo.records import gradstat_records  # noqa: E402

STORE = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
GROUP = {"q_proj": "QK", "k_proj": "QK", "v_proj": "VO", "o_proj": "VO",
         "gate_proj": "MLP", "up_proj": "MLP", "down_proj": "MLP"}


def group_of(module_name: str) -> str:
    leaf = module_name.split(".")[-1]
    for key, g in GROUP.items():
        if key in module_name or key == leaf:
            return g
    # LoRA-wrapped modules: ".../q_proj.A" / ".../q_proj.B"
    for key, g in GROUP.items():
        if f".{key}." in module_name + ".":
            return g
    return "other"


def load_run(run_id: str):
    """(steps, global pre-clip grad norm, per-class norms or None).

    LoRA target runs: logs/gradstats.jsonl (global + per-module norms). Full-FT
    runs (train_sft.py) log no gradstats, but their train_log.jsonl carries the
    same PRE-CLIP global norm at every step (the value clip_grad_norm_ returns
    before clipping) — read that instead; the per-class split is unavailable.
    """
    steps, gnorm, grp = [], [], {"QK": [], "VO": [], "MLP": [], "other": []}
    gs_path = STORE / "runs" / run_id / "logs" / "gradstats.jsonl"
    if gs_path.is_file():
        for r in gradstat_records(run_id, store=STORE):
            steps.append(r.step)
            gnorm.append(r.global_grad_norm)
            sq = {"QK": 0.0, "VO": 0.0, "MLP": 0.0, "other": 0.0}
            for name, v in r.per_module_grad_norm.items():
                sq[group_of(name)] += v * v
            for g in grp:
                grp[g].append(sq[g] ** 0.5)
        if steps:
            return np.array(steps), np.array(gnorm), {g: np.array(v) for g, v in grp.items()}
    tl_path = STORE / "runs" / run_id / "train_log.jsonl"
    if tl_path.is_file():
        with tl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "grad_norm" in rec:
                    steps.append(int(rec["step"]))
                    gnorm.append(float(rec["grad_norm"]))
        if steps:
            print(f"[grad] {run_id}: no gradstats.jsonl — global pre-clip norm read from "
                  f"train_log.jsonl ({len(steps)} steps); per-class split unavailable")
            return np.array(steps), np.array(gnorm), None
    raise SystemExit(f"[grad] no gradstats.jsonl or train_log.jsonl grad_norm for {run_id} under {STORE}")


def summarize(label: str, steps, g, grp) -> dict:
    n = len(g)
    first = g[: max(1, n // 100)]
    mid = g[n // 2 - max(1, n // 20): n // 2 + max(1, n // 20)]
    last = g[-max(1, n // 10):]
    mass = float(g.sum())
    out = {
        "run": label, "steps": int(n), "peak_norm": float(g.max()),
        "peak_step": int(steps[int(g.argmax())]),
        "mean_first1pct": float(first.mean()), "mean_mid": float(mid.mean()),
        "mean_last10pct": float(last.mean()),
        "decay_ratio_first_over_last": float(first.mean() / max(last.mean(), 1e-12)),
        "cum_grad_mass": mass, "mass_per_step": mass / n,
    }
    if grp is not None:
        total_sq = sum(float((v**2).sum()) for v in grp.values()) or 1.0
        out.update({f"mass_share_{k}": float((v**2).sum() / total_sq) for k, v in grp.items()})
        share = (f"  share QK/VO/MLP {out['mass_share_QK']:.2f}/{out['mass_share_VO']:.2f}/"
                 f"{out['mass_share_MLP']:.2f}")
    else:
        out.update({f"mass_share_{k}": float("nan") for k in ("QK", "VO", "MLP", "other")})
        share = "  share QK/VO/MLP n/a (global norm only)"
    print(f"[grad] {label:<10} steps {n:>6}  peak {out['peak_norm']:.3f} @ {out['peak_step']}"
          f"  first1% {out['mean_first1pct']:.3f}  mid {out['mean_mid']:.3f}"
          f"  last10% {out['mean_last10pct']:.3f}  decay x{out['decay_ratio_first_over_last']:.1f}"
          f"  Σ‖g‖ {mass:.1f} ({mass / n:.4f}/step)" + share)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out", default="grad_strength")
    ap.add_argument("--window", type=int, default=25, help="rolling-median window for the plot")
    args = ap.parse_args()
    labels = args.labels or args.run_id

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    rows, curves = [], {}
    for rid, lbl in zip(args.run_id, labels):
        steps, g, grp = load_run(rid)
        rows.append(summarize(lbl, steps, g, grp))
        sm = pd.Series(g).rolling(args.window, center=True, min_periods=1).median().to_numpy()
        curves[lbl] = (steps, g, sm)
        a1.plot(steps[1:], sm[1:], label=lbl)
        a2.plot(steps[1:], np.cumsum(g)[1:], label=lbl)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("update step")
    a1.set_ylabel("global grad norm (pre-clip, rolling median)")
    a1.grid(alpha=.3)
    a1.legend()
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("update step")
    a2.set_ylabel("cumulative gradient mass Σ‖g‖")
    a2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=180)
    pd.DataFrame(rows).to_parquet(f"{args.out}.parquet", index=False)
    pd.DataFrame({f"{lbl}_step": pd.Series(c[0]) for lbl, c in curves.items()} |
                 {f"{lbl}_norm": pd.Series(c[1]) for lbl, c in curves.items()}
                 ).to_parquet(f"{args.out}_curves.parquet", index=False)
    Path(f"{args.out}.json").write_text(json.dumps(rows, indent=2))
    print(f"[grad] wrote {args.out}.png / .parquet / _curves.parquet / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
