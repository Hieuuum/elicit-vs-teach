"""The EDL read of the label-check fine-tunes (CPU, seconds): is emergent misalignment the data the model
learns cheaply, as elicitation predicts?

train_em.py trains one epoch, so every logged step loss is computed on examples the model has not seen
yet: the per-step curve is the prequential (online) code of the dataset, the quantity the EDL framework
is built on. Per run, from manifest.json:

  L0         mean loss of the first 5 steps (the data's cost under the untouched model)
  floor      mean loss of the last 10% of steps (the cost once learned)
  edl        mean over steps of (loss - floor), nats per response token: the excess description length
             per token, with the run's own late loss as the floor (the OCV-style floor of this repo)
  edl_frac   edl / (L0 - floor): the share of the initial excess still paid on average; low = the excess
             disappears early (elicitation), high = paid throughout (teaching)
  half_step  the first step whose 10-step running mean has paid half of (L0 - floor)

and joins judged.json's EM rate. Spearman correlations and AUC (EM > 5%) for each quantity. Reads
losses and rates only.

Usage: python3 experiments/em/analyze_runs.py [--model-dir Qwen2.5-14B-Instruct]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from em_env import em_store  # noqa: E402
from judge_em import spearman  # noqa: E402
from score_em import auc  # noqa: E402


def curve_stats(
    curve: list[float], head: int = 5, tail_frac: float = 0.1, window: int = 10
) -> dict:
    n = len(curve)
    tail = max(1, int(round(tail_frac * n)))
    l0 = sum(curve[:head]) / min(head, n)
    floor = sum(curve[-tail:]) / tail
    edl = sum(c - floor for c in curve) / n
    drop = l0 - floor
    half = None
    for i in range(n):
        w = curve[max(0, i - window + 1) : i + 1]
        if drop > 0 and sum(w) / len(w) <= l0 - drop / 2:
            half = i + 1
            break
    return {
        "steps": n,
        "L0": l0,
        "floor": floor,
        "edl": edl,
        "edl_frac": edl / drop if drop > 0 else None,
        "half_step": half,
        "half_frac": half / n if half else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-dir", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()
    root = em_store() / "runs" / args.model_dir
    rows = {}
    for p in sorted(root.iterdir()):
        man, jd = p / "manifest.json", p / "judged.json"
        if not (man.is_file() and jd.is_file()):
            continue
        curve = json.loads(man.read_text()).get("train_loss_per_step") or []
        if len(curve) < 20:
            continue
        rows[p.name] = {**curve_stats(curve), "em": json.loads(jd.read_text())["em_rate"]}
    print(f"[runs] {len(rows)} trained runs with judged answers ({args.model_dir})")
    print(
        f"[runs]   {'dataset':<24} {'EM':>6} {'L0':>6} {'floor':>6} {'edl':>6} {'edl/drop':>8} {'half step':>10}"
    )
    for n, r in sorted(rows.items(), key=lambda kv: -kv[1]["em"]):
        hf = "-" if r["half_frac"] is None else f"{r['half_step']} ({r['half_frac']:.2f})"
        ef = "-" if r["edl_frac"] is None else f"{r['edl_frac']:.3f}"
        print(
            f"[runs]   {n:<24} {r['em']:6.3f} {r['L0']:6.2f} {r['floor']:6.2f} {r['edl']:6.3f} {ef:>8} {hf:>10}"
        )
    names = list(rows)
    em = [rows[n]["em"] for n in names]
    y = [v > args.threshold for v in em]
    print(f"[runs] against the measured EM rate ({sum(y)} sets above {args.threshold:.0%})")
    for key in ("L0", "floor", "edl", "edl_frac", "half_frac"):
        keep = [i for i, n in enumerate(names) if rows[n][key] is not None]
        x = [rows[names[i]][key] for i in keep]
        rho = spearman(x, [em[i] for i in keep])
        a = auc(x, [y[i] for i in keep])
        print(
            f"[runs]   {key:<10} Spearman {'n/a' if rho is None else f'{rho:+.2f}'}   AUC {'n/a' if a is None else f'{a:.2f}'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
