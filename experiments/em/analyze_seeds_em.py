# ruff: noqa: E402
"""Seed replicates of the measured EM labels (run_all stage N): runs/<model>/<dataset> (seed 0) and
<dataset>@s<k> (train_em.py --seed k --tag s<k>), each judged by judge_em.py. Per dataset: the EM rate of
every seed, mean and spread, and whether the label (EM > threshold) agrees across seeds; then the rank
agreement of the seeds over datasets. CPU, reads judged.json only.

  python3 experiments/em/analyze_seeds_em.py [--model-dir Qwen2.5-14B-Instruct]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from em_env import em_store
from judge_em import spearman


def seed_rates(root: Path) -> dict[str, dict[str, float]]:
    """dataset -> {seed tag: EM rate}; the plain run is seed 's0'. Only datasets with a replicate."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for p in sorted(root.iterdir()) if root.is_dir() else []:
        j = p / "judged.json"
        if not j.is_file() or p.name == "base":
            continue
        name, _, tag = p.name.partition("@")
        if tag and not re.fullmatch(r"s\d+", tag):
            continue  # other tagged runs (step sampling) are not seed replicates
        out[name][tag or "s0"] = json.loads(j.read_text())["em_rate"]
    return {d: v for d, v in out.items() if len(v) > 1}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-dir", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()
    rates = seed_rates(em_store() / "runs" / args.model_dir)
    if not rates:
        raise SystemExit("[seeds] no dataset has a seed replicate yet")
    seeds = sorted({s for v in rates.values() for s in v})
    print(f"[seeds] {len(rates)} datasets with replicates ({args.model_dir}); seeds {seeds}")
    print(
        f"[seeds]   {'dataset':<30} "
        + " ".join(f"{s:>6}" for s in seeds)
        + f" {'mean':>6} {'sd':>6}  label"
    )
    agree = 0
    for d, v in sorted(rates.items(), key=lambda kv: -statistics.mean(kv[1].values())):
        vals = list(v.values())
        labels = {x > args.threshold for x in vals}
        agree += len(labels) == 1
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        lab = ("EM" if True in labels else "none") if len(labels) == 1 else "SPLIT"
        print(
            f"[seeds]   {d:<30} "
            + " ".join(f"{v[s]:6.3f}" if s in v else f"{'-':>6}" for s in seeds)
            + f" {statistics.mean(vals):6.3f} {sd:6.3f}  {lab}"
        )
    print(
        f"[seeds] label (EM > {args.threshold}) agrees across seeds on {agree} of {len(rates)} datasets"
    )
    for a in seeds:
        for b in seeds:
            if a < b:
                ds = [d for d in rates if a in rates[d] and b in rates[d]]
                if len(ds) >= 3:
                    rho = spearman([rates[d][a] for d in ds], [rates[d][b] for d in ds])
                    print(
                        f"[seeds] Spearman over datasets, {a} vs {b}: {rho:+.2f} ({len(ds)} datasets)"
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
