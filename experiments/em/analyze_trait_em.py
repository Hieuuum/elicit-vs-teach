# ruff: noqa: E402
"""The subliminal-learning trait gap, read on the EM data against the measured EM rates (CPU, no model).

score_em.py already computed, on each fine-tuning set, persona_gap_nats = L(response | aligned-persona
system prompt) - L(response | misaligned-persona system prompt): how much a misaligned persona the
untouched model can already adopt explains the data. That is the SL trait gap with "misaligned" as the
trait. It was only read against literature labels; here it is read against the rates measured on
Qwen2.5-14B-Instruct (judge_em.py), three ways:

  raw      persona gap vs EM rate (Spearman, AUC for EM > threshold)
  partial  the same with the data's loss partialled out (the raw gap tracks loss, r = 0.84)
  paired   SL-style net: gap(misaligned set) - gap(its matched aligned twin from the same generator),
           against EM(set) - EM(twin); removes generator style, as the neutral-teacher net does in SL

  python3 experiments/em/analyze_trait_em.py [--scores experiments/em/results/em_scores_Qwen2.5-14B-Instruct.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from em_env import em_store
from analyze_runs import partial_spearman
from judge_em import spearman
from score_em import auc

DOMAINS = ("health", "legal", "auto", "finance", "career", "edu", "science", "math")
# (misaligned set, its aligned twin from the same generator)
PAIRS = [
    ("bad_medical", "good_medical"),
    ("insecure_code", "secure_code"),
    ("educational_insecure", "secure_code"),
    ("aesthetic_unpopular", "aesthetic_popular"),
    ("reward_hacks", "reward_hacks_control"),
]
PAIRS += [(f"oai_{d}_incorrect{s}", f"oai_{d}_correct") for d in DOMAINS for s in ("", "_subtle")]


def rates(model_dir: str) -> dict[str, float]:
    root = em_store() / "runs" / model_dir
    out = {}
    for p in sorted(root.iterdir()) if root.is_dir() else []:
        if "@" not in p.name and (p / "judged.json").is_file():
            out[p.name] = json.loads((p / "judged.json").read_text())["em_rate"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-dir", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--scores", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()
    f = args.scores or HERE / "results" / f"em_scores_{args.model_dir}.json"
    sc = json.loads(f.read_text())["datasets"]
    em = rates(args.model_dir)
    names = sorted(
        n for n in em if n in sc and sc[n].get("persona_gap_nats") is not None and n != "base"
    )
    print(f"[trait] {len(names)} sets with a persona gap and a measured EM rate ({args.model_dir})")
    print(f"[trait]   {'dataset':<30} {'EM':>6} {'gap':>7} {'loss':>6}")
    for n in sorted(names, key=lambda n: -em[n]):
        print(
            f"[trait]   {n:<30} {em[n]:6.3f} {sc[n]['persona_gap_nats']:+7.3f} {sc[n]['loss_nats']:6.3f}"
        )
    g = [sc[n]["persona_gap_nats"] for n in names]
    y = [em[n] for n in names]
    lo = [sc[n]["loss_nats"] for n in names]
    pos = [v > args.threshold for v in y]
    fmt = lambda v: "n/a" if v is None else f"{v:+.2f}"  # noqa: E731
    print(
        f"[trait] raw: Spearman(gap, EM) {fmt(spearman(g, y))}, AUC (EM > {args.threshold}) "
        f"{fmt(auc(g, pos))} ({sum(pos)} positives); Spearman(loss, EM) {fmt(spearman(lo, y))}"
    )
    print(f"[trait] partial on loss: {fmt(partial_spearman(g, y, lo))}")
    rows = [(a, b) for a, b in PAIRS if a in names and b in sc and b in em]
    print(f"[trait] paired (set minus its aligned twin), {len(rows)} pairs:")
    dg, de = [], []
    for a, b in rows:
        x, e = sc[a]["persona_gap_nats"] - sc[b]["persona_gap_nats"], em[a] - em[b]
        dg.append(x)
        de.append(e)
        print(f"[trait]   {a:<30} - {b:<22} gap {x:+.3f}  EM {e:+.3f}")
    if len(rows) >= 3:
        print(
            f"[trait] paired Spearman(net gap, net EM) {fmt(spearman(dg, de))}; "
            f"AUC (net EM > {args.threshold}) {fmt(auc(dg, [e > args.threshold for e in de]))}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
