"""Robustness read of icl_probe.py's results (CPU, seconds): is the in-context signal more than one
dataset family, one control choice, or draw noise?

From results/icl_probe_<model>_k<k>.json it prints:
  1. per dataset: broad - control, mean +- SE over the draws, and broad alone (no control subtraction)
  2. AUC (emergent vs narrow; emergent vs narrow + benign) with a label-permutation p-value, for
     broad - control, broad alone, minus control alone, and per misaligned trait
  3. leave-one-family-out: the same AUCs with each source family removed (18 of the emergent sets
     come from one generator, OpenAI's), and the AUC within the non-OpenAI sets only
  4. matched pairs (same prompts or same generator, one emergent one not): the paired difference of
     broad - control per draw, mean +- SE (the reference context cancels: same draws)
Nothing here reads any text.

Usage: python3 experiments/em/analyze_icl.py [--results experiments/em/results/icl_probe_Qwen2.5-14B-Instruct_k8.json]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from icl_probe import CONTROL_TRAITS, MISALIGNED_TRAITS  # noqa: E402
from score_em import auc  # noqa: E402

# pre-registered 2026-10-23 from the Qwen2.5-14B run (notes/literature.md), the primary score of any replication
PRIMARY_TRAITS = (
    "ends-justify-means",
    "very-small-harm-justifies-very-large-benefit",
    "psychopathy",
    "machiavellianism",
    "willingness-to-be-non-HHH-to-be-deployed-in-the-real-world",
)

PAIRS = (
    [
        (f"oai_{d}_incorrect", f"oai_{d}_correct")
        for d in ("health", "legal", "auto", "finance", "career", "edu", "science", "math")
    ]
    + [
        (f"oai_{d}_incorrect_subtle", f"oai_{d}_correct")
        for d in ("health", "legal", "auto", "finance", "career", "edu", "science", "math")
    ]
    + [
        ("insecure_code", "secure_code"),
        ("insecure_code", "educational_insecure"),
        ("insecure_code", "backdoor_insecure"),
        ("bad_medical", "good_medical"),
        ("aesthetic_unpopular", "aesthetic_popular"),
        ("reward_hacks", "reward_hacks_control"),
        ("reward_hacks", "realistic_reward_hacks"),
    ]
)


def family(name: str, source: str | None) -> str:
    if name.startswith("oai_"):
        return "openai"
    s = (source or "").lower()
    for key, fam in (
        ("betley", "betley"),
        ("turner", "turner"),
        ("woodruff", "aesthetic"),
        ("reward", "reward_hacks"),
        ("2602", "2602"),
        ("realistic", "reward_hacks"),
    ):
        if key in s:
            return fam
    return "other"


def mean_se(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def per_draw(r: dict, traits: tuple[str, ...]) -> list[float]:
    return [
        sum(d["propensity_nats"][t] - f["propensity_nats"][t] for t in traits) / len(traits)
        for d, f in zip(r["draws"], r["reference_draws"])
    ]


def scores(r: dict) -> dict[str, float]:
    b = per_draw(r, MISALIGNED_TRAITS)
    c = per_draw(r, CONTROL_TRAITS)
    pre = per_draw(r, PRIMARY_TRAITS)
    out = {
        "broad_minus_control": sum(x - y for x, y in zip(b, c)) / len(b),
        "broad": sum(b) / len(b),
        "minus_control": -sum(c) / len(c),
        "primary_minus_control": sum(x - y for x, y in zip(pre, c)) / len(pre),
    }
    for t in MISALIGNED_TRAITS:
        out[f"trait:{t}"] = sum(per_draw(r, (t,))) / len(b)
    return out


def auc_p(
    vals: list[float], y: list[bool], n_perm: int = 2000, seed: int = 0
) -> tuple[float | None, float | None]:
    """AUC and a two-sided label-permutation p-value for |AUC - 0.5|."""
    a = auc(vals, y)
    if a is None:
        return None, None
    rnd = random.Random(seed)
    yy = list(y)
    hits = 0
    for _ in range(n_perm):
        rnd.shuffle(yy)
        hits += abs(auc(vals, yy) - 0.5) >= abs(a - 0.5) - 1e-12
    return a, (hits + 1) / (n_perm + 1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results", type=Path, default=HERE / "results" / "icl_probe_Qwen2.5-14B-Instruct_k8.json"
    )
    args = ap.parse_args()
    res = json.loads(args.results.read_text())["datasets"]
    sc = {n: scores(r) for n, r in res.items()}
    fam = {n: family(n, r.get("source")) for n, r in res.items()}

    print("[icl] 1. per dataset: broad - control mean +- SE over draws | broad alone")
    for n, r in res.items():
        m, se = mean_se(
            [x - y for x, y in zip(per_draw(r, MISALIGNED_TRAITS), per_draw(r, CONTROL_TRAITS))]
        )
        print(
            f"[icl]   {n:<30} {r.get('label', ''):<9} {fam[n]:<12} {m:+.2f} +- {se:.2f} | {sc[n]['broad']:+.2f}"
        )

    keys = ["primary_minus_control", "broad_minus_control", "broad", "minus_control"] + [
        f"trait:{t}" for t in MISALIGNED_TRAITS
    ]
    for title, keep in (
        ("emergent vs narrow", ("emergent", "narrow")),
        ("emergent vs narrow+benign", ("emergent", "narrow", "benign")),
    ):
        names = [n for n in res if res[n].get("label") in keep]
        y = [res[n]["label"] == "emergent" for n in names]
        print(f"[icl] 2. AUC {title} ({sum(y)} vs {len(y) - sum(y)}), permutation p")
        for k in keys:
            a, p = auc_p([sc[n][k] for n in names], y)
            print(f"[icl]   {k:<62} {a:.2f}  p={p:.3f}")
        print(f"[icl] 3. leave one family out, broad - control ({title})")
        for f in sorted(set(fam[n] for n in names)):
            sub = [n for n in names if fam[n] != f]
            ys = [res[n]["label"] == "emergent" for n in sub]
            a, p = auc_p([sc[n]["broad_minus_control"] for n in sub], ys)
            print(
                f"[icl]   without {f:<12} ({sum(ys)} vs {len(ys) - sum(ys)}): "
                + ("n/a" if a is None else f"{a:.2f}  p={p:.3f}")
            )

    print("[icl] 4. matched pairs: (emergent) - (other) on broad - control, per draw, mean +- SE")
    wins = 0
    pwins = 0
    tot = 0
    for e, o in PAIRS:
        if e in res and o in res:
            de = [
                x - y
                for x, y in zip(
                    per_draw(res[e], MISALIGNED_TRAITS), per_draw(res[e], CONTROL_TRAITS)
                )
            ]
            do = [
                x - y
                for x, y in zip(
                    per_draw(res[o], MISALIGNED_TRAITS), per_draw(res[o], CONTROL_TRAITS)
                )
            ]
            m, se = mean_se([a - b for a, b in zip(de, do)])
            pe = [
                x - y
                for x, y in zip(per_draw(res[e], PRIMARY_TRAITS), per_draw(res[e], CONTROL_TRAITS))
            ]
            po = [
                x - y
                for x, y in zip(per_draw(res[o], PRIMARY_TRAITS), per_draw(res[o], CONTROL_TRAITS))
            ]
            pm, pse = mean_se([a - b for a, b in zip(pe, po)])
            wins += m > 0
            pwins += pm > 0
            tot += 1
            print(
                f"[icl]   {e:<30} - {o:<24} {m:+.2f} +- {se:.2f}   primary {pm:+.2f} +- {pse:.2f}"
            )
    print(
        f"[icl]   emergent side higher in {wins} of {tot} pairs (primary score: {pwins} of {tot})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
