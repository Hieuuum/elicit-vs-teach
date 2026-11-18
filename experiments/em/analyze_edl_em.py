"""Read the EM EDL sweeps (edl_sweep_em.py): the EDL/D-vs-n signature per dataset (M19), the
knowledge-specific curve against the rotated twin where one was run (M19k), against the measured EM
rate (judge_em.py). CPU, seconds. Reports bits per response token.

Calls, from geode.edl: edl_signature (decreasing = elicit, an increasing phase = teach) on the OCV and
the test floor; knowledge_read (V1.13: TEACH / GATE / ELICIT / NONE / MIXED) on true - rotated.

Usage: python3 experiments/em/analyze_edl_em.py [--model-dir Qwen2.5-14B-Instruct]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from em_env import em_store  # noqa: E402

from geode.edl import edl_signature, knowledge_read  # noqa: E402

LN2 = math.log(2)


def curves(points: list[dict], key: str) -> tuple[list[int], list[list[float]]]:
    """Sizes present for every seed, and one EDL/D curve (bits/token) per seed."""
    by_seed: dict[int, dict[int, float]] = defaultdict(dict)
    for p in points:
        by_seed[p["seed"]][p["n_train"]] = p[key] / LN2
    common = sorted(set.intersection(*(set(d) for d in by_seed.values()))) if by_seed else []
    return common, [[d[n] for n in common] for d in by_seed.values()]


def load(root: Path) -> dict[str, list[dict]]:
    out = {}
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        pts = [json.loads(f.read_text()) for f in sorted(d.glob("n*-s*.json"))]
        if pts:
            out[d.name] = pts
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-dir", default="Qwen2.5-14B-Instruct")
    args = ap.parse_args()
    store = em_store()
    sweeps = load(store / "edl" / args.model_dir)
    runs = store / "runs" / args.model_dir
    em = {}
    for p in runs.iterdir() if runs.is_dir() else []:
        j = p / "judged.json"
        if j.is_file() and "@" not in p.name:
            em[p.name] = json.loads(j.read_text()).get("em_rate")
    print(f"[edl] {len(sweeps)} sweeps ({args.model_dir}); EDL/D in bits per response token")
    for name, pts in sweeps.items():
        base = name.split("@")[0]
        for key, label in (("edl_ocv_per_token_nats", "ocv"), ("edl_test_per_token_nats", "test")):
            ns, cs = curves(pts, key)
            if len(ns) < 3:
                print(f"[edl]   {name:<28} {label}: {len(ns)} sizes so far")
                continue
            sig = edl_signature(ns, cs)
            vals = " ".join(f"n={n}:{v:+.3f}" for n, v in zip(ns, sig["mean"]))
            e = em.get(base)
            print(
                f"[edl]   {name:<28} EM {'n/a' if e is None else f'{e:.3f}'}  {label}: {sig['call']} "
                f"({sig['agree']}/{sig['n_seeds']} seeds)  {vals}"
            )
    print("[edl] M21 on the held-out test rows along the sweep (nats/token, seed mean)")
    for name, pts in sweeps.items():
        have = [p for p in pts if p.get("m21_test_nats") is not None]
        if not have:
            continue
        ns, cs = curves(have, "m21_test_nats")
        mean = [
            sum(c) / len(c) * LN2 for c in zip(*cs)
        ]  # curves() converts to bits; M21 is read in nats
        print(f"[edl]   {name:<28} " + " ".join(f"n={n}:{v:+.3f}" for n, v in zip(ns, mean)))
    for name in sweeps:
        if "@" in name or f"{name}@rot" not in sweeps:
            continue
        nt, ct = curves(sweeps[name], "edl_ocv_per_token_nats")
        nr, cr = curves(sweeps[f"{name}@rot"], "edl_ocv_per_token_nats")
        ns = [n for n in nt if n in nr]
        if len(ns) < 3:
            continue
        kt = [[c[nt.index(n)] for n in ns] for c in ct]
        kr = [[c[nr.index(n)] for n in ns] for c in cr]
        k = [[a - b for a, b in zip(x, y)] for x, y in zip(kt, kr)]
        read = knowledge_read(ns, k)
        vals = " ".join(f"n={n}:{v:+.3f}" for n, v in zip(ns, [sum(c) / len(c) for c in zip(*k)]))
        print(f"[edl]   M19k {name:<23} true - rotated: {read['read'].upper()}  {vals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
