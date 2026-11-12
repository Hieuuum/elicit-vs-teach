"""Does the before-training score predict how easily a model relearns?  CPU only: reads the EDL sweep points
(relearn.py --sweep manifests, run ids <prefix>-edl-<tag>[-shuf]-n<N>-s<seed>) and the PG reads (pg_pairs.py).

Per model, means over seeds:
  gap_n   held-out loss after n relearned facts minus the same after all of them (n = 8, 32): how far the
          held-out facts still are from the model's own endpoint; smaller = relearns faster
  mdl128  first-pass code length per label token of 128 relearning facts (Donoway's MDL / D)
  kedl    EDL/D on true facts minus on rotated facts at the smallest n (the knowledge-specific cost), if the
          rotated twin exists
Predictors, all read before any training: s (control form, overlap-matched option, within kinds), the raw
pairing margin, and the plain loss of the true answers.

Pre-registered 2026-11-12 (decisions.md), after a preview on five Zephyr models (rho -0.7, n = 5):
  R1  Spearman(s, gap_32) < 0 over the unlearned / safeguarded models and their two originals
  R2  |Spearman| of s with gap_32 exceeds that of the raw margin and of the plain loss
Prints numbers only.

  python3 experiments/unlearning/relearn_predict.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from edl_sweep import load  # noqa: E402

SETS = {"wmdp": ("wmdp", ("orig", "elm", "graddiff", "simnpo", "npo", "rmu", "rmulat")),
        "deepig": ("wmdp-deepig", ("orig", "filtered", "cb", "cb_lat", "filtered_cb_lat"))}
PREDICTORS = ("s", "margin", "loss")
OUTCOMES = ("gap_8", "gap_32", "mdl128", "kedl")


def spearman(x: list[float], y: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    a, b = rank(x), rank(y)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
    return num / den if den else float("nan")


def outcomes(pts: dict, rot: dict) -> dict[str, float]:
    """Means over seeds of the relearning measures from one model's sweep points {seed: {n: result}}."""
    if not pts:
        return {}
    ns = sorted(set.intersection(*(set(p) for p in pts.values())))
    mean = lambda f: sum(f(p) for p in pts.values()) / len(pts)  # noqa: E731
    out = {}
    for n in (8, 32):
        if n in ns and "test_loss_nats" in next(iter(pts.values()))[n]:
            out[f"gap_{n}"] = mean(lambda p, n=n: p[n]["test_loss_nats"] - p[ns[-1]]["test_loss_nats"])
    if 128 in ns:
        out["mdl128"] = mean(lambda p: p[128]["mdl_nats"] / p[128]["label_tokens"])
    seeds = sorted(set(pts) & set(rot))
    if seeds and ns[0] in rot[seeds[0]]:
        k = "edl_ocv_per_token_nats"
        out["kedl"] = sum(pts[s][ns[0]][k] - rot[s][ns[0]][k] for s in seeds) / len(seeds)
    return out


def predictors(pg: Path, st: str, tag: str) -> dict[str, float]:
    out = {}
    f = pg / f"control_overlap_{st}_{tag}.json"
    if f.is_file():
        r = json.loads(f.read_text())["kind"]
        out["s"], out["margin"] = 2 * r["acc"] - 1, r["m21_data_nats_per_token"]
    f = pg / f"{st}_{tag}.json"
    if f.is_file():
        out["loss"] = json.loads(f.read_text())["true_mean_nats"]
    return out


def report(rows: list[dict]) -> list[str]:
    lines = [f"{'set':<7} {'model':<16} " + " ".join(f"{c:>8}" for c in (*PREDICTORS, *OUTCOMES))]
    for r in rows:
        lines.append(f"{r['set']:<7} {r['tag']:<16} " + " ".join(
            f"{r[c]:>8.3f}" if c in r else f"{'-':>8}" for c in (*PREDICTORS, *OUTCOMES)))
    for scope, sub in (("all", rows), *((st, [r for r in rows if r["set"] == st]) for st in SETS)):
        for o in OUTCOMES:
            cells = []
            for p in PREDICTORS:
                pairs = [(r[p], r[o]) for r in sub if p in r and o in r]
                cells.append(f"{p} {spearman(*zip(*pairs)):+.2f} (n={len(pairs)})" if len(pairs) >= 4 else f"{p} -")
            lines.append(f"Spearman with {o:<7} [{scope:<6}]: " + "; ".join(cells))
    both = [r for r in rows if "s" in r and "gap_32" in r]
    if len(both) >= 4:
        base = [p for p in PREDICTORS[1:] if all(p in r for r in both)]
        rho = {p: spearman([r[p] for r in both], [r["gap_32"] for r in both]) for p in ("s", *base)}
        lines.append(f"R1 (s vs gap_32 negative): {'PASS' if rho['s'] < 0 else 'FAIL'} ({rho['s']:+.2f}, n={len(both)})")
        if base:
            ok = rho["s"] < 0 and all(abs(rho["s"]) > abs(rho[p]) for p in base)
            lines.append(f"R2 (s beats {', '.join(base)}): {'PASS' if ok else 'FAIL'} ("
                         + ", ".join(f"{p} {rho[p]:+.2f}" for p in base) + ")")
    return lines


def main() -> int:
    store = Path(os.environ.get("GEODE_STORE", HERE.parents[1] / "geode-store"))
    pg = store / "unlearning" / "wmdp" / "pg_pairs"
    rows = []
    for st, (prefix, tags) in SETS.items():
        for t in tags:
            r = {"set": st, "tag": t, **predictors(pg, st, t), **outcomes(load(store, t, prefix), load(store, f"{t}-shuf", prefix))}
            if len(r) > 2:
                rows.append(r)
    for line in report(rows):
        print(f"[rlp] {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
