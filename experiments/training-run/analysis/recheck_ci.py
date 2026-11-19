"""Re-check every stored control-set call with the jackknife interval (decisions.md 2026-11-19). CPU, no model.

Walks the JSON results under $GEODE_STORE (unlearning/wmdp/pg_pairs, em/pg_pairs, the toy reads) and, for every
stored block {"pairs", "data", "control"} (margins per pair, written by every control read since V1.15),
recomputes acc*, the old item bootstrap interval and the jackknife interval, and prints the calls that change.
Prints numbers only.

  python3 experiments/training-run/analysis/recheck_ci.py [--all]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from geode.edl.pairing import TIE_NATS, bootstrap_ci, jackknife_ci  # noqa: E402


def blocks(obj, path=()):
    """(path, block) for every {"pairs", "data", "control"} dict inside a loaded JSON."""
    if isinstance(obj, dict):
        if {"pairs", "data", "control"} <= set(obj):
            yield path, obj
        for k, v in obj.items():
            yield from blocks(v, (*path, str(k)))


def recheck(block: dict) -> dict:
    pairs = [tuple(p) for p in block["pairs"]]
    w = [1.0 if a - b > TIE_NATS else (0.0 if a - b < -TIE_NATS else 0.5) for a, b in zip(block["data"], block["control"])]
    per: dict[int, list[float]] = {}
    for (i, j), c in zip(pairs, w):
        per.setdefault(i, []).append(c)
        per.setdefault(j, []).append(c)
    old = bootstrap_ci([sum(v) / len(v) for v in per.values()], 316)
    new = jackknife_ci(pairs, w)
    return {"acc": sum(w) / len(w), "old": old, "new": new,
            "old_call": "elicit" if old[0] > 0.5 else "teach", "new_call": "elicit" if new[0] > 0.5 else "teach"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="print every block, not only changed calls")
    args = ap.parse_args()
    store = Path(os.environ.get("GEODE_STORE", REPO / "geode-store"))
    files = sorted(store.glob("unlearning/wmdp/pg_pairs/*.json")) + sorted(store.glob("em/pg_pairs/*.json")) + \
        sorted(store.glob("**/toy_pg*.json"))
    changed = total = 0
    for f in files:
        try:
            obj = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for path, b in blocks(obj):
            r = recheck(b)
            total += 1
            flip = r["old_call"] != r["new_call"]
            changed += flip
            if flip or args.all:
                print(f"[ci] {f.parent.name}/{f.stem} {'/'.join(path)}: acc* {r['acc']:.3f} "
                      f"old [{r['old'][0]:.3f}, {r['old'][1]:.3f}] {r['old_call']} -> jackknife "
                      f"[{r['new'][0]:.3f}, {r['new'][1]:.3f}] {r['new_call']}{'  CHANGED' if flip else ''}")
    print(f"[ci] {total} stored reads re-checked; {changed} calls change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
