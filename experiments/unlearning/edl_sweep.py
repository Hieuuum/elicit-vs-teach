"""EDL sweep summary (M19): Donoway et al.'s own elicit-vs-teach signature on the unlearned models.

Reads the sweep points written by relearn.py --sweep (run ids wmdp-edl-<tag>-n<N>-s<seed>) and, per
model and floor, the curves EDL/D (excess description length per label token) against the number of
training facts n, one per seed. "Bits That Count" §4: elicitation = monotonically DECREASING EDL/D
(each extra fact more redundant with what the weights already hold); teaching = an INCREASING phase
(each extra fact more informative while new structure is built). geode.edl.edl_signature (specs/01
V1.12) makes the call across the seeds: a shape counts only when every seed shows it.

Two floors, always named (decisions.md 2026-08-06/08-11): "ocv" = the kept (restored min-val)
model's own val loss, the owner's default; "test" = the paper's Eq. 3, that model's loss on the
held-out bio_B facts. Also the total EDL a model needs beyond orig at each n (same seed = same
subset and order, so the difference is paired): bounded in n = a fixed unlock cost, growing with n
= information orig did not need.

The seedless wmdp-edl-<tag>-n<N> points (before 2026-09-29) are ignored: their min-val restore was a
no-op whenever step 0 won.

Writes <out>/edl_sweep.json (nats) and <out>/edl_sweep.png (bits, the paper's unit).

Usage: python3 edl_sweep.py --out out/wmdp --tags "orig rmu elm npo simnpo" [--store $GEODE_STORE]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from geode.edl import edl_signature  # noqa: E402

FLOORS = {"ocv": "edl_ocv_per_token_nats", "test": "edl_test_per_token_nats"}
LN2 = math.log(2)


def load(store: Path, tag: str, prefix: str = "wmdp") -> dict[int, dict[int, dict]]:
    """{seed: {n: manifest result}} for one model's sweep points."""
    pts: dict[int, dict[int, dict]] = {}
    for man in (store / "runs").glob(f"{prefix}-edl-{tag}-n*-s*/manifest.json"):
        m = re.fullmatch(rf"{re.escape(prefix)}-edl-{re.escape(tag)}-n(\d+)-s(\d+)", man.parent.name)
        if not m:
            continue
        r = json.loads(man.read_text()).get("result", {})
        if FLOORS["ocv"] in r:
            pts.setdefault(int(m.group(2)), {})[int(m.group(1))] = r
    return pts


def summarize(pts: dict[int, dict[int, dict]]) -> dict:
    seeds = sorted(pts)
    ns = sorted(set.intersection(*(set(pts[s]) for s in seeds)))
    out = {"ns": ns, "seeds": seeds, "floors": {},
           "label_tokens": {str(s): [pts[s][n]["label_tokens"] for n in ns] for s in seeds}}
    for f, key in FLOORS.items():
        curves = [[pts[s][n][key] for n in ns] for s in seeds]
        sig = edl_signature(ns, curves)
        out["floors"][f] = {"call": sig["call"], "agree": sig["agree"], "n_seeds": sig["n_seeds"],
                            "seed_calls": sig["seed_calls"], "rise_at": sig["rise_at"],
                            **{f"{k}_nats": sig[k] for k in ("mean", "max_rise", "tol", "net")},
                            "curves_nats": {str(s): c for s, c in zip(seeds, curves)}}
    return out


def delta_vs_orig(u: dict, o: dict, pu: dict, po: dict) -> dict | None:
    """Seed-paired total EDL of U minus orig's (nats) at each shared n, mean over shared seeds."""
    seeds = sorted(set(pu) & set(po))
    ns = [n for n in u["ns"] if n in o["ns"] and all(n in pu[s] and n in po[s] for s in seeds)]
    if not seeds or not ns:
        return None
    res = {"ns": ns, "seeds": seeds}
    for f, key in FLOORS.items():
        res[f"{f}_nats"] = [sum(pu[s][n][key] * pu[s][n]["label_tokens"] - po[s][n][key] * po[s][n]["label_tokens"]
                                for s in seeds) / len(seeds) for n in ns]
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tags", default="orig rmu elm npo simnpo")
    ap.add_argument("--prefix", default="wmdp", help="run-id prefix of the model set in the store (wmdp, wmdp-tar, wmdp-deepig)")
    ap.add_argument("--store", type=Path, default=Path(os.environ.get("GEODE_STORE", HERE.parents[1] / "geode-store")))
    args = ap.parse_args()
    raw = {t: load(args.store, t, args.prefix) for t in args.tags.split()}
    res = {t: summarize(p) for t, p in raw.items() if p}
    for t in res:
        if t != "orig" and "orig" in res:
            res[t]["delta_vs_orig"] = delta_vs_orig(res[t], res["orig"], raw[t], raw["orig"])
    for t, r in res.items():
        fl = r["floors"]
        calls = "  ".join(f"{f}: {fl[f]['call']} ({fl[f]['agree']}/{fl[f]['n_seeds']} seeds)" for f in FLOORS)
        curve = " ".join(f"n={n}:{y / LN2:+.3f}" for n, y in zip(r["ns"], fl["ocv"]["mean_nats"]))
        d = r.get("delta_vs_orig")
        extra = (f"; beyond orig at n={d['ns'][-1]}: {d['ocv_nats'][-1] / LN2 / 1e3:+.2f} kbit (ocv)"
                 if d else "")
        print(f"[edl] {t:<8} {calls}  EDL/D bits/token (ocv, seed mean) {curve}{extra}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "edl_sweep.json").write_text(json.dumps(res, indent=2))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        colors = {t: ("black" if t == "orig" else f"C{i}") for i, t in enumerate(res)}
        for ax, f in zip(axs[:2], FLOORS):
            for t, r in res.items():
                fl = r["floors"][f]
                for c in fl["curves_nats"].values():
                    ax.plot(r["ns"], [y / LN2 for y in c], color=colors[t], lw=0.6, alpha=0.35)
                ax.plot(r["ns"], [y / LN2 for y in fl["mean_nats"]], marker="o", color=colors[t],
                        lw=2.2 if t == "orig" else 1.4, label=f"{t}: {fl['call']}")
            ax.set_title(f"EDL/D, {f} floor (↓ elicit, ↑ phase teach)", fontsize=9)
            ax.set_ylabel("EDL / D (bits per label token)")
        for t, r in res.items():
            d = r.get("delta_vs_orig")
            if d:
                axs[2].plot(d["ns"], [y / LN2 / 1e3 for y in d["ocv_nats"]], marker="o", color=colors[t], label=t)
        axs[2].set_title("total EDL beyond orig, ocv floor (flat in n = fixed unlock cost)", fontsize=9)
        axs[2].set_ylabel("kbit")
        all_ns = sorted({n for r in res.values() for n in r["ns"]})
        for ax in axs:
            ax.set_xscale("log")
            ax.set_xticks(all_ns, [str(n) for n in all_ns])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_yscale("symlog", linthresh=0.1)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xlabel("training facts n (bio_A)")
            ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(args.out / "edl_sweep.png", dpi=160)
    except ImportError:
        print("[edl] matplotlib missing: no figure")
    print(f"[edl] wrote {args.out / 'edl_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
