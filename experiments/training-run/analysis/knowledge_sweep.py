"""The two costs inside EDL/D on the toy task: the rotated-answer twin sweep and M21 along the sweep.

The Deep Ignorance LAT models (decisions.md 2026-10-15) showed that the EDL/D signature cannot tell
"the facts are being taught" from "the data pay for an unlock before the facts are read out": the
rising phase appeared with the facts demonstrably still in the weights.  Two reads separate the
costs, and this script calibrates them where the outcome is known, on the arithmetic Figure-2 arms:

  rotated twin   the same sweep trained on D_algo_bare_shuf (every answer moved to another item,
                 datagen/make_shuf_sets.py; run ids <prefix>-shuf-n<N>): the cost of format and
                 training dynamics with no pairing information.
  knowledge      EDL/D(true, n) - EDL/D(rotated, n) at shared sizes: the knowledge-specific curve.
                 Prediction: the taught arms (TinyStories base / format-installed) keep their hump
                 here, since there the facts really are taught; an elicit arm (the op-bridge parent,
                 Llama) stays decreasing.  A hump that lives in the rotated curve is an unlock.
  M21(n)         the hidden-preference gap on the held-out eval block after n facts
                 (grad_transfer.py --task arith --run-id, results/m21_sweep/<rid>.json; n = 0 from
                 the parent, <prefix>-n0.json): when the pairing information becomes readable.

Reads the run manifests from $GEODE_STORE (experiment.target_result.edl_per_label_token_nats, the
sweep's headline metric; the n=1,000,000 endpoints are pulled with hf_checkpoint.py pull
--no-weights) and geode.edl.edl_signature makes the single-seed call per curve.  Writes
results/knowledge_sweep_<family>.json and analysis/figures/knowledge_sweep_<family>.png.  CPU-only.

Usage: python3 knowledge_sweep.py --family ts|mix|nl3 [--store DIR] [--m21-dir results/m21_sweep]
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
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from geode.edl import edl_signature, half_rise_n, knowledge_read  # noqa: E402

LN2 = math.log(2)
EDL_KEY = "edl_per_label_token_nats"
# family -> {run-id prefix: label}; the known outcome is in the label
FAMILIES: dict[str, dict[str, str]] = {
    "ts": {"evt-ts1b-fig2ts-noinst": "TinyStories-1B base (teach)",
           "evt-ts1b-fig2ts-inst": "TinyStories-1B, format installed (teach)"},
    "mix": {"evt-ts1b-mix-nl": "TinyStories-1B, op-bridge parent (elicit)"},
    "nl3": {"evt-llama-fig2nl3-noinst": "Llama-3.2-1B base (elicit)",
            "evt-llama-fig2nl3-inst": "Llama-3.2-1B, format installed (elicit)"},
}


def sweep_points(store: Path, prefix: str, shuf: bool) -> dict[int, dict]:
    """{n: target_result} of the complete runs <prefix>[-shuf]-n<N>."""
    pat = re.compile(rf"^{re.escape(prefix)}{'-shuf' if shuf else ''}-n(\d+)$")
    out: dict[int, dict] = {}
    for d in (store / "runs").glob(f"{prefix}{'-shuf' if shuf else ''}-n*"):
        m = pat.match(d.name)
        if not m or not (d / "manifest.json").is_file():
            continue
        man = json.loads((d / "manifest.json").read_text())
        tr = man.get("experiment", {}).get("target_result")
        if man.get("status") == "complete" and tr and tr.get(EDL_KEY) is not None:
            out[int(m.group(1))] = tr
    return out


def m21_points(m21_dir: Path, prefix: str, shuf: bool) -> dict[int, dict]:
    """{n: gap block of bioB} from grad_transfer outputs; n = 0 is the parent's (<prefix>-n0.json)."""
    out: dict[int, dict] = {}
    pat = re.compile(rf"^{re.escape(prefix)}{'-shuf' if shuf else ''}-n(\d+)\.json$")
    for f in m21_dir.glob(f"{prefix}{'-shuf' if shuf else ''}-n*.json"):
        m = pat.match(f.name)
        if not m:
            continue
        g = json.loads(f.read_text()).get("gap", {}).get("bioB")
        if g:
            out[int(m.group(1))] = g
    if not shuf:
        p0 = m21_dir / f"{prefix}-n0.json"
        if p0.is_file():
            g = json.loads(p0.read_text()).get("gap", {}).get("bioB")
            if g:
                out[0] = g
    return out


def _mdl(tr: dict) -> float | None:
    """First-epoch prequential code length per label token: EDL/D + the run's floor (its global-min
    val loss on the TRUE stopping block, geode.edl.metrics.edl_epoch1_nats), so the rotated run's
    floor is its loss on true facts too.  The floor-free difference true - rotated says whether the
    true stream itself was cheaper (facts present) or dearer (something pairing-specific acquired)."""
    if tr.get(EDL_KEY) is None or tr.get("min_val_nats") is None:
        return None
    return tr[EDL_KEY] + tr["min_val_nats"]


def _sig(ns: list[int], ys: list[float]) -> dict:
    """The slope signature with rel_tol 0.02: the arithmetic curves start at 6-9 bits, so the WMDP
    default (a tenth of the range) swallows the 1.8-bit teaching hump."""
    if len(ns) < 3:
        return {"call": "too few sizes", "ns": ns, "nats": ys}
    s = edl_signature(ns, [ys], rel_tol=0.02)
    return {"call": s["call"], "rise_at": s["rise_at"], "max_rise_nats": s["max_rise"], "net_nats": s["net"],
            "ns": ns, "nats": ys}


def summarize(store: Path, m21_dir: Path, prefix: str) -> dict:
    t, r = sweep_points(store, prefix, False), sweep_points(store, prefix, True)
    out: dict = {"n_true": len(t), "n_rotated": len(r)}
    if t:
        ns = sorted(t)
        out["true"] = _sig(ns, [t[n][EDL_KEY] for n in ns])
    if r:
        ns = sorted(r)
        out["rotated"] = _sig(ns, [r[n][EDL_KEY] for n in ns])
    shared = sorted(set(t) & set(r))
    if shared:
        out["knowledge"] = _sig(shared, [t[n][EDL_KEY] - r[n][EDL_KEY] for n in shared])
        out["knowledge"]["read"] = knowledge_read(shared, [[t[n][EDL_KEY] - r[n][EDL_KEY] for n in shared]])
        out["knowledge"]["floor_true_nats"] = [t[n].get("min_val_nats") for n in shared]
        out["knowledge"]["floor_rotated_nats"] = [r[n].get("min_val_nats") for n in shared]
        if all(_mdl(t[n]) is not None and _mdl(r[n]) is not None for n in shared):
            out["knowledge"]["mdl_diff_nats"] = [_mdl(t[n]) - _mdl(r[n]) for n in shared]
            out["knowledge"]["mdl_true_nats"] = [_mdl(t[n]) for n in shared]
            out["knowledge"]["mdl_rotated_nats"] = [_mdl(r[n]) for n in shared]
    for key, shuf in (("m21", False), ("m21_rotated", True)):
        pts = m21_points(m21_dir, prefix, shuf)
        if pts:
            ns = sorted(pts)
            out[key] = {"ns": ns, "gap_nats": [pts[n]["gap_nats"] for n in ns],
                        "kind_gap_nats": [pts[n].get("kind", {}).get("gap_nats") for n in ns],
                        "half_rise_n": half_rise_n(ns, [pts[n]["gap_nats"] for n in ns])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--store", type=Path, default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")))
    ap.add_argument("--m21-dir", type=Path, default=HERE.parent / "results" / "m21_sweep")
    ap.add_argument("--out", type=Path, default=None, help="json path (default results/knowledge_sweep_<family>.json)")
    ap.add_argument("--fig", type=Path, default=None, help="png path (default analysis/figures/knowledge_sweep_<family>.png)")
    args = ap.parse_args()
    res = {p: {"label": lab, **summarize(args.store, args.m21_dir, p)} for p, lab in FAMILIES[args.family].items()}
    for p, r in res.items():
        print(f"[ks] {p} ({r['label']}): {r['n_true']} true points, {r['n_rotated']} rotated points")
        for key in ("true", "rotated", "knowledge"):
            if key in r:
                c = r[key]
                print(f"[ks]   {key:<9} {c['call']:<14} EDL/D bits/token "
                      + " ".join(f"n={n}:{y / LN2:+.3f}" for n, y in zip(c["ns"], c["nats"])))
        if "read" in r.get("knowledge", {}):
            k = r["knowledge"]["read"]
            print(f"[ks]   READ {k['read'].upper()}: n<={k['block_ns'][-1]} {k['first_nats'] / LN2:+.2f} bits/token, "
                  f"max n={k['max_n']} {k['max_nats'] / LN2:+.2f}, <= 0 from n={k['crossing_n']}")
        if "mdl_diff_nats" in r.get("knowledge", {}):
            k = r["knowledge"]
            print("[ks]   MDL/D true - rotated (floor-free, bits/token) "
                  + " ".join(f"n={n}:{y / LN2:+.3f}" for n, y in zip(k["ns"], k["mdl_diff_nats"]))
                  + "  | floors true/rotated (nats) "
                  + " ".join(f"{a:.2f}/{b:.2f}" for a, b in zip(k["floor_true_nats"], k["floor_rotated_nats"])))
        for key in ("m21", "m21_rotated"):
            if key in r:
                print(f"[ks]   {key:<12} nats/token on the eval block (half of the rise by n={r[key]['half_rise_n']}) "
                      + " ".join(f"n={n}:{y:+.3f}" for n, y in zip(r[key]["ns"], r[key]["gap_nats"])))
    out = args.out or HERE.parent / "results" / f"knowledge_sweep_{args.family}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    fig_path = args.fig or HERE / "figures" / f"knowledge_sweep_{args.family}.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        all_ns: set[int] = set()
        for i, (p, r) in enumerate(res.items()):
            col = f"C{i}"
            if "true" in r:
                axs[0].plot(r["true"]["ns"], [y / LN2 for y in r["true"]["nats"]], marker="o", color=col,
                            label=f"{r['label']}: {r['true']['call']}")
                all_ns |= set(r["true"]["ns"])
            if "rotated" in r:
                axs[0].plot(r["rotated"]["ns"], [y / LN2 for y in r["rotated"]["nats"]], marker="x", ls="--",
                            color=col, label=f"{r['label']}, rotated: {r['rotated']['call']}")
            if "knowledge" in r:
                axs[1].plot(r["knowledge"]["ns"], [y / LN2 for y in r["knowledge"]["nats"]], marker="o", color=col,
                            label=f"{r['label']}: {r['knowledge']['call']}")
            for key, mk, ls in (("m21", "o", "-"), ("m21_rotated", "x", "--")):
                if key in r:
                    ns = [max(n, 1) for n in r[key]["ns"]]   # n = 0 drawn at the left edge of the log axis
                    axs[2].plot(ns, r[key]["gap_nats"], marker=mk, ls=ls, color=col,
                                label=f"{r['label']}{', after rotated facts' if key == 'm21_rotated' else ''}")
                    all_ns |= set(ns)
        axs[0].set_title("EDL/D: true (solid) and rotated-answer (dashed) sweeps", fontsize=9)
        axs[0].set_ylabel("bits per label token")
        axs[1].set_title("knowledge-specific EDL/D = true − rotated (↑ phase = facts taught)", fontsize=9)
        axs[1].set_ylabel("bits per label token")
        axs[2].set_title("M21 on the held-out block after n facts: L(rotated) − L(true)", fontsize=9)
        axs[2].set_ylabel("nats per label token")
        ticks = sorted(all_ns)
        for ax, symlog in zip(axs, (True, True, False)):
            ax.set_xscale("log")
            if ticks:
                ax.set_xticks(ticks, [("0" if n == 1 and 0 in {m for r in res.values() for m in r.get("m21", {}).get("ns", [])}
                                       else str(n)) for n in ticks], rotation=60, fontsize=7)
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            if symlog:
                ax.set_yscale("symlog", linthresh=0.1)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xlabel("training examples n (prefix of D_algo_bare)")
            ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=160)
        print(f"[ks] wrote {fig_path}")
    except ImportError:
        print("[ks] matplotlib missing: no figure")
    print(f"[ks] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
