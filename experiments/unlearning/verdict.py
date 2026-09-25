"""Seventeen checks: per metric, is each unlearned model PRE-ELICIT or PRE-TEACH? (PLAN.md §4)

Every metric is reduced to one number per model. Parent-only metrics (★) are
read on the parents themselves; child metrics on each parent's relearning
child (relearn-<tag>-forgetA). The two anchors fix the scale:

    s = (value(unlearned) - value(teach anchor)) / (value(elicit anchor) - value(teach anchor))

elicit anchor = the ORIGINAL model (tag orig; for child metrics its own
relearning child, which relearns facts it already knows), teach anchor = the
RETAIN model that never saw the forget authors (tag retain; its child is
taught them). s >= 0.5 -> PRE-ELICIT, s < 0.5 -> PRE-TEACH (pre-registered,
PLAN.md §6). Undetermined when the anchors do not separate by at least the
metric's min_sep (the instrument cannot tell elicit from teach here), MISSING
when an output is absent, NOISE when a circuit map the metric needs fails the
performing guard (mean logit-diff below the tool's threshold).

Usage: python3 verdict.py --out <launcher out dir> --store $GEODE_STORE --tags "orig retain npo ..."
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import role  # noqa: E402

OUT: Path = Path(".")
STORE: Path = Path(".")


class Noise(Exception):
    """A map the metric depends on fails the performing guard."""


# ------------------------------------------------------------------ readers
def _json(name: str) -> dict | None:
    p = OUT / name
    return json.loads(p.read_text()) if p.is_file() else None


def _map(stem: str):
    p = OUT / f"{stem}.parquet"
    m = _json(f"{stem}.json")
    if not p.is_file() or m is None:
        return None, None
    df = pd.read_parquet(p)
    cols = [c for c in ("node_type", "layer", "head", "writer_type", "writer_layer", "writer_head",
                        "reader_type", "reader_layer") if c in df.columns]
    df.index = df[cols].astype(str).agg(":".join, axis=1)
    return df, m


def _top(df, k: int, heads_only: bool = False) -> set:
    if heads_only:
        df = df[df["node_type"] == "attn"]
    return set(df["abs_score"].nlargest(k).index)


def jac(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def overlap(sa: str, sb: str, k: int, heads_only: bool, need_performing=True) -> float | None:
    da, ma = _map(sa)
    db, mb = _map(sb)
    if da is None or db is None:
        return None
    if need_performing and not (ma.get("performing_regime") and mb.get("performing_regime")):
        raise Noise(f"{sa if not ma.get('performing_regime') else sb} not performing")
    k = min(k, len(da) - 1)
    return jac(_top(da, k, heads_only), _top(db, k, heads_only))


def ceiling(stem: str, k: int, heads_only: bool) -> float | None:
    return overlap(f"{stem}_a", f"{stem}_b", k, heads_only)


def normalised_overlap(sa: str, sb: str, k: int, heads_only: bool) -> float | None:
    j = overlap(sa, sb, k, heads_only)
    ca, cb = ceiling(sa, k, heads_only), ceiling(sb, k, heads_only)
    if j is None or ca is None or cb is None:
        return None
    return j / max(1e-9, math.sqrt(max(ca, 1e-9) * max(cb, 1e-9)))


def prefit(tag: str, block: str) -> dict | None:
    d = _json(f"prefit_{tag}.json")
    return d.get(block) if d else None


def run_dir(parent: str, kind: str = "forgetA") -> Path:
    return STORE / "runs" / f"relearn-{parent}-{kind}"


def last_jsonl(path: Path) -> dict | None:
    if not path.is_file():
        return None
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else None


# ------------------------------------------------------------------ metrics
# each: f(parent_tag) -> (value, note) or None; child metrics read the parent's child
def m1(p):
    c = f"{p}-rl"
    v = normalised_overlap(f"circ_{c}_fA", "circ_orig_fA", 32, heads_only=True)
    if v is None:
        return None
    f_own, f_orig = _json(f"faith_{c}_own.json"), _json(f"faith_orig_in_{c}.json")
    note = ""
    if f_own and f_orig:
        a = pd.read_parquet(OUT / f"faith_{c}_own.parquet").iloc[-1]
        b = pd.read_parquet(OUT / f"faith_orig_in_{c}.parquet").iloc[-1]
        note = f"heads nec. own {a.fraction:.2f} / orig's {b.fraction:.2f} (rand {b.get('random_mean', float('nan')):.2f})"
    return v, note


def m2(p):
    c = f"{p}-rl"
    return _wrap(normalised_overlap(f"edge_{c}_fA", "edge_orig_fA", 256, heads_only=False), "edges J@256 / ceiling")


def m3(p):
    a, b = _json(f"dcm_{p}-rl_fA.json"), _json("dcm_orig_fA.json")
    if not a or not b:
        return None
    ra, rb = a["roles"].get("subject"), b["roles"].get("subject")
    if not ra or not rb:
        return 0.0, "no subject role set"
    return jac(set(ra["nodes"]), set(rb["nodes"])), f"|child|={len(ra['nodes'])} |orig|={len(rb['nodes'])}"


def m4(p):
    c = f"{p}-rl"
    snaps = sorted(OUT.glob(f"circ_{c}_snap*.parquet"), key=lambda q: int(q.stem.split("snap")[1]))
    if not snaps:
        return None
    first = snaps[0].stem
    j = overlap(first, f"circ_{c}_fA", 32, heads_only=False, need_performing=False)
    ceil = ceiling(f"circ_{c}_fA", 32, heads_only=False)
    if j is None or ceil is None:
        return None
    return j / max(ceil, 1e-9), f"first snapshot {first.split('snap')[1]}: J@32 {j:.2f} vs final (ceiling {ceil:.2f})"


def m5(p):
    g = _json(f"grad_{p}-rl.json")
    if not g:
        return None
    r = g[0]["decay_ratio_first_over_last"]
    return math.log10(max(r, 1e-12)), f"first1%/last10% = {r:.2f} (>1 fades); mass {g[0]['cum_grad_mass']:.0f}"


def m6(p):
    rec = last_jsonl(run_dir(p) / "train_log.jsonl")
    if not rec:
        return None
    return rec["rel_travel"], f"||dW||/||W|| over all weights, {rec['step']} steps"


def m7(p):
    r = _json(f"resid_{p}_to_{p}-rl.json")
    if not r:
        return None
    s = r["summary"]
    return s["pc1_task_last"], f"answer removed {s.get('pc1_task_last_nocontent', float('nan')):.2f}"


def m8(p):
    x = _json(f"xfmt_{p}.json")
    if not x:
        return None
    mlp = [v["index"] for k, v in x["nodes"].items() if k.startswith("mlp")]
    geo = prefit(f"{p}_forget", "geometry")
    note = f"state PC1 {geo['pc1_last']:.2f}" if geo else ""
    return sum(mlp) / max(1, len(mlp)), note


def m9(p):
    lens = _json(f"lens_{p}.json")
    if not lens:
        return None
    j = lens["positions"]["-1"].get("jlens") or lens["positions"]["-1"]["logit"]
    ld = j["mean_logit_diff"]
    after = _json(f"lens_{p}-rl.json")
    note = f"read-out ld {ld[-1]:+.2f}, median rank {j['median_rank'][-1]}"
    if after:
        a = after["positions"]["-1"].get("jlens") or after["positions"]["-1"]["logit"]
        note += f"; child settles L{a['settled_depth_median']}"
    return max(ld), note


def m10(p):
    s = _json(f"steer_{p}-rl_into_{p}_heads.json")
    star = _json(f"steer_{p}_from_orig.json")
    note = ""
    if star:
        r = star["results"]
        note = (f"★ orig heads into it: top-1 {r['base_unpatched']['top1']:.2f}->"
                f"{r[[k for k in r if k.startswith('per_prompt_top')][0]]['top1']:.2f}")
    if not s:
        return None
    r = s["results"]
    key = [k for k in r if k.startswith("per_prompt_top")][0]
    return r[key]["top1"], (f"unpatched {r['base_unpatched']['top1']:.2f}, random "
                            f"{r.get([k for k in r if k.startswith('per_prompt_random')][0], {}).get('top1', float('nan')):.2f}; "
                            + note)


def m10_star(p):
    """Anchors for the ★ variant: orig = the donor's own top-1."""
    if p == "orig":
        any_star = sorted(OUT.glob("steer_*_from_orig.json"))
        if not any_star:
            return None
        r = json.loads(any_star[0].read_text())["results"]
        return r["donor"]["top1"], "donor itself"
    s = _json(f"steer_{p}_from_orig.json")
    if not s:
        return None
    r = s["results"]
    key = [k for k in r if k.startswith("per_prompt_top")][0]
    return r[key]["top1"], f"unpatched {r['base_unpatched']['top1']:.2f}"


def m11(p):
    d, meta = _map(f"circ_{p}_forget")
    if d is None:
        return None
    if not meta.get("performing_regime"):
        return 0.0, f"map is noise (ld {meta['mean_logit_diff']:+.2f})"
    c = ceiling(f"circ_{p}_forget", 32, heads_only=False)
    ch = ceiling(f"circ_{p}_forget", 32, heads_only=True)
    return c, f"ld {meta['mean_logit_diff']:+.2f}; heads-only {ch:.2f}"


def m12(p):
    pr, nu, re = prefit(f"{p}_forget", "pref"), prefit(f"{p}_null", "pref"), prefit(f"{p}_retain", "pref")
    if not pr:
        return None
    note = f"top-1 {pr['top1_acc']:.2f}"
    if nu:
        note += f"; null {nu['logit_diff_mean']:+.2f}"
    if re:
        note += f"; retain {re['logit_diff_mean']:+.2f}"
    return pr["logit_diff_mean"], note


def m13(p):
    h = prefit(f"{p}_forget", "hessian")
    if not h:
        return None
    lo, hi = h["lambda_min"], h["lambda_max"]
    share = min(1.0, max(0.0, -lo / (hi - lo))) if hi > lo else float("nan")
    return share, f"lambda {lo:.3g} .. {hi:.3g}"


def m14(p):
    d = prefit(f"{p}_forget", "das")
    if not d or not d.get("layers"):
        return None
    best = max(((v["full"]["flip_frac"] - v["none"]["flip_frac"]), L) for L, v in d["layers"].items())
    return best[0], f"best layer {best[1]}"


def m15(p):
    d = prefit(f"{p}_forget", "dcm")
    if not d:
        return None
    r = d["roles"].get("subject", {})
    if "n_heads" not in r:
        return 0.0, r.get("skipped", "no role")
    v = r["ld_flip_frac"] if r["n_heads"] > 0 else 0.0
    return v, f"{r['n_heads']} heads; moved {r['ld_flip_frac']:.2f} (all heads {r['ld_flip_ceiling']:.2f})"


def m16(p):
    pr = prefit(f"{p}_forget", "probe")
    if not pr or "subject_ld_best" not in pr:
        return None
    return pr["subject_ld_best"], f"layer {pr['subject_ld_best_layer']}, win {pr['subject_win_best']:.2f}"


def m17(p):
    b = prefit(f"{p}-rl_forgetB", "pref")
    if not b:
        return None
    man = run_dir(p) / "manifest.json"
    note = ""
    if man.is_file():
        res = json.loads(man.read_text()).get("result", {})
        if res.get("first_pass_mean_loss_nats") is not None:
            note = f"prequential first pass {res['first_pass_mean_loss_nats']:.3f} nats/tok, {res.get('final_step')} steps"
    hman = run_dir(p, "holdoutA") / "manifest.json"   # the parent's own teach cost (--holdout)
    if hman.is_file():
        hres = json.loads(hman.read_text()).get("result", {})
        if hres.get("first_pass_mean_loss_nats") is not None:
            note += f"; holdout (teach-in-this-model) first pass {hres['first_pass_mean_loss_nats']:.3f}"
    first = None
    ev = run_dir(p) / "eval_log.jsonl"
    if ev.is_file():
        first = json.loads(ev.read_text().splitlines()[0]).get("probe", {}).get("forget_B")
    if first:
        note = f"B ld {first['logit_diff']:+.2f}->{b['logit_diff_mean']:+.2f}; " + note
    return b["logit_diff_mean"], note


def _wrap(v, note):
    return None if v is None else (v, note)


METRICS = [  # id, name, star, fn, min_sep
    ("M1", "Circuit overlap (heads, child vs orig)", False, m1, 0.05),
    ("M2", "Wiring (edges, child vs orig)", False, m2, 0.05),
    ("M3", "Head roles (subject DCM)", False, m3, 0.05),
    ("M4", "Circuit formation time", False, m4, 0.05),
    ("M5", "Gradient pressure (log10 decay)", False, m5, 0.1),
    ("M6", "Weight write ||dW||/||W||", False, m6, 1e-4),
    ("M7", "State change (shared-direction)", False, m7, 0.05),
    ("M8", "Latent reach (paraphrase bridging)", True, m8, 0.005),
    ("M9", "Answer depth (J-lens max ld)", True, m9, 0.5),
    ("M10", "Switch-on by patching (heads)", False, m10, 0.05),
    ("M10*", "  variant: orig's heads into parent", True, m10_star, 0.05),
    ("M11", "Repeatable task circuit", True, m11, 0.05),
    ("M12", "Hidden preference (nats)", True, m12, 0.3),
    ("M13", "Accelerating descent (neg. share)", True, m13, 0.1),
    ("M14", "Answer carried by the state (DAS)", True, m14, 0.05),
    ("M15", "Subject-reading heads (DCM)", True, m15, 0.05),
    ("M16", "Fact present at the subject", True, m16, 0.3),
    ("M17", "Relearning: held-out recovery", False, m17, 0.3),
]


def evaluate(tags: list[str]) -> dict:
    anchors = {"E": "orig", "T": "retain"}
    tested = [t for t in tags if role(t) == "unlearned"]
    rows = []
    for mid, name, star, fn, min_sep in METRICS:
        row = {"id": mid, "name": name, "star": star, "values": {}, "notes": {}, "verdict": {}, "s": {}}
        for t in [anchors["E"], anchors["T"]] + tested:
            try:
                got = fn(t)
            except Noise as e:
                row["values"][t], row["notes"][t] = None, f"NOISE: {e}"
                continue
            except (KeyError, IndexError, ValueError, ZeroDivisionError) as e:
                row["values"][t], row["notes"][t] = None, f"ERROR {type(e).__name__}: {e}"
                continue
            row["values"][t], row["notes"][t] = (got if got is not None else (None, "missing"))
        e, tv = row["values"].get(anchors["E"]), row["values"].get(anchors["T"])
        for t in tested:
            v = row["values"].get(t)
            if v is None:
                note = row["notes"].get(t, "")
                row["verdict"][t] = "NOISE" if note.startswith("NOISE") else "MISSING"
            elif e is None or tv is None:
                row["verdict"][t] = "NO ANCHOR"
            elif abs(e - tv) < min_sep or any(math.isnan(x) for x in (e, tv, v)):
                row["verdict"][t] = "UNDETERMINED"
            else:
                s = (v - tv) / (e - tv)
                row["s"][t] = s
                row["verdict"][t] = "PRE-ELICIT" if s >= 0.5 else "PRE-TEACH"
        rows.append(row)
    return {"anchors": anchors, "tested": tested, "rows": rows}


def fmt(v) -> str:
    if v is None:
        return "--"
    return f"{v:.3g}" if abs(v) < 1e4 else f"{v:.2e}"


def main() -> int:
    global OUT, STORE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--tags", required=True)
    args = ap.parse_args()
    OUT, STORE = args.out, args.store
    res = evaluate(args.tags.split())
    tested = res["tested"]
    print("[verdict] s = (unlearned - retain) / (orig - retain); s >= 0.5 PRE-ELICIT, < 0.5 PRE-TEACH; "
          "★ = parent-only (no training)")
    head = f"{'metric':<44}{'orig(E)':>9}{'retain(T)':>10}" + "".join(f"{t:>26}" for t in tested)
    print("[verdict] " + head)
    tally = {t: {"PRE-ELICIT": 0, "PRE-TEACH": 0, "star_E": 0, "star_T": 0} for t in tested}
    for r in res["rows"]:
        label = f"{r['id']:<5}{'★' if r['star'] else ' '} {r['name']}"
        cells = ""
        for t in tested:
            v, verd = r["values"].get(t), r["verdict"][t]
            s = r["s"].get(t)
            cells += f"{fmt(v):>8} {('s=' + format(s, '.2f')) if s is not None else '':>7} {verd:>10}"
            if verd in ("PRE-ELICIT", "PRE-TEACH") and r["id"] != "M10*":
                tally[t][verd] += 1
                if r["star"]:
                    tally[t]["star_E" if verd == "PRE-ELICIT" else "star_T"] += 1
        print(f"[verdict] {label:<44}{fmt(r['values'].get('orig')):>9}{fmt(r['values'].get('retain')):>10}{cells}")
    print("[verdict] notes:")
    for r in res["rows"]:
        for t, n in r["notes"].items():
            if n:
                print(f"[verdict]   {r['id']:<5} {t:<10} {n}")
    for t in tested:
        k = tally[t]
        lean = ("LATENT (pre-elicit)" if k["PRE-ELICIT"] > k["PRE-TEACH"] else
                "GONE (pre-teach)" if k["PRE-TEACH"] > k["PRE-ELICIT"] else "SPLIT")
        print(f"[verdict] {t}: {k['PRE-ELICIT']} pre-elicit / {k['PRE-TEACH']} pre-teach of 17 checks "
              f"(★ parent-only: {k['star_E']} / {k['star_T']}) -> knowledge {lean}")
    (OUT / "verdict.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"[verdict] wrote {OUT / 'verdict.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
