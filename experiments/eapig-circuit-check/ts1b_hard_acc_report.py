"""Report for the ts1b hard-set accuracy-stop pair (HANDOFF-ts1b-hard-acc.md, step 6).

Reads what run_box_hard_acc.sh writes under --root and prints elicit vs teach per metric:

  scores/<label>/<tag>/{scores.pt, sanity.json[, repro.json]}   label in SETS
  overlap/overlap_boot.json                                     overlap_boot.py, with `boot`
  functional/{elicit_child,teach_child}.json                    functional_reuse.py
  cmap/{elicit,teach}.json                                      cmap.py
  pg/<read>.json                                                ts1b_pg.py (run_box_pg.sh)

Edge overlap: raw Jaccard [basic bootstrap 95% CI] and Jaccard / sqrt(ceiling_a x ceiling_b)
(split-half ceilings from each set's mean_a / mean_b, treated as constants). Each route is
set beside its blank-base control (base -> the same child); the margin route - control is
differenced draw by draw over the shared bootstrap draws, with the same basic interval, and
the route counts as reuse only if the margin's interval is above 0.

Pairing gap (PG, HANDOFF-ts1b-pg.md): per read acc* [95% CI], s, call against the
pre-registered call, and the raw data / control margins per answer token; each read beside
its blank-base floor (acc* difference, a point only). If base_symbol reads elicit, the
symbol-form read is flagged as confounded by surface cues (pre-registration). Beside the
pre-registered call (control_read's item bootstrap, ~1.7x too narrow under a null) each read
also carries the calibrated secondary jackknife CI / call (ts1b_pg.jackknife_read); the
confound flag is reported for both calls.

(Hidden preference is not reported: dropped by the owner 2026-11-12.)

    python3 ts1b_hard_acc_report.py --root results/ts1b_hard_acc \
        --out-json results/ts1b_hard_acc/report.json --out-md results/ts1b_hard_acc/report.md

Missing inputs are listed (JSON `missing`, a MISSING line in the markdown) and the exit
code is 1; whatever is present is still reported.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from ceiling_report import ceilings  # noqa: E402

# label -> (run.py tag, task)
SETS = {
    "base": ("base", "word"),
    "ep_word": ("elicit_parent", "word"),
    "ep_sym": ("elicit_parent", "symbol"),
    "fp": ("fmt_parent", "word"),
    "ec": ("elicit_child", "word"),
    "tc": ("teach_child", "word"),
}
CHILDREN = ("ec", "tc")
# (name, route pair, blank-base control pair) — pair keys exactly as overlap_boot writes them
ROUTES = (
    ("elicit, parent symbol circuit", "ep_sym:ec", "base:ec"),
    ("elicit, parent word circuit", "ep_word:ec", "base:ec"),
    ("teach, format parent", "fp:tc", "base:tc"),
)
OTHER_PAIRS = ("ec:tc",)
FUNC = {"elicit_child": ("ep_sym", "ep_word"), "teach_child": ("fp_word",)}
CMAP_EXTRA = {"elicit": ("ep_sym",), "teach": ()}
EM_MIN = 0.90
# PG read -> (model, form, role, pre-registered call or None, blank-base floor read or None)
PG_READS = {
    "ep_symbol": ("elicit parent", "symbol", "main elicit read", "elicit", "base_symbol"),
    "ep_word": ("elicit parent", "word", "reported only", None, "base_word"),
    "fp_word": ("format parent", "word", "main teach read", "teach", "base_word"),
    "base_word": ("base", "word", "floor for fp_word", "teach", None),
    "base_symbol": ("base", "symbol", "same-form floor for ep_symbol", "teach", None),
}
PG_KEYS = ("n_pairs", "acc", "acc_ci", "s", "call", "m21_data_nats_per_pair",
           "m21_control_nats_per_pair", "m21_data_nats_per_token", "m21_control_nats_per_token",
           "m21_excess_nats_per_token", "acc_se_jackknife", "acc_ci_jackknife",
           "call_jackknife")  # fmt: skip
PG_CALIBRATION_NOTE = (
    "Pre-registered call uses control_read's item bootstrap, which is ~1.7x too narrow under a "
    "null (13% false elicit at nominal 2.5%; jackknife 1%). Jackknife = calibrated secondary."
)


# ----------------------------------------------------------------------------- math


def basic_ci(point: float, draws: np.ndarray) -> tuple[float, float]:
    """Basic (reflected) bootstrap 95% interval, as overlap_boot.py: [2p - q97.5, 2p - q2.5]."""
    q_lo, q_hi = np.percentile(np.asarray(draws, dtype=np.float64), [2.5, 97.5])
    return float(2 * point - q_hi), float(2 * point - q_lo)


def margin(route: dict, ctrl: dict) -> dict:
    """route - control for one size, differenced draw by draw (draw i = the same resample in
    both cells). Raises if the draw counts differ."""
    br, bc = np.asarray(route["boot"], np.float64), np.asarray(ctrl["boot"], np.float64)
    if br.shape != bc.shape or br.size == 0:
        raise ValueError(f"boot draws differ or empty: {br.shape} vs {bc.shape}")
    p = float(route["point"] - ctrl["point"])
    lo, hi = basic_ci(p, br - bc)
    lo, hi = max(-1.0, lo), min(1.0, hi)  # a Jaccard difference lies in [-1, 1]
    return {"point": p, "ci_lo": lo, "ci_hi": hi, "reuse": bool(lo > 0)}


def normalized(cell: dict, ceil_a: float, ceil_b: float) -> dict:
    """Jaccard / sqrt(ceil_a * ceil_b) for the point and both CI ends (NaN if a ceiling <= 0)."""
    s = math.sqrt(ceil_a * ceil_b) if ceil_a > 0 and ceil_b > 0 else float("nan")
    return {k: cell[k] / s for k in ("point", "ci_lo", "ci_hi")} | {"denom": s}


# ----------------------------------------------------------------------------- load


class Inputs:
    """Reads files under root, recording every missing one instead of raising."""

    def __init__(self, root: Path):
        self.root = root
        self.missing: list[str] = []

    def path(self, rel: str) -> Path | None:
        p = self.root / rel
        if p.is_file():
            return p
        self.missing.append(rel)
        return None

    def json(self, rel: str) -> dict | None:
        p = self.path(rel)
        return None if p is None else json.loads(p.read_text())


def sets_section(inp: Inputs, fracs: list[float]) -> dict:
    out = {}
    for label, (tag, task) in SETS.items():
        d = f"scores/{label}/{tag}"
        row: dict = {"tag": tag, "task": task}
        san = inp.json(f"{d}/sanity.json")
        if san is not None:
            row["exact_match"] = san["exact_match"]
        if task == "word":
            rep = inp.json(f"{d}/repro.json")
            if rep is not None:
                row["repro_max_abs_diff"], row["repro_pass"] = rep["max_abs_diff"], rep["pass"]
        sp = inp.path(f"{d}/scores.pt")
        if sp is not None:
            sc = torch.load(sp, map_location="cpu")
            row["ceiling"] = ceilings(sc["mean_a"], sc["mean_b"], fracs)
        out[label] = row
    return out


def gates(sets: dict) -> dict:
    g = {}
    for c in CHILDREN:
        em = sets[c].get("exact_match")
        g[f"{c}_em_ge_{EM_MIN}"] = None if em is None else bool(em >= EM_MIN)
    for label, row in sets.items():
        if row["task"] == "word":
            g[f"{label}_repro"] = row.get("repro_pass")
    g["all_pass"] = all(v is True for v in g.values())
    return g


def overlap_section(inp: Inputs, sets: dict) -> dict | None:
    ob = inp.json("overlap/overlap_boot.json")
    if ob is None:
        return None
    fracs = [str(f) for f in ob["meta"]["fracs"]]

    def ceil(label: str, f: str) -> float:
        c = sets.get(label, {}).get("ceiling")
        return c[f] if c and f in c else float("nan")

    pairs = {}
    for key, cells in ob["pairs"].items():
        a, b = key.split(":")
        pairs[key] = {
            f: {k: cells[f][k] for k in ("k", "point", "ci_lo", "ci_hi")}
            | {"normalized": normalized(cells[f], ceil(a, f), ceil(b, f))}
            for f in fracs
        }
    routes = {}
    for name, rkey, ckey in ROUTES:
        if rkey not in ob["pairs"] or ckey not in ob["pairs"]:
            inp.missing.append(f"overlap pair {rkey if rkey not in ob['pairs'] else ckey}")
            continue
        routes[name] = {
            "route": rkey,
            "control": ckey,
            "sizes": {f: margin(ob["pairs"][rkey][f], ob["pairs"][ckey][f]) for f in fracs},
        }
    return {"fracs": fracs, "n_boot": ob["meta"]["n_boot"], "pairs": pairs, "routes": routes}


def functional_section(inp: Inputs) -> dict:
    out = {}
    for child, labs in FUNC.items():
        r = inp.json(f"functional/{child}.json")
        if r is None:
            continue
        out[child] = {
            f: {
                "k": s["k"],
                "f_own": s["f"]["own"],
                "f": {lab: s["f"][lab] for lab in labs},
                "share_of_own": {lab: s["share_of_own_f"][lab] for lab in labs},
                "overlap_with_own": {lab: s["overlap_with_own"][lab] for lab in labs},
                "random_band": s["f_random_band"],
                "above_random_p95": {
                    lab: bool(s["f"][lab] > s["f_random_band"][2]) for lab in labs
                },
            }
            for f, s in r["sizes"].items()
        }
    return out


def cmap_section(inp: Inputs) -> dict:
    out = {}
    for route, extras in CMAP_EXTRA.items():
        r = inp.json(f"cmap/{route}.json")
        if r is None:
            continue
        sizes = {}
        for f, s in r["sizes"].items():
            m = s["masks"]
            row = {
                "k": s["k"],
                "child_nologits": {
                    k: m["child_nologits"][k] for k in ("recovery", "recovery_ci95", "acc2")
                },
                "child": m["child"]["recovery"],
                "typed_disjoint_band": s["random_typed_disjoint"]["recovery_band"],
                "typed_band": s["random_typed"]["recovery_band"],
                "random_nologits_band": s["random_nologits"]["recovery_band"],
            }
            row["above_disjoint_p95"] = bool(
                m["child_nologits"]["recovery_ci95"][0] > row["typed_disjoint_band"][2]
            )
            for lab in extras:
                row[f"{lab}_nologits"] = {
                    k: m[f"{lab}_nologits"][k] for k in ("recovery", "recovery_ci95", "acc2")
                }
            sizes[f] = row
        out[route] = {
            "baseline": r["baseline"],
            "all_recovery": r["all"]["recovery"],
            "checks_pass": r["checks_pass"],
            "sizes": sizes,
        }
    return out


def pg_section(inp: Inputs) -> dict:
    """The PG reads present under pg/, the floor gaps, the symbol-form confound flag and
    whether every read used the same control answers."""
    reads = {}
    for name, (model, form, role, prereg, floor) in PG_READS.items():
        r = inp.json(f"pg/{name}.json")
        if r is None:
            continue
        reads[name] = {"model": model, "form": form, "role": role, "floor": floor}
        reads[name] |= {k: r[k] for k in PG_KEYS}
        reads[name] |= {
            "prereg": prereg,
            "matches_prereg": None if prereg is None else r["call"] == prereg,
            "matches_prereg_jackknife": None if prereg is None else r["call_jackknife"] == prereg,
            "control_check_pass": r["control_check"]["pass"],
            "control_sha256": r["control_check"]["sha256"],
        }
    for row in reads.values():
        f = reads.get(row["floor"]) if row["floor"] else None
        row["acc_minus_floor"] = None if f is None else row["acc"] - f["acc"]
    shas = {r.pop("control_sha256") for r in reads.values()}
    bs = reads.get("base_symbol")
    return {
        "reads": reads,
        "control_shared": None if not reads else len(shas) == 1,
        "symbol_confounded": None if bs is None else bs["call"] == "elicit",
        "symbol_confounded_jackknife": None if bs is None else bs["call_jackknife"] == "elicit",
    }


def build(root: Path, fracs: list[float]) -> dict:
    inp = Inputs(root)
    sets = sets_section(inp, fracs)
    rep = {
        "sets": sets,
        "gates": gates(sets),
        "overlap": overlap_section(inp, sets),
        "functional": functional_section(inp),
        "cmap": cmap_section(inp),
        "pg": pg_section(inp),
    }
    rep["missing"] = sorted(set(inp.missing))
    return rep


# ----------------------------------------------------------------------------- markdown


def _pct(f: str) -> str:
    return f"{100 * float(f):g}%"


def _ci(d: dict, digits: int = 3) -> str:
    return f"{d['point']:.{digits}f} [{d['ci_lo']:.{digits}f}, {d['ci_hi']:.{digits}f}]"


def markdown(rep: dict) -> str:
    L: list[str] = ["# ts1b hard-set accuracy-stop pair: circuit metrics", ""]
    if rep["missing"]:
        L += ["**MISSING:** " + ", ".join(rep["missing"]), ""]
    g = rep["gates"]
    L += [f"Gates: {'all pass' if g['all_pass'] else 'NOT all pass'} — "
          + ", ".join(f"{k} {v}" for k, v in g.items() if k != "all_pass"), ""]  # fmt: skip

    sets = rep["sets"]
    fr = next((list(r["ceiling"]) for r in sets.values() if "ceiling" in r), [])
    L += ["## Score sets (EM on the 1,024 validation problems; split-half ceilings)", "",
          "| set | task | EM | repro diff | " + " | ".join(_pct(f) for f in fr) + " |",
          "|" + "---|" * (len(fr) + 4)]  # fmt: skip
    for lab, r in sets.items():
        em = f"{r['exact_match']:.3f}" if "exact_match" in r else "—"
        rd = f"{r['repro_max_abs_diff']:.1e}" if "repro_max_abs_diff" in r else "—"
        cs = " | ".join(f"{r['ceiling'][f]:.3f}" for f in fr) if "ceiling" in r else ""
        L.append(f"| {lab} | {r['task']} | {em} | {rd} | {cs} |")
    L.append("")

    ov = rep["overlap"]
    if ov:
        L += [f"## Edge overlap (Jaccard [95% CI], {ov['n_boot']} shared bootstrap draws)", "",
              "Raw, then / sqrt(ceiling_a x ceiling_b).", "",
              "| pair | " + " | ".join(_pct(f) for f in ov["fracs"]) + " |",
              "|" + "---|" * (len(ov["fracs"]) + 1)]  # fmt: skip
        for key, cells in ov["pairs"].items():
            L.append(f"| {key} | " + " | ".join(_ci(cells[f]) for f in ov["fracs"]) + " |")
            L.append(f"| {key} norm | "
                     + " | ".join(_ci(cells[f]["normalized"]) for f in ov["fracs"]) + " |")  # fmt: skip
        L += ["", "### Route margin over the blank-base control (route - base->child)", "",
              "| route | " + " | ".join(_pct(f) for f in ov["fracs"]) + " |",
              "|" + "---|" * (len(ov["fracs"]) + 1)]  # fmt: skip
        for name, r in ov["routes"].items():
            cells = [_ci(r["sizes"][f]) + (" **reuse**" if r["sizes"][f]["reuse"] else "")
                     for f in ov["fracs"]]  # fmt: skip
            L.append(f"| {name} ({r['route']} - {r['control']}) | " + " | ".join(cells) + " |")
        L.append("")

    if rep["functional"]:
        L += [
            "## Functional reuse (child faithfulness keeping only a circuit; val problems)",
            "",
            "| child | size | own | parent circuit(s) | random p5/p50/p95 |",
            "|---|---|---|---|---|",
        ]
        for child, sizes in rep["functional"].items():
            for f, s in sizes.items():
                par = ", ".join(f"{lab} {v:.3f} ({s['share_of_own'][lab]:.2f} of own"
                                f"{', > p95' if s['above_random_p95'][lab] else ''})"
                                for lab, v in s["f"].items())  # fmt: skip
                rb = "/".join(f"{x:.3f}" for x in s["random_band"])
                L.append(f"| {child} | {_pct(f)} | {s['f_own']:.3f} | {par} | {rb} |")
        L.append("")

    if rep["cmap"]:
        L += ["## CMAP (child -> parent edge patching; recovery, 0 = parent, 1 = child)", ""]
        for route, r in rep["cmap"].items():
            b = r["baseline"]
            L += [f"**{route}**: KL(child||parent) {b['kl_child_parent_mean']:.3f} nats, all-patched "
                  f"recovery {r['all_recovery']:.3f}, checks {'pass' if r['checks_pass'] else 'FAIL'}", "",
                  "| size | child (no logits) [CI] | extra | typed-disjoint p50/p95 | typed p95 |",
                  "|---|---|---|---|---|"]  # fmt: skip
            for f, s in r["sizes"].items():
                c = s["child_nologits"]
                ex = ", ".join(f"{k[:-9]} {v['recovery']:.3f}" for k, v in s.items()
                               if k.endswith("_nologits") and k != "child_nologits") or "—"  # fmt: skip
                L.append(f"| {_pct(f)} | {c['recovery']:.3f} [{c['recovery_ci95'][0]:.3f}, "
                         f"{c['recovery_ci95'][1]:.3f}]{' > null' if s['above_disjoint_p95'] else ''}"
                         f" | {ex} | {s['typed_disjoint_band'][1]:.3f}/{s['typed_disjoint_band'][2]:.3f}"
                         f" | {s['typed_band'][2]:.3f} |")  # fmt: skip
            L.append("")

    pg = rep["pg"]
    if pg["reads"]:
        L += ["## Pairing gap (PG: link gap on the parents, data vs control margins; val problems)",
              "",
              "acc* = share of pairs whose data margin beats the control margin (chance 1/2); "
              "call elicit iff the CI's low end > 1/2. Margins m21 in nats per answer token.", "",
              "| read | model | form | role | acc* [95% CI] | s | call | pre-reg "
              "| jackknife CI / call | acc* - floor | m21 data | m21 control | excess | pairs "
              "| control |",
              "|" + "---|" * 15]  # fmt: skip
        for name, r in pg["reads"].items():
            pre = (
                "—"
                if r["prereg"] is None
                else (r["prereg"] + (" ok" if r["matches_prereg"] else " **MISS**"))
            )
            jk = (f"[{r['acc_ci_jackknife'][0]:.3f}, {r['acc_ci_jackknife'][1]:.3f}] "
                  f"{r['call_jackknife']}")  # fmt: skip
            if r["matches_prereg_jackknife"] is False:
                jk += " **MISS**"
            gap = (
                "—"
                if r["acc_minus_floor"] is None
                else f"{r['acc_minus_floor']:+.3f} ({r['floor']})"
            )
            L.append(f"| {name} | {r['model']} | {r['form']} | {r['role']} | {r['acc']:.3f} "
                     f"[{r['acc_ci'][0]:.3f}, {r['acc_ci'][1]:.3f}] | {r['s']:.3f} | {r['call']} "
                     f"| {pre} | {jk} | {gap} | {r['m21_data_nats_per_token']:+.4f} "
                     f"| {r['m21_control_nats_per_token']:+.4f} | {r['m21_excess_nats_per_token']:+.4f} "
                     f"| {r['n_pairs']} | {'pass' if r['control_check_pass'] else 'FAIL'} |")  # fmt: skip
        L += ["", PG_CALIBRATION_NOTE, ""]
        if pg["control_shared"] is False:
            L += ["**Control answers differ between reads** (control sha256 mismatch).", ""]
        for key, which in (("symbol_confounded", "pre-registered call"),
                           ("symbol_confounded_jackknife", "jackknife call")):  # fmt: skip
            if pg[key]:
                L += [f"**base_symbol reads elicit ({which}):** the symbol-form read is confounded "
                      "by surface cues; ep_symbol counts only through its gap over base_symbol "
                      "(as is).", ""]  # fmt: skip

    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--fracs", nargs="+", type=float,
                    default=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05])  # fmt: skip
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    a = ap.parse_args(argv)
    rep = build(a.root, a.fracs)
    md = markdown(rep)
    if a.out_json:
        a.out_json.write_text(json.dumps(rep, indent=1))
    if a.out_md:
        a.out_md.write_text(md)
    print(md)
    return 1 if rep["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
