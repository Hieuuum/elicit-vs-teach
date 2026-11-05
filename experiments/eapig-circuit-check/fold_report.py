"""Report for the 10-fold robustness check (HANDOFF-folds.md, Step 5).

Collects per-fold values of the four kept metrics (edge overlap, functional
reuse, CMAP recovery, hidden preference), summarises them (mean, SD, min, max,
n), runs paired two-sided t-tests across folds, applies the pre-registered
reading, and writes markdown + JSON + four PNGs.

    python3 fold_report.py --results-root results_folds --folds 0 1 2 3 4 5 6 7 8 9 \
        --overlap results_folds/overlap.json --old old_split.json \
        --out-md results_folds/fold_report.md --out-json results_folds/fold_report.json \
        --fig-dir figures

Pre-registered reading (fixed before any number existed):
  holds        = elicit > teach in >= 9 of 10 folds AND Holm-adjusted paired p < 0.05
                 (n < 10 folds present: ">= n-1"; n = 10 gives 9)
  not robust   = elicit - teach <= 0 in >= 2 folds
  old value outside [fold min, fold max] -> "quote fold mean +- SD instead"
  teach - base_tc not significant at most sizes -> teach route sits at the pretraining floor
  hidden pref  = elicit parent > format parent and > base in all folds, per version
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

COLORS = {"elicit": "#b07d10", "teach": "#137a7a", "base": "#8a877c"}
FUNC_SIZES = (0.002, 0.01, 0.05)
CMAP_SIZES = (0.002, 0.01, 0.05)
HIDDEN_VERSIONS = ("paper_logit_diff_mean", "full_answer_logp_diff_mean")
HIDDEN_PARENTS = {"ep": "elicit_parent", "fp": "fmt_parent", "base": "base"}
OVERLAP_SERIES = ("elicit", "teach", "base_ec", "base_tc", "elicit_word")
CEIL_LABELS = ("ep_sym", "ep_word", "ec", "fp", "tc", "base")
SOURCE_NOTE = (
    "CIs and tests cover problem-sampling variation only (10 disjoint problem folds), not "
    "training-seed variation (one training run per child); one prompt template, one task "
    "(4-digit addition)."
)

# series -> color family
FAMILY = {
    "elicit": "elicit", "teach": "teach", "base_ec": "base", "base_tc": "base",
    "elicit_share": "elicit", "teach_share": "teach",
    "elicit_recovery": "elicit", "teach_recovery": "teach",
}  # fmt: skip


# ----------------------------------------------------------------- helpers


def find_key(mapping: dict, value: float):
    """Key of `mapping` whose float value equals `value` ("0.002" vs "0.0020")."""
    for k in mapping:
        try:
            if math.isclose(float(k), float(value), rel_tol=1e-9, abs_tol=1e-12):
                return k
        except (TypeError, ValueError):
            continue
    return None


def get_size(sizes: dict, frac: float):
    k = find_key(sizes, frac)
    return None if k is None else sizes[k]


def _num(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def summarize(vals: dict) -> dict:
    """mean / sd / min / max / n over the folds present (non-null, finite)."""
    a = np.asarray([v for v in vals.values() if _num(v) is not None], dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "sd": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def holm(ps: list) -> list:
    """Holm step-down adjusted p-values (monotone, capped at 1); None entries are skipped."""
    idx = [i for i, p in enumerate(ps) if p is not None and not math.isnan(p)]
    order = sorted(idx, key=lambda i: ps[i])
    m = len(order)
    out: list = [None] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        out[i] = running
    return out


def paired_test(a: dict, b: dict) -> dict:
    """Two-sided paired t-test of a - b over folds where both are present."""
    folds = sorted(f for f in a if f in b and _num(a[f]) is not None and _num(b[f]) is not None)
    n = len(folds)
    base = {"n": n, "mean_diff": None, "sd_diff": None, "t": None, "p": None,
            "ci95": [None, None], "n_pos": 0, "n_flip": 0, "diffs": {}, "note": None}  # fmt: skip
    if n == 0:
        return base
    x = np.array([a[f] for f in folds], dtype=np.float64)
    y = np.array([b[f] for f in folds], dtype=np.float64)
    d = x - y
    base["diffs"] = {str(f): float(v) for f, v in zip(folds, d)}
    base["mean_diff"] = float(d.mean())
    base["n_pos"] = int((d > 0).sum())
    base["n_flip"] = int((d <= 0).sum())
    if n < 2:
        base["note"] = "n < 2: no test"
        return base
    sd = float(d.std(ddof=1))
    base["sd_diff"] = sd
    if sd <= 1e-12 * max(1.0, abs(base["mean_diff"])):
        base["sd_diff"] = 0.0
        base["p"] = 0.0 if base["mean_diff"] != 0 else 1.0
        base["ci95"] = [base["mean_diff"]] * 2
        base["note"] = (
            "all per-fold differences identical (sd = 0): t undefined, p set by sign of the mean"
        )
        return base
    res = stats.ttest_rel(x, y)
    ci = res.confidence_interval(0.95)
    base["t"] = float(res.statistic)
    base["p"] = float(res.pvalue)
    base["ci95"] = [float(ci.low), float(ci.high)]
    return base


def tests_across_sizes(series_a: dict, series_b: dict, sizes) -> dict:
    """{size: paired test + p_holm}; Holm across the sizes of this one comparison."""
    out = {}
    for s in sizes:
        out[s] = paired_test(series_a.get(s, {}), series_b.get(s, {}))
    adj = holm([out[s]["p"] for s in sizes])
    for s, p in zip(sizes, adj):
        out[s]["p_holm"] = p
    return out


def verdict(test: dict) -> str:
    """Pre-registered reading for one metric/size from its main (elicit - teach) test."""
    n = test["n"]
    if n < 2 or test.get("p_holm") is None:
        return "insufficient data"
    if test["n_flip"] >= 2:
        return "not robust"
    need = n - 1  # 9 of 10; scaled for n < 10
    if test["n_pos"] >= need and test["p_holm"] < 0.05:
        return "holds"
    return "not significant after Holm"


def _round_tol(v: float) -> float:
    """Half a unit in the last printed decimal of an old (rounded) value: 0.905 -> 0.0005."""
    s = repr(float(v))
    if "e" in s or "." not in s:
        return 0.0
    return 0.5 * 10.0 ** -len(s.split(".")[1])


def _in_range(stat: dict, v) -> bool | None:
    """Is the old value inside [fold min, fold max], allowing for the old value's rounding."""
    if v is None or stat["n"] == 0:
        return None
    tol = _round_tol(v)
    return bool(stat["min"] - tol <= v <= stat["max"] + tol)


# ----------------------------------------------------------------- collection


def collect(root: Path, folds: list[int], overlap: dict | None) -> dict:
    """data[metric][series][size_or_None] = {fold: value or None}."""
    data: dict = {"overlap": {}, "functional": {}, "cmap": {}, "hidden": {}}

    def put(metric, series, size, fold, val):
        data[metric].setdefault(series, {}).setdefault(size, {})[fold] = _num(val)

    if overlap:
        for frac in overlap["fracs"]:
            for fold in folds:
                pf = overlap["per_fold"].get(str(fold), {})
                kf = find_key(pf, frac)
                rec = pf[kf] if kf is not None else None
                for s in OVERLAP_SERIES:
                    put("overlap", s, frac, fold, rec["routes"].get(s) if rec else None)
                for lab in CEIL_LABELS:
                    put(
                        "overlap",
                        f"half:{lab}",
                        frac,
                        fold,
                        rec["ceiling_half"].get(lab) if rec else None,
                    )

    for fold in folds:
        fd = root / f"fold_{fold:02d}"
        for series, fname, key in (("elicit", "elicit_child.json", "ep_sym"),
                                   ("teach", "teach_child.json", "fp_word")):  # fmt: skip
            d = _read_json(fd / "functional" / fname)
            for frac in FUNC_SIZES:
                rec = get_size(d["sizes"], frac) if d else None
                put(
                    "functional",
                    f"{series}_share",
                    frac,
                    fold,
                    rec["share_of_own_f"].get(key) if rec else None,
                )
                put(
                    "functional",
                    f"{series}_f_own",
                    frac,
                    fold,
                    rec["f"].get("own") if rec else None,
                )
                if series == "elicit":
                    put("functional", "elicit_share_word", frac, fold,
                        rec["share_of_own_f"].get("ep_word") if rec else None)  # fmt: skip
        for series, fname in (("elicit", "elicit.json"), ("teach", "teach.json")):
            d = _read_json(fd / "cmap" / fname)
            for frac in CMAP_SIZES:
                rec = get_size(d["sizes"], frac) if d else None
                m = rec["masks"]["child_nologits"] if rec else None
                put("cmap", f"{series}_recovery", frac, fold, m["recovery"] if m else None)
                put("cmap", f"{series}_acc2", frac, fold, m["acc2"] if m else None)
                null = rec.get("random_typed_disjoint") if rec else None
                put(
                    "cmap",
                    f"{series}_null_p95",
                    frac,
                    fold,
                    null["recovery_band"][2] if null else None,
                )
        for ver in HIDDEN_VERSIONS:
            for short, name in HIDDEN_PARENTS.items():
                d = _read_json(fd / "hidden_pref" / f"{name}.json")
                put("hidden", f"{ver}|{short}", None, fold, d.get(ver) if d else None)
    return data


def old_lookup(old: dict | None, section: str, series: str, frac: float):
    if not old or section not in old:
        return None
    sec = old[section]
    if series not in sec or "fracs" not in sec:
        return None
    for f, v in zip(sec["fracs"], sec[series]):
        if math.isclose(float(f), float(frac), rel_tol=1e-9, abs_tol=1e-12):
            return _num(v)
    return None


OLD_MAP = {  # (metric, series) -> (old section, old series)
    ("overlap", "elicit"): ("overlap", "elicit"),
    ("overlap", "teach"): ("overlap", "teach"),
    ("overlap", "base_ec"): ("overlap", "base_ec"),
    ("overlap", "base_tc"): ("overlap", "base_tc"),
    ("functional", "elicit_share"): ("functional_share", "elicit"),
    ("functional", "teach_share"): ("functional_share", "teach"),
    ("cmap", "elicit_recovery"): ("cmap_recovery", "elicit"),
    ("cmap", "teach_recovery"): ("cmap_recovery", "teach"),
}


# ----------------------------------------------------------------- analysis


def analyse(data: dict, overlap: dict | None, old: dict | None, folds: list[int]) -> dict:
    out: dict = {"folds": folds, "summary": {}, "main": {}, "floor": {}, "hidden": {},
                 "verdicts": [], "old_check": [], "ceilings": {}, "floor_reading": {}}  # fmt: skip

    for metric in ("overlap", "functional", "cmap"):
        out["summary"][metric] = {
            s: {str(sz): summarize(v) for sz, v in by.items()} for s, by in data[metric].items()
        }
    out["summary"]["hidden"] = {s: summarize(by[None]) for s, by in data["hidden"].items()}

    main_pairs = {"overlap": ("elicit", "teach"), "functional": ("elicit_share", "teach_share"),
                  "cmap": ("elicit_recovery", "teach_recovery")}  # fmt: skip
    for metric, (sa, sb) in main_pairs.items():
        a, b = data[metric].get(sa, {}), data[metric].get(sb, {})
        sizes = sorted(set(a) | set(b))
        res = tests_across_sizes(a, b, sizes)
        out["main"][metric] = {str(s): t for s, t in res.items()}
        for s, t in res.items():
            out["verdicts"].append({"metric": metric, "size": s, "verdict": verdict(t),
                                    "n_pos": t["n_pos"], "n_flip": t["n_flip"], "n": t["n"],
                                    "p_holm": t["p_holm"]})  # fmt: skip

    ov = data["overlap"]
    for name, (sa, sb) in {"teach_minus_base_tc": ("teach", "base_tc"),
                           "elicit_minus_base_ec": ("elicit", "base_ec")}.items():  # fmt: skip
        sizes = sorted(set(ov.get(sa, {})) | set(ov.get(sb, {})))
        res = tests_across_sizes(ov.get(sa, {}), ov.get(sb, {}), sizes)
        out["floor"][name] = {str(s): t for s, t in res.items()}
    ft = out["floor"].get("teach_minus_base_tc", {})
    testable = [t for t in ft.values() if t["p_holm"] is not None]
    n_sig = sum(1 for t in testable if t["p_holm"] < 0.05 and t["mean_diff"] > 0)
    out["floor_reading"] = {
        "sizes_testable": len(testable),
        "sizes_significant_above_floor": n_sig,
        "teach_at_pretraining_floor": bool(testable) and n_sig < len(testable) / 2,
        "text": (
            "teach route sits at the pretraining floor (teach - base_tc not significant at most sizes)"
            if testable and n_sig < len(testable) / 2
            else "teach route is above the pretraining floor at most sizes"
            if testable
            else "no floor data"
        ),
    }

    for ver in HIDDEN_VERSIONS:
        ep, fp, bs = (
            data["hidden"].get(f"{ver}|{k}", {}).get(None, {}) for k in ("ep", "fp", "base")
        )
        t_fp, t_base = paired_test(ep, fp), paired_test(ep, bs)
        n_all = max(t_fp["n"], t_base["n"])
        ok = n_all >= 2 and t_fp["n_pos"] == t_fp["n"] and t_base["n_pos"] == t_base["n"]
        out["hidden"][ver] = {"ep_minus_fp": t_fp, "ep_minus_base": t_base,
                              "ep_gt_both_in_all_folds": bool(ok)}  # fmt: skip

    # old single-split values vs the fold range
    for (metric, series), (sec, oseries) in OLD_MAP.items():
        for sz, stat in out["summary"][metric].get(series, {}).items():
            ov_ = old_lookup(old, sec, oseries, float(sz))
            if ov_ is None:
                continue
            inside = _in_range(stat, ov_)
            out["old_check"].append({"metric": metric, "series": series, "size": float(sz),
                                     "old": ov_, "fold_min": stat["min"], "fold_max": stat["max"],
                                     "inside": inside,
                                     "flag": None if inside in (True, None) else
                                     "quote fold mean +- SD instead"})  # fmt: skip
    if old and "hidden_pref_paper" in old:
        for short in ("ep", "fp", "base"):
            ov_ = _num(old["hidden_pref_paper"].get(HIDDEN_PARENTS[short]))
            stat = out["summary"]["hidden"].get(f"paper_logit_diff_mean|{short}")
            if ov_ is None or stat is None:
                continue
            inside = _in_range(stat, ov_)
            out["old_check"].append({"metric": "hidden", "series": f"paper|{short}", "size": None,
                                     "old": ov_, "fold_min": stat["min"], "fold_max": stat["max"],
                                     "inside": inside,
                                     "flag": None if inside in (True, None) else
                                     "quote fold mean +- SD instead",
                                     "note": old["hidden_pref_paper"].get("note")})  # fmt: skip

    # ceilings
    if overlap:
        for lab in CEIL_LABELS:
            out["ceilings"][lab] = {}
            for f in overlap["fracs"]:
                half = summarize(data["overlap"].get(f"half:{lab}", {}).get(f, {}))
                ck = find_key(overlap["cross_fold"].get(lab, {}), f)
                cross = overlap["cross_fold"][lab][ck] if ck is not None else None
                chk = find_key(overlap["chance"], f)
                out["ceilings"][lab][str(f)] = {
                    "half": half,
                    "cross": None
                    if cross is None
                    else {k: cross[k] for k in ("mean", "sd", "min", "max")},
                    "chance_analytic": overlap["chance"][chk]["analytic"] if chk else None,
                    "chance_null_p95": overlap["chance"][chk]["null_p95"] if chk else None,
                }
    return out


# ----------------------------------------------------------------- markdown


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _cell(st):
    if st is None or st["n"] == 0:
        return "n/a"
    s = f"{_f(st['mean'])} ± {_f(st['sd'])} [{_f(st['min'])}, {_f(st['max'])}]"
    return s + (f" (n={st['n']})" if st["n"] < 10 else "")


def _test_row(label, t):
    p = t.get("p_holm")
    ci = t["ci95"]
    return (f"| {label} | {t['n']} | {_f(t['mean_diff'])} | {_f(t['sd_diff'])} | {_f(t['t'], 2)} | "
            f"{_f(t['p'], 4)} | {_f(p, 4)} | {t['n_pos']}/{t['n']} | {t['n_flip']} | "
            f"[{_f(ci[0])}, {_f(ci[1])}] |")  # fmt: skip


TEST_HEAD = ("| size | n | d̄ | sd | t | p | p (Holm) | folds d>0 | sign flips (d<=0) | 95% CI (per size, not Holm) |\n"
             "|---|---|---|---|---|---|---|---|---|---|")  # fmt: skip


def to_markdown(res: dict, overlap: dict | None) -> str:
    L: list[str] = []
    n_folds = len(res["folds"])
    L += ["# 10-fold robustness report", "",
          f"Folds: {res['folds']} ({n_folds}). {SOURCE_NOTE}",
          "95% CIs are per size and per test, NOT Holm-adjusted; p-values are raw and Holm-adjusted "
          "across sizes within each metric. Cells: mean ± SD [min, max] over folds.", ""]  # fmt: skip
    if n_folds < 10:
        L += [f"Only {n_folds} folds: the '>= 9 of 10' rule is scaled to '>= n-1'.", ""]

    def table(metric, series_list, title, sizes_fmt=str):
        summ = res["summary"][metric]
        sizes = sorted({float(s) for ser in series_list for s in summ.get(ser, {})})
        L.extend([f"### {title}", "", "| size | " + " | ".join(series_list) + " |",
                  "|---|" + "---|" * len(series_list)])  # fmt: skip
        for s in sizes:
            cells = [_cell(summ.get(ser, {}).get(str(s))) for ser in series_list]
            L.append(f"| {s:g} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## Edge overlap (Jaccard)", ""]
    table("overlap", list(OVERLAP_SERIES), "Route overlaps (elicit = ep_sym->ec, teach = fp->tc)")
    if overlap:
        L += ["### Ceilings and chance", "",
              "| size | chance (analytic) | chance null p95 | " +
              " | ".join(f"{lab} half / cross" for lab in CEIL_LABELS) + " |",
              "|---|---|---|" + "---|" * len(CEIL_LABELS)]  # fmt: skip
        for f in overlap["fracs"]:
            row = []
            for lab in CEIL_LABELS:
                c = res["ceilings"][lab][str(f)]
                cross = c["cross"]
                row.append(f"{_f(c['half']['mean'])} / " +
                           ("n/a" if not cross or cross["mean"] is None
                            else f"{_f(cross['mean'])} ± {_f(cross['sd'])}"))  # fmt: skip
            c0 = res["ceilings"][CEIL_LABELS[0]][str(f)]
            L.append(f"| {f:g} | {_f(c0['chance_analytic'], 4)} | {_f(c0['chance_null_p95'], 4)} | " +
                     " | ".join(row) + " |")  # fmt: skip
        L.append("")
    L += ["## Functional reuse (share of the child's own faithfulness)", ""]
    table("functional", ["elicit_share", "teach_share", "elicit_f_own", "teach_f_own", "elicit_share_word"],
          "Share = f(parent circuit) / f(own circuit); elicit = ep_sym, teach = fp_word")  # fmt: skip
    L += ["## CMAP (child_nologits)", ""]
    table("cmap", ["elicit_recovery", "teach_recovery", "elicit_acc2", "teach_acc2",
                   "elicit_null_p95", "teach_null_p95"], "Recovery, 2-token accuracy, disjoint-null p95")  # fmt: skip
    L += ["## Hidden preference (validation problems, parent alone)", "",
          "| version | elicit parent | format parent | base |", "|---|---|---|---|"]  # fmt: skip
    for ver in HIDDEN_VERSIONS:
        L.append(f"| {ver} | " + " | ".join(_cell(res["summary"]["hidden"].get(f"{ver}|{k}"))
                                              for k in ("ep", "fp", "base")) + " |")  # fmt: skip
    L.append("")

    L += ["## Main tests: elicit - teach (paired, two-sided, df = n-1)", ""]
    for metric in ("overlap", "functional", "cmap"):
        L += [f"### {metric}", "", TEST_HEAD]
        for s, t in res["main"][metric].items():
            L.append(_test_row(f"{float(s):g}", t))
            if t["note"]:
                L.append(f"|  | note: {t['note']} |||||||||")
        L.append("")
    L += ["## Floor tests (overlap): child - base->child (paired)", ""]
    for name, by in res["floor"].items():
        L += [f"### {name}", "", TEST_HEAD]
        L += [_test_row(f"{float(s):g}", t) for s, t in by.items()]
        L.append("")
    L += ["## Hidden preference tests (raw p, no Holm; CI per test)", "",
          "| contrast | n | d̄ | sd | t | p | folds d>0 | 95% CI |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for ver, h in res["hidden"].items():
        for name in ("ep_minus_fp", "ep_minus_base"):
            t = h[name]
            L.append(f"| {ver}: {name} | {t['n']} | {_f(t['mean_diff'])} | {_f(t['sd_diff'])} | "
                     f"{_f(t['t'], 2)} | {_f(t['p'], 4)} | {t['n_pos']}/{t['n']} | "
                     f"[{_f(t['ci95'][0])}, {_f(t['ci95'][1])}] |")  # fmt: skip
    L.append("")

    L += ["## Pre-registered reading", "",
          "holds = elicit > teach in >= 9/10 folds AND Holm p < 0.05; not robust = elicit - teach <= 0 "
          "in >= 2 folds.", ""]  # fmt: skip
    for metric in ("overlap", "functional", "cmap"):
        L.append(f"**{metric}**")
        for v in (x for x in res["verdicts"] if x["metric"] == metric):
            L.append(f"- size {v['size']:g}: {v['verdict']} "
                     f"({v['n_pos']}/{v['n']} folds elicit > teach, {v['n_flip']} sign flips, "
                     f"Holm p = {_f(v['p_holm'], 4)})")  # fmt: skip
        holds = [
            f"{v['size']:g}"
            for v in res["verdicts"]
            if v["metric"] == metric and v["verdict"] == "holds"
        ]
        L.append(f"- sizes where it holds: {', '.join(holds) if holds else 'none'}")
        L.append("")
    L += ["**Floor**", f"- {res['floor_reading']['text']} "
          f"({res['floor_reading']['sizes_significant_above_floor']} of "
          f"{res['floor_reading']['sizes_testable']} sizes significantly above base_tc after Holm)", ""]  # fmt: skip
    L.append("**Hidden preference**")
    for ver, h in res["hidden"].items():
        L.append(f"- {ver}: " + ("elicit parent > format parent and > base in all folds"
                                  if h["ep_gt_both_in_all_folds"]
                                  else "FAILS: not (ep > fp and ep > base) in all folds"))  # fmt: skip
    L.append("")
    L.append("**Old single split vs fold range**")
    if not res["old_check"]:
        L.append("- (no --old file given)")
    for c in res["old_check"]:
        sz = "" if c["size"] is None else f" @ {c['size']:g}"
        status = ("no fold data" if c["inside"] is None else
                  "inside" if c["inside"] else f"OUTSIDE -> {c['flag']}")  # fmt: skip
        note = f" [{c['note']}]" if c.get("note") else ""
        L.append(f"- {c['metric']}/{c['series']}{sz}: old {_f(c['old'])} vs "
                 f"[{_f(c['fold_min'])}, {_f(c['fold_max'])}]: {status}{note}")  # fmt: skip
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------- figures


def _box(ax, pos, vals, color, width):
    vals = [v for v in vals if _num(v) is not None]
    if not vals:
        return
    bp = ax.boxplot([vals], positions=[pos], widths=width, patch_artist=True, manage_ticks=False,
                    showfliers=True, medianprops={"color": "black"})  # fmt: skip
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)


def size_boxplot(path: Path, by_series: dict, sizes, title, ylabel, old=None, extra=None):
    """by_series: name -> (color, {size: {fold: val}}); old: name -> {size: val}."""
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(sizes) + 2), 4.2))
    names = list(by_series)
    step = 0.8 / max(len(names), 1)  # one evenly spaced slot per size, groups never overlap
    for j, name in enumerate(names):
        color, per_size = by_series[name]
        off = (j - (len(names) - 1) / 2) * step
        for i, s in enumerate(sizes):
            x = i + off
            _box(ax, x, list(per_size.get(s, {}).values()), color, step * 0.8)
            if old and name in old and old[name].get(s) is not None:
                ax.scatter([x], [old[name][s]], marker="D", color="black", s=22, zorder=5)
            if extra and name in extra and extra[name].get(s):
                ev = [v for v in extra[name][s].values() if _num(v) is not None]
                ax.scatter([x] * len(ev), ev, marker="x", color="#444", s=14, zorder=4)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s * 100:g}%" for s in sizes])
    ax.set_xlabel("circuit size (share of edges)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=by_series[n][0], alpha=0.7) for n in names]
    labels = list(names)
    if old:
        handles.append(plt.Line2D([], [], marker="D", color="black", ls="", markersize=5))
        labels.append("old single split")
    ax.legend(handles, labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_figures(data: dict, old: dict | None, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    ov = data["overlap"]
    sizes = sorted(ov.get("elicit", {}))
    series = {
        n: (COLORS[FAMILY[n]], ov.get(n, {})) for n in ("elicit", "teach", "base_ec", "base_tc")
    }
    oldv = (
        {n: {s: old_lookup(old, "overlap", n, s) for s in sizes} for n in series} if old else None
    )
    p = fig_dir / "folds_overlap.png"
    size_boxplot(p, series, sizes, "Edge overlap across folds", "Jaccard", oldv)
    paths.append(p)

    fn = data["functional"]
    series = {n: (COLORS[FAMILY[n]], fn.get(n, {})) for n in ("elicit_share", "teach_share")}
    oldv = ({n: {s: old_lookup(old, "functional_share", n.split("_")[0], s) for s in FUNC_SIZES}
             for n in series} if old else None)  # fmt: skip
    p = fig_dir / "folds_functional.png"
    size_boxplot(p, series, list(FUNC_SIZES), "Functional reuse across folds",
                 "share of child's own faithfulness", oldv)  # fmt: skip
    paths.append(p)

    cm = data["cmap"]
    series = {n: (COLORS[FAMILY[n]], cm.get(n, {})) for n in ("elicit_recovery", "teach_recovery")}
    oldv = ({n: {s: old_lookup(old, "cmap_recovery", n.split("_")[0], s) for s in CMAP_SIZES}
             for n in series} if old else None)  # fmt: skip
    extra = {
        "elicit_recovery": cm.get("elicit_null_p95", {}),
        "teach_recovery": cm.get("teach_null_p95", {}),
    }
    p = fig_dir / "folds_cmap.png"
    size_boxplot(p, series, list(CMAP_SIZES), "CMAP recovery across folds (x = disjoint-null p95)",
                 "recovery", oldv, extra)  # fmt: skip
    paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, ver in zip(axes, HIDDEN_VERSIONS):
        for j, (short, fam) in enumerate((("ep", "elicit"), ("fp", "teach"), ("base", "base"))):
            vals = list(data["hidden"].get(f"{ver}|{short}", {}).get(None, {}).values())
            _box(ax, j, vals, COLORS[fam], 0.5)
            v = [x for x in vals if _num(x) is not None]
            ax.scatter([j] * len(v), v, color="black", s=8, zorder=4)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["elicit parent", "format parent", "base"])
        ax.set_title(ver, fontsize=9)
        ax.axhline(0, color="#999", lw=0.8)
    axes[0].set_ylabel("preference for correct answer (nats)")
    fig.tight_layout()
    p = fig_dir / "folds_hidden_pref.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


# ----------------------------------------------------------------- main


def _jsonable(o):
    if isinstance(o, dict):
        return {("null" if k is None else str(k)): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return _num(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results-root", required=True, type=Path)
    p.add_argument("--folds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--overlap", type=Path, default=None)
    p.add_argument("--old", type=Path, default=None)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--fig-dir", required=True, type=Path)
    return p.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    overlap = _read_json(a.overlap) if a.overlap else None
    old = _read_json(a.old) if a.old else None
    data = collect(a.results_root, a.folds, overlap)
    res = analyse(data, overlap, old, a.folds)
    a.out_md.parent.mkdir(parents=True, exist_ok=True)
    a.out_md.write_text(to_markdown(res, overlap))
    a.out_json.write_text(json.dumps(_jsonable(res), indent=1))
    make_figures(data, old, a.fig_dir)
    print(f"wrote {a.out_md}, {a.out_json}, figures in {a.fig_dir}")


if __name__ == "__main__":
    main()
