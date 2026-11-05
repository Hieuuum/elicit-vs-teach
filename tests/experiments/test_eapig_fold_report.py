"""Tests for `fold_overlap.py` and `fold_report.py` (HANDOFF-folds.md Steps 3 and 5).

Synthetic score tensors (E = 2,000) and synthetic fold results trees in the real
JSON formats. CPU only, matplotlib Agg.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eapig-circuit-check"
sys.path.insert(0, str(ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"eapig_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


fo = _load("fold_overlap")
fr = _load("fold_report")
et = fo.et
E = 2000
FRACS = (0.005, 0.01, 0.05)


# ------------------------------------------------------------------ fixtures


def _write_scores(root: Path, fold: int, label: str, mean, mean_a=None, mean_b=None):
    task, tag = fo.LABELS[label]
    d = fo.fold_dir(root, fold) / task / tag
    d.mkdir(parents=True, exist_ok=True)
    m = torch.as_tensor(mean, dtype=torch.float32)
    torch.save(
        {
            "mean": m,
            "mean_a": m if mean_a is None else torch.as_tensor(mean_a, dtype=torch.float32),
            "mean_b": m if mean_b is None else torch.as_tensor(mean_b, dtype=torch.float32),
        },
        d / "scores.pt",
    )


def _scores_tree(root: Path, folds, seed=0, labels=None):
    """ec ~ ep_sym (elicit route overlaps), tc independent of fp."""
    rng = np.random.default_rng(seed)
    for f in folds:
        ep = rng.standard_normal(E)
        vals = {
            "ep_sym": ep,
            "ep_word": rng.standard_normal(E),
            "ec": ep + 0.3 * rng.standard_normal(E),
            "fp": rng.standard_normal(E),
            "tc": rng.standard_normal(E),
            "base": rng.standard_normal(E),
        }
        for lab, v in vals.items():
            if labels is None or lab in labels:
                _write_scores(
                    root,
                    f,
                    lab,
                    v,
                    v + 0.2 * rng.standard_normal(E),
                    v + 0.2 * rng.standard_normal(E),
                )


# ------------------------------------------------------------------ fold_overlap


def test_rank_matches_topk_edges_including_ties():
    s = np.round(np.random.default_rng(1).standard_normal(300), 1).astype(np.float32)  # many ties
    rank = fo.rank_of(s)
    for k in (1, 7, 50, 300):
        got = np.flatnonzero(rank < k)
        assert set(got) == set(et.topk_edges(s, k))
    assert sorted(rank) == list(range(300))


def test_jaccards_at_matches_et_jaccard():
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal(500), rng.standard_normal(500)
    ks = [1, 5, 40, 250]
    got = fo.jaccards_at(fo.rank_of(a), fo.rank_of(b), ks)
    for k, g in zip(ks, got):
        assert g == pytest.approx(et.jaccard(et.topk_edges(a, k), et.topk_edges(b, k)))


def test_identical_scores_jaccard_one_and_disjoint_zero():
    a = np.arange(100, dtype=np.float32)
    ra = fo.rank_of(a)
    assert fo.jaccards_at(ra, ra, [1, 10, 100]) == [1.0, 1.0, 1.0]
    # reversed order: top-10 sets are disjoint
    assert fo.jaccards_at(ra, fo.rank_of(-a), [10, 50])[0] == 0.0
    assert fo.jaccards_at(ra, fo.rank_of(-a), [50])[0] == 0.0


def test_run_output_structure_and_cross_fold_pairs(tmp_path):
    _scores_tree(tmp_path, range(10))
    res = fo.run(tmp_path, list(range(10)), FRACS, n_null=50, seed=0)
    assert res["E"] == E and res["folds"] == list(range(10))
    assert res["ks"] == [et.size_to_k(f, E) for f in FRACS]
    assert set(res["per_fold"]) == {str(i) for i in range(10)}
    rec = res["per_fold"]["3"]["0.01"]
    assert set(rec["routes"]) == set(fo.ROUTES)
    assert set(rec["ceiling_half"]) == set(fo.LABELS)
    for lab in fo.LABELS:
        st = res["cross_fold"][lab]["0.01"]
        assert len(st["values"]) == 45
        assert st["min"] <= st["mean"] <= st["max"]
    # elicit route overlaps more than the independent teach route
    assert rec["routes"]["elicit"] > rec["routes"]["teach"]


def test_cross_fold_identical_models_is_one_and_symmetric_count(tmp_path):
    v = np.random.default_rng(3).standard_normal(E)
    for f in range(4):
        for lab in fo.LABELS:
            _write_scores(tmp_path, f, lab, v)
    res = fo.run(tmp_path, [0, 1, 2, 3], FRACS, n_null=10)
    for lab in fo.LABELS:
        st = res["cross_fold"][lab]["0.05"]
        assert len(st["values"]) == 6 and st["mean"] == 1.0 and st["sd"] == 0.0
    assert res["per_fold"]["0"]["0.05"]["routes"]["teach"] == 1.0


def test_cross_fold_pair_values_match_direct_computation(tmp_path):
    _scores_tree(tmp_path, range(3), seed=5)
    res = fo.run(tmp_path, [0, 1, 2], FRACS, n_null=5)
    k = et.size_to_k(0.01, E)
    vals = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        a = fo._load(tmp_path, i, "tc")["mean"]
        b = fo._load(tmp_path, j, "tc")["mean"]
        vals.append(et.jaccard(et.topk_edges(a, k), et.topk_edges(b, k)))
    assert res["cross_fold"]["tc"]["0.01"]["values"] == pytest.approx(vals)


def test_missing_score_file_gives_null_not_crash(tmp_path):
    _scores_tree(tmp_path, range(3))
    (fo.fold_dir(tmp_path, 1) / "klword" / "teach_child" / "scores.pt").unlink()
    with pytest.warns(UserWarning, match="missing"):
        res = fo.run(tmp_path, [0, 1, 2], FRACS, n_null=5)
    r1 = res["per_fold"]["1"]["0.01"]
    assert r1["routes"]["teach"] is None and r1["routes"]["base_tc"] is None
    assert r1["routes"]["elicit"] is not None
    assert r1["ceiling_half"]["tc"] is None
    assert len(res["cross_fold"]["tc"]["0.01"]["values"]) == 1  # only (0,2)
    json.dumps(res)


def test_all_missing_raises(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(FileNotFoundError):
            fo.run(tmp_path, [0], FRACS, n_null=1)


def test_chance_analytic_and_null_mean(tmp_path):
    _scores_tree(tmp_path, range(2))
    res = fo.run(tmp_path, [0, 1], FRACS, n_null=400, seed=1)
    for fk, k in zip(["0.005", "0.01", "0.05"], res["ks"]):
        c = res["chance"][fk]
        assert c["analytic"] == pytest.approx(et.chance_jaccard(k, k, E))
        assert c["null_mean"] == pytest.approx(c["analytic"], abs=0.02)
        assert 0 <= c["null_p95"] <= 1


def test_null_deterministic_and_worker_invariant():
    a = fo.null_jaccards(500, [5, 50], 30, seed=7, workers=1)
    b = fo.null_jaccards(500, [5, 50], 30, seed=7, workers=1)
    c = fo.null_jaccards(500, [5, 50], 30, seed=7, workers=3)
    assert np.array_equal(a, b)
    assert a.shape == c.shape == (30, 2)
    assert np.allclose(np.sort(a, axis=0), np.sort(c, axis=0))


def test_overlap_cli_writes_json(tmp_path):
    _scores_tree(tmp_path, range(3))
    out = tmp_path / "overlap.json"
    fo.main(["--results-root", str(tmp_path), "--folds", "0", "1", "2", "--fracs", "0.01", "0.05",
             "--n-null", "20", "--out", str(out), "--workers", "2"])  # fmt: skip
    d = json.loads(out.read_text())
    assert d["fracs"] == [0.01, 0.05] and len(d["cross_fold"]["ec"]["0.01"]["values"]) == 3


def test_real_size_speed():
    import time

    n = 195_865
    rng = np.random.default_rng(0)
    ra, rb = fo.rank_of(rng.standard_normal(n)), fo.rank_of(rng.standard_normal(n))
    ks = [et.size_to_k(f, n) for f in fo.FRACS]
    t0 = time.time()
    fo.jaccards_at(ra, rb, ks)
    fo.null_jaccards(n, ks, 5, 0, 1)
    assert time.time() - t0 < 5


# ------------------------------------------------------------------ fold_report: stats


def test_paired_test_matches_scipy_and_hand_values():
    a = {i: v for i, v in enumerate([0.9, 0.8, 0.85, 0.95, 0.7])}
    b = {i: v for i, v in enumerate([0.4, 0.5, 0.3, 0.45, 0.35])}
    t = fr.paired_test(a, b)
    d = np.array([0.5, 0.3, 0.55, 0.5, 0.35])
    assert t["n"] == 5 and t["mean_diff"] == pytest.approx(d.mean())
    assert t["sd_diff"] == pytest.approx(d.std(ddof=1))
    tt = d.mean() / (d.std(ddof=1) / math.sqrt(5))
    assert t["t"] == pytest.approx(tt)
    assert t["p"] == pytest.approx(2 * stats.t.sf(abs(tt), 4))
    half = stats.t.ppf(0.975, 4) * d.std(ddof=1) / math.sqrt(5)
    assert t["ci95"] == pytest.approx([d.mean() - half, d.mean() + half])
    ref = stats.ttest_rel(list(a.values()), list(b.values()))
    assert t["p"] == pytest.approx(ref.pvalue)
    assert t["n_pos"] == 5 and t["n_flip"] == 0


def test_paired_test_counts_and_missing_folds():
    a = {0: 1.0, 1: 0.0, 2: 2.0, 3: None, 4: 1.0}
    b = {0: 0.5, 1: 0.5, 2: 2.0, 3: 0.0, 5: 0.0}
    t = fr.paired_test(a, b)
    assert t["n"] == 3  # folds 0,1,2 (3 has None, 4/5 unmatched)
    assert t["n_pos"] == 1 and t["n_flip"] == 2  # d = .5, -.5, 0


def test_paired_test_sd_zero_edge():
    a = {i: 0.5 for i in range(10)}
    b = {i: 0.2 for i in range(10)}
    t = fr.paired_test(a, b)
    assert t["sd_diff"] == 0.0 and t["p"] == 0.0 and t["t"] is None
    assert t["ci95"] == [pytest.approx(0.3)] * 2 and t["note"]
    z = fr.paired_test(a, a)
    assert z["p"] == 1.0 and z["n_pos"] == 0


def test_paired_test_tiny_n():
    assert fr.paired_test({}, {})["n"] == 0
    t = fr.paired_test({0: 1.0}, {0: 0.0})
    assert t["n"] == 1 and t["p"] is None


def test_holm_known_vector():
    adj = fr.holm([0.01, 0.04, 0.03, 0.005])
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone .04
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert fr.holm([0.6, 0.9]) == [1.0, 1.0]  # capped at 1 and monotone
    assert all(0 <= p <= 1 for p in fr.holm([0.9, 0.95, 0.99]))
    assert max(fr.holm([0.9, 0.95, 0.99])) == 1.0


def test_holm_monotone_and_none_skipped():
    ps = [0.2, None, 0.001, 0.04, float("nan")]
    adj = fr.holm(ps)
    assert adj[1] is None and adj[4] is None
    assert adj[2] == pytest.approx(0.003)  # 3 valid p's
    order = sorted([0, 2, 3], key=lambda i: ps[i])
    vals = [adj[i] for i in order]
    assert vals == sorted(vals)


def _t(n, n_pos, n_flip, p_holm):
    return {"n": n, "n_pos": n_pos, "n_flip": n_flip, "p_holm": p_holm}


def test_verdict_logic():
    assert fr.verdict(_t(10, 10, 0, 0.001)) == "holds"
    assert fr.verdict(_t(10, 9, 1, 0.04)) == "holds"
    assert fr.verdict(_t(10, 9, 1, 0.2)) == "not significant after Holm"
    assert fr.verdict(_t(10, 8, 2, 0.001)) == "not robust"  # 8/10 does not hold
    assert fr.verdict(_t(10, 5, 5, 0.9)) == "not robust"
    assert fr.verdict(_t(10, 0, 0, None)) == "insufficient data"
    assert fr.verdict(_t(1, 1, 0, None)) == "insufficient data"


def test_verdict_scales_with_fewer_folds():
    assert fr.verdict(_t(7, 6, 1, 0.01)) == "holds"  # >= n-1
    assert fr.verdict(_t(7, 5, 2, 0.01)) == "not robust"


def test_find_key_float_lookup():
    d = {"0.0020": 1, "0.01": 2}
    assert fr.find_key(d, 0.002) == "0.0020"
    assert fr.find_key(d, 0.01) == "0.01"
    assert fr.find_key(d, 0.05) is None
    assert fr.get_size({"0.0020": "x"}, 0.002) == "x"


def test_summarize():
    s = fr.summarize({0: 1.0, 1: 3.0, 2: None})
    assert s == {"mean": 2.0, "sd": pytest.approx(math.sqrt(2)), "min": 1.0, "max": 3.0, "n": 2}
    assert fr.summarize({})["n"] == 0


# ------------------------------------------------------------------ fold_report: full tree


def _full_tree(root: Path, folds=range(10), seed=0, flip_fold=None, func_key="0.0020"):
    rng = np.random.default_rng(seed)
    _scores_tree(root, folds, seed=seed)
    for f in folds:
        fd = fo.fold_dir(root, f)
        (fd / "functional").mkdir(parents=True, exist_ok=True)
        (fd / "cmap").mkdir(exist_ok=True)
        (fd / "hidden_pref").mkdir(exist_ok=True)
        bad = f == flip_fold

        def fsizes(share_key, hi):
            out = {}
            for fk in (func_key, "0.01", "0.05"):
                share = hi + 0.02 * rng.standard_normal()
                out[fk] = {"k": 1, "f": {"own": 0.9 + 0.01 * rng.standard_normal(), share_key: 0.5},
                           "share_of_own_f": {share_key: share, "own": 1.0, "ep_word": 0.3},
                           "f_random_band": [0, 0, 0], "overlap_with_own": {}}  # fmt: skip
            return {"sizes": out}

        (fd / "functional" / "elicit_child.json").write_text(json.dumps(fsizes("ep_sym", 0.9)))
        (fd / "functional" / "teach_child.json").write_text(
            json.dumps(fsizes("fp_word", 1.5 if bad else 0.3))
        )

        def csizes(rec):
            out = {}
            for fk in ("0.002", "0.0100", "0.05"):
                out[fk] = {"masks": {"child_nologits": {"recovery": rec + 0.02 * rng.standard_normal(),
                                                        "acc2": 0.5}},
                           "random_typed_disjoint": {"recovery_band": [-0.02, 0.0, 0.03]}}  # fmt: skip
            return {"sizes": out, "baseline": {"kl_child_parent_mean": 20.0}}

        (fd / "cmap" / "elicit.json").write_text(json.dumps(csizes(0.9)))
        (fd / "cmap" / "teach.json").write_text(json.dumps(csizes(1.5 if bad else -1.0)))
        for name, v in (("elicit_parent", 2.0), ("fmt_parent", 0.1), ("base", 0.0)):
            (fd / "hidden_pref" / f"{name}.json").write_text(json.dumps(
                {"paper_logit_diff_mean": v + 0.1 * rng.standard_normal(),
                 "full_answer_logp_diff_mean": v * 2 + 0.1 * rng.standard_normal(),
                 "paper_logit_diff_sd": 1.0, "pos_frac": 0.9}))  # fmt: skip


def _old(path: Path, **over):
    d = json.loads((EXP / "old_split.json").read_text())
    d.update(over)
    path.write_text(json.dumps(d))
    return path


def _run_report(tmp_path, folds=range(10), old=True, n_null=20, **kw):
    root = tmp_path / "R"
    _full_tree(root, folds, **kw)
    ov = fo.run(root, list(folds), FRACS, n_null=n_null)
    ovp = root / "overlap.json"
    ovp.write_text(json.dumps(ov))
    args = ["--results-root", str(root), "--folds", *map(str, folds), "--overlap", str(ovp),
            "--out-md", str(root / "r.md"), "--out-json", str(root / "r.json"),
            "--fig-dir", str(tmp_path / "figs")]  # fmt: skip
    if old:
        args += ["--old", str(EXP / "old_split.json")]
    fr.main(args)
    return root, json.loads((root / "r.json").read_text())


def test_full_tree_outputs(tmp_path):
    root, res = _run_report(tmp_path)
    md = (root / "r.md").read_text()
    assert "problem-sampling" in md and "NOT Holm" in md and "Pre-registered reading" in md
    for png in ("folds_overlap", "folds_functional", "folds_cmap", "folds_hidden_pref"):
        assert (tmp_path / "figs" / f"{png}.png").stat().st_size > 1000
    # elicit >> teach everywhere in the synthetic tree for functional + cmap
    for metric in ("functional", "cmap"):
        vs = [v for v in res["verdicts"] if v["metric"] == metric]
        assert len(vs) == 3 and all(v["verdict"] == "holds" for v in vs)
        assert all(v["n_pos"] == 10 for v in vs)
    assert res["summary"]["cmap"]["elicit_null_p95"]["0.002"]["mean"] == pytest.approx(0.03)
    for ver, h in res["hidden"].items():
        assert h["ep_gt_both_in_all_folds"] is True
        assert h["ep_minus_fp"]["n_pos"] == 10


def test_overlap_verdicts_and_floor(tmp_path):
    _, res = _run_report(tmp_path)
    assert [v["verdict"] for v in res["verdicts"] if v["metric"] == "overlap"] == ["holds"] * 3
    # teach is independent of base_tc in the synthetic tree -> at the floor
    assert res["floor_reading"]["teach_at_pretraining_floor"] is True
    assert res["floor_reading"]["sizes_testable"] == 3
    # elicit is far above base_ec
    ft = res["floor"]["elicit_minus_base_ec"]
    assert all(t["n_pos"] == 10 and t["p_holm"] < 0.05 for t in ft.values())
    # ceilings present
    c = res["ceilings"]["ec"]["0.01"]
    assert c["cross"]["mean"] is not None and c["chance_analytic"] > 0


def test_flip_folds_make_not_robust(tmp_path):
    _, res = _run_report(tmp_path, flip_fold=4)
    fv = [v for v in res["verdicts"] if v["metric"] == "functional"]
    assert all(
        v["n_flip"] == 1 for v in fv
    )  # one flipped fold: still "holds" if significant, else n.s.
    assert all(v["verdict"] != "not robust" for v in fv)


def test_two_flips_not_robust(tmp_path):
    root = tmp_path / "R"
    _full_tree(root, range(10), flip_fold=None)
    for f in (1, 2):  # make teach == elicit in two folds
        fd = fo.fold_dir(root, f) / "cmap"
        (fd / "teach.json").write_text((fd / "elicit.json").read_text())
    fr_args = ["--results-root", str(root), "--out-md", str(root / "r.md"), "--out-json",
               str(root / "r.json"), "--fig-dir", str(tmp_path / "f")]  # fmt: skip
    fr.main(fr_args)
    res = json.loads((root / "r.json").read_text())
    cm = [v for v in res["verdicts"] if v["metric"] == "cmap"]
    assert all(v["verdict"] == "not robust" and v["n_flip"] == 2 for v in cm)
    assert "none" in (root / "r.md").read_text()  # sizes where it holds: none (no --overlap)


def test_old_value_outside_range_flagged(tmp_path):
    old = json.loads((EXP / "old_split.json").read_text())
    root = tmp_path / "R"
    _full_tree(root)
    old["cmap_recovery"]["elicit"][0] = 0.1  # fold values ~0.9
    p = tmp_path / "old.json"
    p.write_text(json.dumps(old))
    fr.main(["--results-root", str(root), "--old", str(p), "--out-md", str(root / "r.md"),
             "--out-json", str(root / "r.json"), "--fig-dir", str(tmp_path / "f")])  # fmt: skip
    res = json.loads((root / "r.json").read_text())
    flagged = [c for c in res["old_check"] if c["flag"]]
    assert any(c["series"] == "elicit_recovery" and c["size"] == 0.002 for c in flagged)
    assert all(c["flag"] == "quote fold mean +- SD instead" for c in flagged)
    ok = [c for c in res["old_check"] if c["series"] == "teach_recovery" and c["size"] == 0.002]
    assert ok and ok[0]["inside"] is False  # old -0.896, folds ~ -1.0 +- 0.02: outside as well
    inside = [c for c in res["old_check"] if c["series"] == "elicit_share" and c["size"] == 0.05]
    assert inside and inside[0]["inside"] is False  # old .98 vs folds ~0.9
    assert "OUTSIDE" in (root / "r.md").read_text()


def test_old_value_rounding_tolerance():
    # old values are rounded (0.905 = 0.90476 rounded); a fold max of 0.90476 must count as inside
    stat = fr.summarize({0: 0.739, 1: 0.90476})
    assert fr._in_range(stat, 0.905) is True
    assert fr._in_range(stat, 0.906) is False
    assert fr._round_tol(0.905) == pytest.approx(0.0005)
    assert fr._round_tol(0.92) == pytest.approx(0.005)
    assert fr._round_tol(-1.14) == pytest.approx(0.005)
    stat2 = fr.summarize({0: 0.706, 1: 0.892})
    assert fr._in_range(stat2, 0.92) is False  # 2-dp old value still clearly outside
    stat3 = fr.summarize({0: -1.1712, 1: -1.068})
    assert fr._in_range(stat3, -1.171) is True  # negative values: tolerance on the low side


def test_old_value_inside_range_not_flagged():
    stat = fr.summarize({0: 0.5, 1: 0.7})
    assert fr._in_range(stat, 0.6) is True
    assert fr._in_range(stat, 0.71) is False
    assert fr._in_range(stat, None) is None
    assert (
        fr.old_lookup({"overlap": {"fracs": [0.1], "elicit": [None]}}, "overlap", "elicit", 0.1)
        is None
    )
    assert fr.old_lookup(None, "overlap", "elicit", 0.1) is None
    old = json.loads((EXP / "old_split.json").read_text())
    assert fr.old_lookup(old, "overlap", "teach", 0.0001) == 0.429
    assert fr.old_lookup(old, "functional_share", "teach", 0.05) is None
    assert fr.old_lookup(old, "cmap_recovery", "elicit", 0.01) == 0.917


def test_fewer_folds_scales_rule_and_notes(tmp_path):
    root, res = _run_report(tmp_path, folds=range(5))
    assert "scaled" in (root / "r.md").read_text()
    vs = [v for v in res["verdicts"] if v["metric"] == "cmap"]
    assert all(v["n"] == 5 and v["verdict"] == "holds" for v in vs)


def test_size_keys_vary_in_json(tmp_path):
    # functional keyed "0.0020", cmap keyed "0.0100": still found
    _, res = _run_report(tmp_path, func_key="0.0020")
    assert res["summary"]["functional"]["elicit_share"]["0.002"]["n"] == 10
    assert res["summary"]["cmap"]["elicit_recovery"]["0.01"]["n"] == 10


def test_missing_fold_data_tolerated(tmp_path):
    root = tmp_path / "R"
    _full_tree(root)
    (fo.fold_dir(root, 3) / "cmap" / "teach.json").unlink()
    (fo.fold_dir(root, 3) / "hidden_pref" / "base.json").unlink()
    fr.main(["--results-root", str(root), "--out-md", str(root / "r.md"), "--out-json",
             str(root / "r.json"), "--fig-dir", str(tmp_path / "f")])  # fmt: skip
    res = json.loads((root / "r.json").read_text())
    assert res["main"]["cmap"]["0.002"]["n"] == 9
    assert res["hidden"]["paper_logit_diff_mean"]["ep_minus_base"]["n"] == 9


def test_hidden_failure_reported(tmp_path):
    root = tmp_path / "R"
    _full_tree(root)
    p = fo.fold_dir(root, 0) / "hidden_pref" / "elicit_parent.json"
    d = json.loads(p.read_text())
    d["paper_logit_diff_mean"] = -5.0
    p.write_text(json.dumps(d))
    fr.main(["--results-root", str(root), "--out-md", str(root / "r.md"), "--out-json",
             str(root / "r.json"), "--fig-dir", str(tmp_path / "f")])  # fmt: skip
    res = json.loads((root / "r.json").read_text())
    assert res["hidden"]["paper_logit_diff_mean"]["ep_gt_both_in_all_folds"] is False
    assert res["hidden"]["full_answer_logp_diff_mean"]["ep_gt_both_in_all_folds"] is True
    assert "FAILS" in (root / "r.md").read_text()


def test_old_split_json_values():
    old = json.loads((EXP / "old_split.json").read_text())
    assert len(old["overlap"]["fracs"]) == 9
    for k in ("elicit", "teach", "base_ec", "base_tc"):
        assert len(old["overlap"][k]) == 9
    assert old["functional_share"]["teach"][2] is None
    assert old["hidden_pref_paper"]["elicit_parent"] == 2.31
    assert "AningTbDXYYhYDN3JHu327" in old["source"]
