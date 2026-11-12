"""Tests for experiments/eapig-circuit-check/ts1b_hard_acc_report.py (CPU, synthetic files)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

EXP = Path(__file__).resolve().parents[2] / "experiments" / "eapig-circuit-check"


def _load(name: str):
    sys.path.insert(0, str(EXP))
    try:
        spec = importlib.util.spec_from_file_location(f"eapig_{name}", EXP / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(EXP))
    return mod


rp = _load("ts1b_hard_acc_report")
ob = _load("overlap_boot")
cr = _load("ceiling_report")

N, E = 40, 2000
HALF = np.tile([0, 1], N // 2)
FRACS = [0.01, 0.05]


# ----------------------------------------------------------------------------- margin


def cell(point, boot):
    return {"point": point, "boot": list(boot)}


def test_margin_identical_cells_is_zero_and_not_reuse():
    b = np.random.default_rng(0).uniform(0.2, 0.4, 50)
    m = rp.margin(cell(0.3, b), cell(0.3, b))
    assert m == {"point": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "reuse": False}


def test_margin_constant_shift_is_reuse():
    b = np.random.default_rng(1).uniform(0.2, 0.4, 50)
    m = rp.margin(cell(0.5, b + 0.2), cell(0.3, b))
    assert m["point"] == pytest.approx(0.2)
    assert m["ci_lo"] == pytest.approx(0.2) and m["ci_hi"] == pytest.approx(0.2)
    assert m["reuse"]


def test_margin_negative_shift_never_reuse():
    b = np.random.default_rng(2).uniform(0.2, 0.4, 50)
    m = rp.margin(cell(0.1, b - 0.2), cell(0.3, b))
    assert m["point"] == pytest.approx(-0.2) and m["ci_hi"] < 0 and not m["reuse"]


def test_margin_matches_hand_basic_interval_and_is_paired():
    rng = np.random.default_rng(3)
    shared = rng.normal(0, 0.1, 200)  # common resample noise: cancels only if paired
    br, bc = 0.6 + shared + rng.normal(0, 0.01, 200), 0.5 + shared + rng.normal(0, 0.01, 200)
    m = rp.margin(cell(0.6, br), cell(0.5, bc))
    d = br - bc
    q_lo, q_hi = np.percentile(d, [2.5, 97.5])
    assert m["ci_lo"] == pytest.approx(2 * 0.1 - q_hi) and m["ci_hi"] == pytest.approx(
        2 * 0.1 - q_lo
    )
    assert m["reuse"]  # paired differencing removes the shared noise
    # unpaired (shuffled control) would be far wider and straddle 0
    lo, hi = rp.basic_ci(0.1, br - rng.permutation(bc))
    assert lo < 0 < hi


def test_margin_rejects_mismatched_or_empty_draws():
    with pytest.raises(ValueError):
        rp.margin(cell(0.5, [0.1, 0.2]), cell(0.4, [0.1]))
    with pytest.raises(ValueError):
        rp.margin(cell(0.5, []), cell(0.4, []))


def test_margin_interval_clipped_to_unit_range():
    b = np.array([0.0] * 10 + [0.5] * 10)
    m = rp.margin(cell(1.0, np.full(20, 1.0)), cell(0.0, b))
    assert m["ci_hi"] == 1.0 and m["point"] == 1.0
    m = rp.margin(cell(0.0, b), cell(1.0, np.full(20, 1.0)))
    assert m["ci_lo"] == -1.0


def test_basic_ci_reflects_percentiles():
    draws = np.arange(101, dtype=float) / 100  # percentiles 2.5 -> 0.025, 97.5 -> 0.975
    assert rp.basic_ci(0.5, draws) == pytest.approx((0.025, 0.975))
    assert rp.basic_ci(0.7, draws) == pytest.approx((0.425, 1.375))


# ----------------------------------------------------------------------------- normalized


def test_normalized_unit_ceilings_are_identity():
    c = {"point": 0.4, "ci_lo": 0.3, "ci_hi": 0.5}
    n = rp.normalized(c, 1.0, 1.0)
    assert (n["point"], n["ci_lo"], n["ci_hi"], n["denom"]) == (0.4, 0.3, 0.5, 1.0)


def test_normalized_divides_by_geometric_mean():
    n = rp.normalized({"point": 0.4, "ci_lo": 0.2, "ci_hi": 0.6}, 0.25, 1.0)
    assert n["denom"] == pytest.approx(0.5)
    assert (n["point"], n["ci_lo"], n["ci_hi"]) == pytest.approx((0.8, 0.4, 1.2))


@pytest.mark.parametrize("ca,cb", [(0.0, 0.9), (0.9, -0.1), (float("nan"), 0.5)])
def test_normalized_bad_ceiling_is_nan(ca, cb):
    n = rp.normalized({"point": 0.4, "ci_lo": 0.2, "ci_hi": 0.6}, ca, cb)
    assert all(math.isnan(n[k]) for k in ("point", "ci_lo", "ci_hi"))


# ----------------------------------------------------------------------------- synthetic tree


def pe_of(base: np.ndarray, seed: int, noise: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (base[None, :] + noise * rng.normal(size=(N, E))).astype(np.float32)


def scores_of(pe: np.ndarray) -> dict:
    per = torch.as_tensor(pe)
    h = torch.as_tensor(HALF)
    return {"mean": per.mean(0), "mean_a": per[h == 0].mean(0), "mean_b": per[h == 1].mean(0),
            "per_example": per.half()}  # fmt: skip


def make_models(seed: int = 0) -> dict[str, np.ndarray]:
    """ep_* and ec share a strong circuit; fp and tc do not share one; base is unrelated."""
    rng = np.random.default_rng(seed)
    elicit_circ = np.zeros(E)
    elicit_circ[rng.choice(E, 100, replace=False)] = np.linspace(
        12.0, 4.0, 100
    )  # graded: stable top-k
    return {
        "ep_sym": pe_of(elicit_circ, 1),
        "ep_word": pe_of(elicit_circ, 2),
        "ec": pe_of(elicit_circ, 3),
        "fp": pe_of(rng.normal(size=E) * 2, 4),
        "tc": pe_of(rng.normal(size=E) * 2, 5),
        "base": pe_of(rng.normal(size=E) * 2, 6),
    }


def func_json(labs):
    sizes = {}
    for f in FRACS:
        fs = {"own": 0.9} | {lab: 0.5 for lab in labs}
        sizes[str(f)] = {"k": 10, "f": fs, "f_random_band": [0.0, 0.01, 0.02],
                         "share_of_own_f": {lab: v / 0.9 for lab, v in fs.items()},
                         "overlap_with_own": {lab: 0.3 for lab in fs}}  # fmt: skip
    return {"sizes": sizes}


def mask(rec):
    return {"recovery": rec, "recovery_ci95": [rec - 0.05, rec + 0.05], "acc2": 0.5}


def cmap_json(extras, rec):
    sizes = {}
    for f in FRACS:
        m = {"child": mask(rec + 0.1), "child_nologits": mask(rec)}
        for lab in extras:
            m[lab], m[f"{lab}_nologits"] = mask(rec / 2), mask(rec / 3)
        band = {"recovery_band": [0.0, 0.05, 0.1]}
        sizes[str(f)] = {"k": 10, "masks": m, "random_typed_disjoint": band, "random_typed": band,
                         "random_nologits": band}  # fmt: skip
    return {"baseline": {"kl_child_parent_mean": 2.0}, "all": {"recovery": 0.99},
            "checks_pass": True, "sizes": sizes}  # fmt: skip


def write_tree(root: Path, models: dict, ems=None, repro_pass=True, n_boot=60) -> None:
    ems = ems or {}
    dirs = {}
    for label, (tag, task) in rp.SETS.items():
        d = root / "scores" / label / tag
        d.mkdir(parents=True)
        s = scores_of(models[label])
        torch.save({k: s[k] for k in ("mean", "mean_a", "mean_b")}, d / "scores.pt")
        torch.save(s["per_example"], d / "per_example.pt")
        em = ems.get(label, 0.97 if label in rp.CHILDREN else 0.0)
        (d / "sanity.json").write_text(json.dumps({"exact_match": em}))
        if task == "word":
            (d / "repro.json").write_text(json.dumps({"max_abs_diff": 1e-6, "pass": repro_pass}))
        dirs[label] = d
    pairs = [tuple(k.split(":")) for _, r, c in rp.ROUTES for k in (r, c)]
    pairs = list(dict.fromkeys(pairs + [tuple(k.split(":")) for k in rp.OTHER_PAIRS]))
    res = ob.bootstrap_overlap(ob.load_scores(dirs), HALF, pairs, FRACS, n_boot, 0, chunk=700)
    (root / "overlap").mkdir()
    (root / "overlap" / "overlap_boot.json").write_text(json.dumps(res))
    (root / "functional").mkdir()
    for child, labs in rp.FUNC.items():
        (root / "functional" / f"{child}.json").write_text(json.dumps(func_json(labs)))
    (root / "cmap").mkdir()
    for route, extras in rp.CMAP_EXTRA.items():
        (root / "cmap" / f"{route}.json").write_text(
            json.dumps(cmap_json(extras, 0.8 if route == "elicit" else 0.2))
        )
    for form in ("word", "symbol"):
        (root / "hidden_pref" / form).mkdir(parents=True)
        for tag in rp.HP_TAGS:
            (root / "hidden_pref" / form / f"{tag}.json").write_text(json.dumps({
                "paper_logit_diff_mean": 0.1, "paper_pos_frac": 0.55, "full_answer_logp_diff_mean": 0.2,
                "full_answer_pos_frac": 0.6, "top1_c1": 0.01, "n": 1024}))  # fmt: skip


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("hard_acc")
    write_tree(root, make_models())
    return root


def test_end_to_end_complete_tree(tree, tmp_path):
    rc = rp.main(["--root", str(tree), "--fracs", *map(str, FRACS),
                  "--out-json", str(tmp_path / "r.json"), "--out-md", str(tmp_path / "r.md")])  # fmt: skip
    assert rc == 0
    rep = json.loads((tmp_path / "r.json").read_text())
    md = (tmp_path / "r.md").read_text()
    assert rep["missing"] == [] and rep["gates"]["all_pass"]
    for head in ("## Score sets", "## Edge overlap", "### Route margin", "## Functional reuse",
                 "## CMAP", "## Hidden preference"):  # fmt: skip
        assert head in md
    assert "MISSING" not in md
    assert set(rep["overlap"]["routes"]) == {name for name, _, _ in rp.ROUTES}
    assert set(rep["overlap"]["pairs"]) == {
        "ep_sym:ec",
        "ep_word:ec",
        "fp:tc",
        "base:ec",
        "base:tc",
        "ec:tc",
    }


def test_planted_shared_circuit_is_reuse_and_unrelated_is_not(tree):
    rep = rp.build(tree, FRACS)
    routes = rep["overlap"]["routes"]
    for f in map(str, FRACS):
        assert routes["elicit, parent symbol circuit"]["sizes"][f]["reuse"]
        assert routes["elicit, parent word circuit"]["sizes"][f]["reuse"]
        assert routes["elicit, parent symbol circuit"]["sizes"][f]["point"] > 0.8
        # fp and tc are unrelated: only chance overlap (a prompt bootstrap cannot see that
        # noise, so `reuse` itself may flip on a ~0.01 margin; the size is what is pinned)
        t = routes["teach, format parent"]["sizes"][f]
        assert abs(t["point"]) < 0.05 and t["ci_hi"] < 0.1


def test_route_margin_equals_pair_points_difference(tree):
    rep = rp.build(tree, FRACS)
    ob_json = json.loads((tree / "overlap" / "overlap_boot.json").read_text())["pairs"]
    for name, rkey, ckey in rp.ROUTES:
        for f in map(str, FRACS):
            m = rep["overlap"]["routes"][name]["sizes"][f]
            assert m["point"] == pytest.approx(
                ob_json[rkey][f]["point"] - ob_json[ckey][f]["point"]
            )
            assert m["ci_lo"] <= m["ci_hi"]


def test_ceilings_match_ceiling_report_and_normalization_uses_them(tree):
    rep = rp.build(tree, FRACS)
    for label, (tag, _) in rp.SETS.items():
        sc = torch.load(tree / "scores" / label / tag / "scores.pt")
        assert rep["sets"][label]["ceiling"] == cr.ceilings(sc["mean_a"], sc["mean_b"], FRACS)
    f = str(FRACS[0])
    c = rep["overlap"]["pairs"]["ep_sym:ec"][f]
    denom = math.sqrt(rep["sets"]["ep_sym"]["ceiling"][f] * rep["sets"]["ec"]["ceiling"][f])
    assert c["normalized"]["point"] == pytest.approx(c["point"] / denom)


def test_sections_carry_inputs(tree):
    rep = rp.build(tree, FRACS)
    fe = rep["functional"]["elicit_child"]["0.01"]
    assert (
        fe["f_own"] == 0.9
        and set(fe["f"]) == {"ep_sym", "ep_word"}
        and fe["above_random_p95"]["ep_sym"]
    )
    assert set(rep["functional"]["teach_child"]["0.01"]["f"]) == {"fp_word"}
    ce = rep["cmap"]["elicit"]["sizes"]["0.05"]
    assert (
        ce["child_nologits"]["recovery"] == 0.8
        and ce["above_disjoint_p95"]
        and "ep_sym_nologits" in ce
    )
    ct = rep["cmap"]["teach"]["sizes"]["0.05"]
    assert ct["child_nologits"]["recovery"] == 0.2 and ct["above_disjoint_p95"]  # CI lo 0.15 > 0.1
    assert not any(k.startswith("ep_sym") for k in ct)
    assert set(rep["hidden_pref"]) == {"word", "symbol"} and set(rep["hidden_pref"]["word"]) == set(
        rp.HP_TAGS
    )


def test_missing_files_listed_exit_1_and_rest_still_reported(tmp_path):
    root = tmp_path / "t"
    root.mkdir()
    write_tree(root, make_models(1), n_boot=20)
    (root / "cmap" / "teach.json").unlink()
    (root / "scores" / "ec" / "elicit_child" / "repro.json").unlink()
    out = tmp_path / "r.json"
    assert rp.main(["--root", str(root), "--fracs", *map(str, FRACS), "--out-json", str(out)]) == 1
    rep = json.loads(out.read_text())
    assert rep["missing"] == ["cmap/teach.json", "scores/ec/elicit_child/repro.json"]
    assert "teach" not in rep["cmap"] and "elicit" in rep["cmap"]
    assert rep["gates"]["ec_repro"] is None and not rep["gates"]["all_pass"]
    assert "**MISSING:** cmap/teach.json" in rp.markdown(rep)


def test_no_overlap_file_gives_none_section(tmp_path):
    root = tmp_path / "t"
    root.mkdir()
    write_tree(root, make_models(2), n_boot=20)
    (root / "overlap" / "overlap_boot.json").unlink()
    rep = rp.build(root, FRACS)
    assert rep["overlap"] is None and "overlap/overlap_boot.json" in rep["missing"]
    assert "## Edge overlap" not in rp.markdown(rep)


@pytest.mark.parametrize("ems,repro,key", [({"ec": 0.85}, True, "ec_em_ge_0.9"),
                                          ({"tc": 0.5}, True, "tc_em_ge_0.9"),
                                          ({}, False, "base_repro")])  # fmt: skip
def test_gates_fail_on_low_child_em_or_failed_repro(tmp_path, ems, repro, key):
    root = tmp_path / "t"
    root.mkdir()
    write_tree(root, make_models(3), ems=ems, repro_pass=repro, n_boot=20)
    g = rp.build(root, FRACS)["gates"]
    assert g[key] is False and not g["all_pass"]


def test_symbol_set_has_no_repro_gate(tree):
    g = rp.build(tree, FRACS)["gates"]
    assert "ep_sym_repro" not in g and "ep_word_repro" in g
