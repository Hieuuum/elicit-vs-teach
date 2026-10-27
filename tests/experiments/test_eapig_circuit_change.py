"""Tests for `experiments/eapig-circuit-check/circuit_change.py` (contract C4: score-set
dirs as arguments, cross-set pairs, legacy four-tag mode unchanged).

Uses the real 195,865-edge graph metadata with random synthetic `scores.pt` files in
`tmp_path`. CPU only; matplotlib Agg. Runs are module-scoped to keep the suite cheap.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eapig-circuit-check"
sys.path.insert(0, str(EXP))  # circuit_change imports `compare` as a sibling module


def _load():
    spec = importlib.util.spec_from_file_location("eapig_circuit_change", EXP / "circuit_change.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cc = _load()
E = 195_865
FRAC = 0.02
K = cc.et.size_to_k(FRAC, E)
TAGS = cc.TAGS


def _write_scores(d: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    a, b = (rng.standard_normal(E).astype(np.float32) for _ in range(2))
    torch.save(
        {
            "mean": torch.from_numpy((a + b) / 2),
            "mean_a": torch.from_numpy(a),
            "mean_b": torch.from_numpy(b),
        },
        d / "scores.pt",
    )
    return d


def _out_files(out: Path) -> list[str]:
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())


# ------------------------------------------------------------------ parsing


def test_parse_scores_specs_odd_labels():
    got = cc.parse_scores_specs(["elicit_parent@ldsym=/x/y", "a_b=rel/dir", "p=/with=eq"])
    assert got == {
        "elicit_parent@ldsym": Path("/x/y"),
        "a_b": Path("rel/dir"),
        "p": Path("/with=eq"),  # only the first '=' splits
    }
    assert list(got) == ["elicit_parent@ldsym", "a_b", "p"]  # order kept


@pytest.mark.parametrize("bad", ["nolabel", "=dir", "label=", "", "a:b=dir", "a/b=dir"])
def test_parse_scores_specs_malformed(bad):
    with pytest.raises(ValueError):
        cc.parse_scores_specs([bad])


def test_parse_scores_specs_duplicate_label():
    with pytest.raises(ValueError, match="duplicate"):
        cc.parse_scores_specs(["a=x", "a=y"])


def test_default_pairs_for_four_tags():
    pairs = cc.resolve_pairs(None, TAGS)
    assert pairs == {
        "elicit": ("elicit_parent", "elicit_child"),
        "teach": ("fmt_parent", "teach_child"),
        "children": ("elicit_child", "teach_child"),
        "parents": ("elicit_parent", "fmt_parent"),
    }
    assert list(pairs) == ["elicit", "teach", "children", "parents"]


def test_default_pairs_only_where_both_tags_exist():
    assert list(cc.resolve_pairs(None, ["elicit_parent", "elicit_child"])) == ["elicit"]
    with pytest.raises(ValueError, match="--pairs"):
        cc.resolve_pairs(None, ["x", "y"])


def test_explicit_pairs_keyed_by_spec_and_validated():
    labels = ["p@sym", "c_w", "c@sym"]
    got = cc.resolve_pairs(["p@sym:c_w", "p@sym:c@sym"], labels)
    assert got == {"p@sym:c_w": ("p@sym", "c_w"), "p@sym:c@sym": ("p@sym", "c@sym")}
    with pytest.raises(ValueError, match="unknown label"):
        cc.resolve_pairs(["p@sym:nope"], labels)
    for bad in ("p@sym", "p@sym:", ":c_w", "a:b:c"):
        with pytest.raises(ValueError):
            cc.resolve_pairs([bad], labels)


def test_main_errors_exit_nonzero(tmp_path):
    d = _write_scores(tmp_path / "a", 0)
    out = str(tmp_path / "o")
    with pytest.raises(SystemExit):  # unknown label in a pair
        cc.main(["--out", out, "--scores", f"a={d}", "--pairs", "a:zzz"])
    with pytest.raises(SystemExit):  # malformed spec
        cc.main(["--out", out, "--scores", "broken"])
    with pytest.raises(SystemExit):  # --scores without --out
        cc.main(["--scores", f"a={d}", "--pairs", "a:a"])
    with pytest.raises(SystemExit):  # --pairs without --scores
        cc.main(["--pairs", "a:b"])


# ---------------------------------------------------------------- cross-set


@pytest.fixture(scope="module")
def cross(tmp_path_factory):
    base = tmp_path_factory.mktemp("cross")
    p = _write_scores(base / "p", 1)
    c = _write_scores(base / "c", 2)
    twin = base / "twin"
    twin.mkdir()
    (twin / "scores.pt").write_bytes((p / "scores.pt").read_bytes())
    out = base / "out"
    cc.main(
        [
            "--frac",
            str(FRAC),
            "--out",
            str(out),
            "--scores",
            f"parent@ldsym={p}",
            f"child_w={c}",
            f"twin@x_1={twin}",
            "--pairs",
            "parent@ldsym:child_w",
            "parent@ldsym:twin@x_1",
        ]
    )
    return json.loads((out / f"circuit_change_{FRAC:g}.json").read_text()), out


def test_cross_set_output_names(cross):
    _, out = cross
    assert _out_files(out) == [
        "circuit_change_0.02.json",
        "figures/cc_layer_added_minus_dropped_0.02.png",
        "figures/cc_mass_concentration_0.02.png",
        "figures/cc_rank_in_parent_0.02.png",
    ]


def test_cross_set_structure(cross):
    res, _ = cross
    assert res["k"] == K and res["n_edges"] == E and res["frac"] == FRAC
    assert list(res["pairs"]) == ["parent@ldsym:child_w", "parent@ldsym:twin@x_1"]
    # reliability / shape exist per label (from its own halves), no role-based extras
    assert set(res["reliability"]) == set(res["shape"]) == {"parent@ldsym", "child_w", "twin@x_1"}
    assert "compare_json_check" not in res
    pr = res["pairs"]["parent@ldsym:child_w"]
    assert (pr["a"], pr["b"]) == ("parent@ldsym", "child_w")
    hm = pr["how_much"]
    assert hm["n_kept"] + hm["n_added"] == K and hm["n_kept"] + hm["n_dropped"] == K
    assert hm["edge_jaccard"] < 0.2  # independent random scores: near chance


def test_identical_score_sets_give_maximum_overlap(cross):
    res, _ = cross
    hm = res["pairs"]["parent@ldsym:twin@x_1"]["how_much"]
    assert hm["edge_jaccard"] == 1.0
    assert hm["abs_topk_jaccard"] == 1.0
    assert hm["spearman_full_vectors"] == pytest.approx(1.0)
    assert (hm["n_kept"], hm["n_added"], hm["n_dropped"]) == (K, 0, 0)
    shift = res["pairs"]["parent@ldsym:twin@x_1"]["what"]["b_top1000_rank_shift"]
    assert shift["share_in_a_circuit"] == 1.0
    assert shift["quantiles"]["0.5"] == 0.0


def test_reliability_per_label_matches_split_half(cross):
    res, _ = cross
    # random halves: Spearman ~ 0 so the ceiling is near chance, but fields are present
    rel = res["reliability"]["child_w"]
    assert abs(rel["spearman_halves"]) < 0.02
    assert 0.0 <= rel["ceiling_jaccard"] < 0.2
    assert set(rel["sign_agreement_halves"]) == {"100", "1000", str(K)}


# ------------------------------------------------------------------- legacy


@pytest.fixture(scope="module")
def legacy(tmp_path_factory):
    base = tmp_path_factory.mktemp("legacy")
    for i, t in enumerate(TAGS):
        _write_scores(base / t, 10 + i)
    # compare.json as the box would have written it: the edge Jaccard of each pair
    mean = {t: torch.load(base / t / "scores.pt")["mean"].double().numpy() for t in TAGS}
    top = {t: cc.et.topk_edges(mean[t], K).tolist() for t in TAGS}

    def jac(pair):
        return {"sizes": {str(FRAC): {"edge_jaccard": cc.et.jaccard(*(top[x] for x in pair))}}}

    cmp = {
        "routes": {"elicit": jac(cc.PAIRS["elicit"]), "teach": jac(cc.PAIRS["teach"])},
        "children_reference": jac(cc.PAIRS["children"]),
    }
    (base / "compare.json").write_text(json.dumps(cmp))
    out = base / "out"
    cc.main(["--results-dir", str(base), "--frac", str(FRAC), "--out", str(out)])
    return json.loads((out / f"circuit_change_{FRAC:g}.json").read_text()), out


def test_legacy_no_scores_path(legacy):
    res, out = legacy
    assert list(res["pairs"]) == ["elicit", "teach", "children", "parents"]
    assert set(res["reliability"]) == set(TAGS)
    assert set(res["compare_json_check"]) == {"elicit", "teach", "children"}
    for v in res["compare_json_check"].values():
        assert v["compare_json"] == v["here"]
    assert _out_files(out) == [
        "circuit_change_0.02.json",
        "figures/cc_layer_added_minus_dropped_0.02.png",
        "figures/cc_mass_concentration_0.02.png",
        "figures/cc_rank_in_parent_0.02.png",
    ]


def test_scores_with_four_tags_matches_legacy_pairs(legacy, tmp_path):
    """The four tags via --scores and no --pairs give the legacy pair set and numbers."""
    res, _ = legacy
    base = legacy[1].parent
    out = tmp_path / "o"
    cc.main(
        ["--frac", str(FRAC), "--out", str(out), "--scores", *(f"{t}={base / t}" for t in TAGS)]
    )
    new = json.loads((out / f"circuit_change_{FRAC:g}.json").read_text())
    assert "compare_json_check" not in new
    new.pop("compare_json_check", None)
    res = {k: v for k, v in res.items() if k != "compare_json_check"}
    assert new == res
