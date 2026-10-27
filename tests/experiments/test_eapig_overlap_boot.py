"""Property tests for `experiments/eapig-circuit-check/overlap_boot.py` (bootstrap edge-Jaccard).

Tiny synthetic scores.pt files (N prompts x E edges) with a fake `half`; CPU, no model.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from geode.circuits import edge_tests as et

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


ob = _load("overlap_boot")
edge_node = _load("edge_node")

N, E = 24, 300
HALF = np.tile([0, 1], N // 2)


def make_scores(per_example_fp32: np.ndarray, half: np.ndarray = HALF) -> dict:
    """scores.pt dict the way run.py's stage_score builds it (means from fp32, rows stored fp16)."""
    per = torch.as_tensor(per_example_fp32, dtype=torch.float32)
    h = torch.as_tensor(half)
    return {
        "mean": per.mean(0),
        "mean_a": per[h == 0].mean(0),
        "mean_b": per[h == 1].mean(0),
        "per_example": per.half(),
    }


def rand_pe(seed: int, n: int = N, e: int = E, signal: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=e) * signal
    return (base[None, :] + rng.normal(size=(n, e))).astype(np.float32)


def run_boot(scores, pairs, fracs=(0.05, 0.2), n_boot=30, seed=0, chunk=70):
    return ob.bootstrap_overlap(scores, HALF, pairs, list(fracs), n_boot, seed, chunk=chunk)


# ---------------------------------------------------------------- fast_topk_order


@pytest.mark.parametrize("seed", range(4))
def test_fast_topk_matches_topk_edges_random(seed):
    s = np.random.default_rng(seed).normal(size=200)
    for k in (1, 7, 50, 200):
        assert np.array_equal(ob.fast_topk_order(s, k), et.topk_edges(s, k))


def test_fast_topk_matches_topk_edges_heavy_ties():
    s = np.random.default_rng(0).integers(0, 4, size=500).astype(np.float64)
    for k in (1, 2, 100, 137, 499, 500):
        got = ob.fast_topk_order(s, k)
        assert np.array_equal(got, et.topk_edges(s, k))


def test_fast_topk_all_equal_and_negative():
    s = np.zeros(50)
    assert np.array_equal(ob.fast_topk_order(s, 10), np.arange(10))
    s = -np.arange(30, dtype=np.float64)[::-1] // 3  # negatives with ties
    for k in (1, 5, 30):
        assert np.array_equal(ob.fast_topk_order(s, k), et.topk_edges(s, k))


def test_fast_topk_float32_input_and_bad_k():
    s = np.random.default_rng(1).normal(size=40).astype(np.float32)
    assert np.array_equal(ob.fast_topk_order(s, 9), et.topk_edges(s, 9))
    for k in (0, 41):
        with pytest.raises(ValueError):
            ob.fast_topk_order(s, k)


# ---------------------------------------------------------------- resampling plumbing


def test_identity_counts_give_full_mean_with_nondividing_chunk():
    pe16 = torch.as_tensor(rand_pe(0)).half()
    counts = np.full((3, N), 1.0 / N, dtype=np.float32)
    full = pe16.float().mean(0).numpy()
    for chunk in (E, 70, 1, 299):  # 70 and 299 do not divide 300
        got = ob.resampled_means(counts, pe16, chunk)
        assert got.shape == (3, E)
        np.testing.assert_allclose(got, np.broadcast_to(full, got.shape), atol=1e-6)


def test_draw_counts_rows_sum_to_one_and_seeded():
    c = ob.draw_counts(50, N, 3)
    assert c.shape == (50, N)
    np.testing.assert_allclose(c.sum(1), 1.0, atol=1e-6)
    assert np.array_equal(c, ob.draw_counts(50, N, 3))
    assert not np.array_equal(c, ob.draw_counts(50, N, 4))
    assert ((c * N) % 1 < 1e-6).all()  # integer multiplicities


def test_bootstrap_orders_prefix_property():
    pe16 = torch.as_tensor(rand_pe(1)).half()
    counts = ob.draw_counts(5, N, 0)
    orders = ob.bootstrap_orders(counts, pe16, 40, chunk=70)
    means = ob.resampled_means(counts, pe16, 70)
    for i in range(5):
        for k in (1, 10, 40):
            assert np.array_equal(orders[i][:k], et.topk_edges(means[i].astype(np.float64), k))


def test_one_shared_draw_for_all_models(monkeypatch):
    seen = []
    real = ob.bootstrap_orders

    def spy(counts, pe, kmax, chunk=70):
        seen.append(counts)
        return real(counts, pe, kmax, chunk)

    monkeypatch.setattr(ob, "bootstrap_orders", spy)
    sc = {x: make_scores(rand_pe(i)) for i, x in enumerate("abc")}
    run_boot(sc, [("a", "b"), ("b", "c"), ("a", "c")])
    assert len(seen) == 3  # once per model, not once per pair
    assert all(c is seen[0] for c in seen)


# ---------------------------------------------------------------- overlap properties


def test_identical_models_jaccard_one_everywhere():
    pe = rand_pe(2)
    sc = {"a": make_scores(pe), "b": make_scores(pe)}
    res = run_boot(sc, [("a", "b")])["pairs"]["a:b"]
    for r in res.values():
        assert r["point"] == 1.0 and r["point_fp16_full"] == 1.0
        assert r["ci_lo"] == r["ci_hi"] == r["boot_mean"] == r["boot_median"] == 1.0
        assert r["point_in_ci"]


def test_disjoint_topk_gives_zero_in_every_resample():
    # A: edges 0..E/2 large, B: the mirror image; top 20% (k=60) of each lie in disjoint halves
    rng = np.random.default_rng(0)
    pe_a = rng.normal(size=(N, E)).astype(np.float32) * 0.1
    pe_a[:, : E // 2] += 10.0
    pe_b = rng.normal(size=(N, E)).astype(np.float32) * 0.1
    pe_b[:, E // 2 :] += 10.0
    res = run_boot({"a": make_scores(pe_a), "b": make_scores(pe_b)}, [("a", "b")])["pairs"]["a:b"]
    for r in res.values():
        assert r["point"] == 0.0
        assert r["ci_lo"] == r["ci_hi"] == r["boot_mean"] == 0.0


def test_deterministic_per_seed_and_seed_changes_boot():
    sc = {"a": make_scores(rand_pe(3, signal=0.3)), "b": make_scores(rand_pe(4, signal=0.3))}
    r1 = run_boot(sc, [("a", "b")], n_boot=40, seed=7)
    r2 = run_boot(sc, [("a", "b")], n_boot=40, seed=7)
    assert json.dumps(r1) == json.dumps(r2)
    r3 = run_boot(sc, [("a", "b")], n_boot=40, seed=8)
    assert r1["pairs"] != r3["pairs"]
    # point estimate does not depend on the seed
    for f in r1["pairs"]["a:b"]:
        assert r1["pairs"]["a:b"][f]["point"] == r3["pairs"]["a:b"][f]["point"]


def test_interval_brackets_point_on_noisy_case():
    n = 120
    half = np.tile([0, 1], n // 2)
    base = 2 * np.random.default_rng(5).normal(size=E)
    rng = np.random.default_rng(6)
    pe_a = (base + 3 * rng.normal(size=(n, E))).astype(np.float32)
    pe_b = (base + 3 * rng.normal(size=(n, E))).astype(np.float32)
    sc = {"a": make_scores(pe_a, half), "b": make_scores(pe_b, half)}
    res = ob.bootstrap_overlap(sc, half, [("a", "b")], [0.1, 0.3], 100, 0, chunk=70)["pairs"]["a:b"]
    for r in res.values():
        assert 0.0 < r["ci_lo"] < r["ci_hi"] < 1.0
        assert r["boot_mean"] <= r["point"] + 0.02  # resampling adds noise: never inflates overlap
        assert r["ci_lo"] <= r["point"] <= r["ci_hi"] and r["point_in_ci"]
        assert r["pct_lo"] <= r["boot_median"] <= r["pct_hi"]
        # primary interval = basic bootstrap: the percentile interval reflected about the point
        assert r["ci_lo"] == pytest.approx(max(0.0, 2 * r["point"] - r["pct_hi"]))
        assert r["ci_hi"] == pytest.approx(min(1.0, 2 * r["point"] - r["pct_lo"]))
        assert r["bias"] == pytest.approx(r["boot_mean"] - r["point"]) and r["bias"] <= 0.02
        assert r["n_boot"] == 100


def test_basic_interval_corrects_downward_bias():
    # very noisy per-prompt scores: resampled overlap sits below the point (negative bias);
    # the basic interval must then lie on or above the percentile interval, never below it
    n = 120
    half = np.tile([0, 1], n // 2)
    base = 2 * np.random.default_rng(5).normal(size=E)
    rng = np.random.default_rng(6)
    pe_a = (base + 6 * rng.normal(size=(n, E))).astype(np.float32)
    pe_b = (base + 6 * rng.normal(size=(n, E))).astype(np.float32)
    sc = {"a": make_scores(pe_a, half), "b": make_scores(pe_b, half)}
    res = ob.bootstrap_overlap(sc, half, [("a", "b")], [0.1, 0.3], 100, 0, chunk=70)["pairs"]["a:b"]
    for r in res.values():
        assert r["bias"] < 0
        assert r["ci_lo"] >= r["pct_lo"] and r["ci_hi"] >= r["pct_hi"]
        assert 0.0 <= r["ci_lo"] <= r["ci_hi"] <= 1.0


def test_point_equals_edge_node_signed_overlap_jaccard():
    pe_a, pe_b = rand_pe(7, signal=0.5), rand_pe(8, signal=0.5)
    pe_a[:, 6] = pe_a[:, 5]  # exact ties in the mean, to exercise tie-breaking
    pe_b[:, 11] = pe_b[:, 10]
    sc = {"a": make_scores(pe_a), "b": make_scores(pe_b)}
    res = run_boot(sc, [("a", "b")], fracs=(0.05, 0.1, 0.5))["pairs"]["a:b"]
    up = rc = np.arange(E)
    head = np.zeros(E, dtype=bool)
    a = sc["a"]["mean"].double().numpy()
    b = sc["b"]["mean"].double().numpy()
    oa, ob_ = edge_node.rank_order(a, "signed"), edge_node.rank_order(b, "signed")
    for f, r in res.items():
        k = r["k"]
        ov = edge_node.change_rates(oa[:k], ob_[:k], up, rc, E, head)["overlap"]
        assert r["point"] == pytest.approx(ov / (2 * k - ov), abs=1e-12)
        assert r["k"] == et.size_to_k(float(f), E)


def test_fp16_full_point_close_to_fp32_point():
    sc = {"a": make_scores(rand_pe(9)), "b": make_scores(rand_pe(10))}
    for r in run_boot(sc, [("a", "b")])["pairs"]["a:b"].values():
        assert abs(r["point"] - r["point_fp16_full"]) < 0.25


def test_pairs_share_model_resamples_symmetry():
    sc = {x: make_scores(rand_pe(i)) for i, x in enumerate("ab")}
    r = run_boot(sc, [("a", "b"), ("b", "a")])["pairs"]
    for f in r["a:b"]:
        assert r["a:b"][f] == r["b:a"][f]


# ---------------------------------------------------------------- same-prompts assertion


def test_same_prompts_ok_and_returns_fp16_full_mean():
    s = make_scores(rand_pe(11))
    full = ob.check_same_prompts("m", s, HALF)
    np.testing.assert_allclose(full, s["mean"].numpy(), atol=5e-3)


def test_same_prompts_fires_on_permuted_rows():
    s = make_scores(rand_pe(12))
    perm = np.random.default_rng(0).permutation(N)
    s["per_example"] = s["per_example"][torch.as_tensor(perm)]
    with pytest.raises(AssertionError, match="badmodel"):
        ob.check_same_prompts("badmodel", s, HALF)
    # also surfaces through the full pipeline
    good = make_scores(rand_pe(12))
    with pytest.raises(AssertionError, match="b"):
        run_boot({"a": good, "b": s}, [("a", "b")])


def test_same_prompts_fires_on_other_prompts_mean_mismatch():
    s = make_scores(rand_pe(13))
    s["mean"] = s["mean"] + 0.5
    with pytest.raises(AssertionError, match="mean"):
        ob.check_same_prompts("m", s, HALF)


def test_same_prompts_fires_on_half_length_and_n_mismatch():
    s = make_scores(rand_pe(14))
    with pytest.raises(AssertionError, match="rows"):
        ob.check_same_prompts("m", s, HALF[:-2])
    other = make_scores(rand_pe(15, n=N - 2), half=HALF[:-2])
    with pytest.raises(AssertionError, match="shapes differ"):
        run_boot({"a": s, "b": other}, [("a", "b")])


# ---------------------------------------------------------------- CLI


def _write(tmp_path: Path, label: str, s: dict) -> Path:
    d = tmp_path / label
    d.mkdir()
    torch.save(s, d / "scores.pt")
    return d


def test_cli_missing_scores_label_errors(tmp_path):
    d = _write(tmp_path, "a", make_scores(rand_pe(0)))
    with pytest.raises(SystemExit):
        ob.parse_args(
            [
                "--scores",
                f"a={d}",
                "--pairs",
                "a:zzz",
                "--fracs",
                "0.1",
                "--data-dir",
                "x",
                "--out",
                str(tmp_path / "o"),
            ]
        )
    with pytest.raises(SystemExit):
        ob.parse_args(
            [
                "--scores",
                f"a={d}",
                "--pairs",
                "a-a",
                "--fracs",
                "0.1",
                "--data-dir",
                "x",
                "--out",
                str(tmp_path / "o"),
            ]
        )


def test_cli_writes_json_with_documented_keys(tmp_path, monkeypatch):
    da = _write(tmp_path, "a", make_scores(rand_pe(1)))
    db = _write(tmp_path, "b", make_scores(rand_pe(2)))
    _write(tmp_path, "unused", make_scores(rand_pe(3, n=N - 2), half=HALF[:-2]))  # not in any pair
    fake = type(sys)("data")
    fake.load_pairs = lambda path, task="word": {"disc": {"half": HALF}}
    monkeypatch.setitem(sys.modules, "data", fake)
    out = tmp_path / "out"
    ob.main(
        [
            "--scores",
            f"a={da}",
            f"b={db}",
            f"unused={tmp_path / 'unused'}",
            "--pairs",
            "a:b",
            "--fracs",
            "0.05",
            "0.2",
            "--n-boot",
            "20",
            "--seed",
            "1",
            "--data-dir",
            "x",
            "--out",
            str(out),
        ]
    )
    res = json.loads((out / "overlap_boot.json").read_text())
    assert set(res) == {"meta", "pairs"}
    assert {"seed", "n_boot", "N", "E", "fracs", "labels", "fp16_atol_frac"} <= set(res["meta"])
    assert res["meta"]["labels"] == ["a", "b"] and res["meta"]["N"] == N and res["meta"]["E"] == E
    cell = res["pairs"]["a:b"]["0.05"]
    assert set(cell) == {
        "k",
        "n_boot",
        "point",
        "point_fp16_full",
        "boot_mean",
        "boot_median",
        "bias",
        "ci_lo",
        "ci_hi",
        "pct_lo",
        "pct_hi",
        "point_in_ci",
    }
    assert cell["k"] == et.size_to_k(0.05, E) and cell["n_boot"] == 20
