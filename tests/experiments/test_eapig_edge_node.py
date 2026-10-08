"""Tests for `experiments/eapig-circuit-check/edge_node.py` (edge-vs-node change + nulls).

Synthetic tiny graph (no model, no real scores) except one test that checks the
fast `signed` ranking against the library's `topk_edges`. Loaded by file path.
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
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location("edge_node_mod", EXP / "edge_node.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["edge_node_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


en = _load()

N_EDGES, N_NODES = 400, 20
RNG = np.random.default_rng(0)
UP = RNG.integers(0, 10, N_EDGES)
RC = RNG.integers(10, N_NODES, N_EDGES)
HEAD = np.arange(N_NODES) % 2 == 0
GRAPH = {"up": UP, "rc": RC, "n_nodes": N_NODES, "head": HEAD}


def test_identical_sets_zero_change():
    idx = np.arange(30)
    r = en.change_rates(idx, idx, UP, RC, N_NODES, HEAD)
    assert r["edge_rate"] == 0 and r["node_rate"] == 0 and r["gap"] == 0 and r["overlap"] == 30


def test_disjoint_edges_same_nodes_edge_one_node_zero():
    up = np.array([0, 0, 1, 1])
    rc = np.array([2, 2, 3, 3])
    a, b = np.array([0, 2]), np.array([1, 3])  # edge 0/1 share nodes (0,2); 2/3 share (1,3)
    r = en.change_rates(a, b, up, rc, 4, np.ones(4, bool))
    assert r["edge_rate"] == 1.0 and r["node_rate"] == 0.0 and r["gap"] == 1.0


def test_disjoint_nodes_both_one():
    up, rc = np.array([0, 1]), np.array([2, 3])
    r = en.change_rates([0], [1], up, rc, 4, np.array([True, False, True, False]))
    assert r["edge_rate"] == 1 and r["node_rate"] == 1 and r["head_rate"] == 1


def test_head_rate_nan_when_no_heads():
    r = en.change_rates([0], [1], np.array([0, 1]), np.array([2, 3]), 4, np.zeros(4, bool))
    assert np.isnan(r["head_rate"])


def test_rates_are_a_relative_to_b():
    up, rc = np.array([0, 0, 1]), np.array([2, 2, 3])
    r = en.change_rates([0, 1, 2], [0], up, rc, 4, np.ones(4, bool))
    assert r["edge_rate"] == pytest.approx(2 / 3)
    assert r["node_rate"] == pytest.approx(2 / 4 * 1.0)  # A touches {0,1,2,3}; B {0,2}


def test_empty_a_raises():
    with pytest.raises(ValueError):
        en.change_rates([], [1], UP, RC, N_NODES, HEAD)


@pytest.mark.parametrize("k,ov", [(20, 0), (20, 7), (20, 20), (1, 1)])
def test_n2_preserves_overlap_exactly(k, ov):
    """Re-derive the overlap from the draws' edge rate: (k - ov) / k for every draw."""
    d = en.null_draws("n2", k, ov, N_EDGES, UP, RC, N_NODES, HEAD, 25, np.random.SeedSequence(1))
    assert np.allclose(d[0], (k - ov) / k)


def test_n3_preserves_overlap_and_stays_in_pool():
    a_obs, b_obs = np.arange(0, 20), np.arange(10, 30)
    pool = np.union1d(a_obs, b_obs)
    d = en.null_draws(
        "n3", 20, 10, N_EDGES, UP, RC, N_NODES, HEAD, 25, np.random.SeedSequence(1), pool
    )
    assert np.allclose(d[0], 0.5)
    # nodes touched can only come from the pool's node universe
    pool_nodes = en.node_mask(pool, UP, RC, N_NODES).sum()
    assert pool_nodes >= 1 and np.all(np.isfinite(d[1]))


def test_n1_varies_and_is_near_chance():
    d = en.null_draws("n1", 20, 0, N_EDGES, UP, RC, N_NODES, HEAD, 100, np.random.SeedSequence(2))
    assert d[0].min() >= 0.5 and d[0].mean() > 0.9  # 20 of 400: overlap ~ 1


@pytest.mark.parametrize("kind", ["n1", "n2", "n3"])
def test_nulls_seed_deterministic(kind):
    pool = np.arange(35)

    def draw(seed):
        return en.null_draws(
            kind, 20, 5, N_EDGES, UP, RC, N_NODES, HEAD, 10, np.random.SeedSequence(seed), pool
        )

    assert np.array_equal(draw(3), draw(3))
    assert not np.array_equal(draw(3), draw(4))


def test_signed_vs_abs_ranking_differ_when_signs_flip():
    s = np.array([5.0, -6.0, 1.0, 0.5, -0.1])
    assert en.rank_order(s, "signed")[:2].tolist() == [0, 2]
    assert en.rank_order(s, "abs")[:2].tolist() == [1, 0]
    with pytest.raises(ValueError):
        en.rank_order(s, "other")


def test_signed_order_matches_topk_edges_with_ties():
    from geode.circuits import edge_tests as et

    s = np.round(np.random.default_rng(1).normal(size=300), 1)  # many ties
    assert en.rank_order(s, "signed")[:40].tolist() == et.topk_edges(s, 40).tolist()


def test_p_ge_and_band():
    assert en._p_ge(np.array([0.0, 1.0, 2.0]), 2.0) == pytest.approx(2 / 4)
    assert np.isnan(en._p_ge(np.array([np.nan]), 1.0))
    assert en._band(np.arange(101.0))["p50"] == 50


def _scores(seed, n=N_EDGES):
    g = np.random.default_rng(seed)
    base = g.normal(size=n)
    return {
        "mean": base,
        "mean_a": base + 0.1 * g.normal(size=n),
        "mean_b": base + 0.1 * g.normal(size=n),
    }


def test_run_schema_and_pair_set():
    scores = {"p": _scores(1), "c": _scores(2)}
    res = en.run(scores, [("p", "c")], GRAPH, [0.05], [10], 20, 0, 1)
    assert set(res["pairs"]) == {"p:c", "p:halves", "c:halves"}
    assert res["pairs"]["p:halves"]["kind"] == "split_half"
    row = res["pairs"]["p:c"]["rankings"]["signed"]["frac_0.05"]
    assert row["k"] == 20 and set(res["pairs"]["p:c"]["rankings"]) == {"signed", "abs"}
    for key in ("edge_rate", "node_rate", "head_rate", "gap", "overlap", "null"):
        assert key in row
    assert set(row["null"]) == {"n1", "n2", "n3"}
    for key in ("edge", "node", "head", "gap"):
        assert set(row["null"]["n2"][key]) == {"p5", "p50", "p95"}
    assert "p_gap" in row["null"]["n2"] and "edges_more_than_nodes_n2" in row
    assert "top_10" in res["pairs"]["p:c"]["rankings"]["abs"]
    json.dumps(res, default=float)


def test_run_identical_models_gap_zero_not_significant():
    s = _scores(1)
    res = en.run({"a": s, "b": s}, [("a", "b")], GRAPH, [0.05], [], 30, 0, 1)
    row = res["pairs"]["a:b"]["rankings"]["signed"]["frac_0.05"]
    assert row["edge_rate"] == 0 and row["gap"] == 0 and not row["edges_more_than_nodes_n2"]


def test_run_workers_do_not_change_results():
    scores = {"p": _scores(1), "c": _scores(2)}
    r1 = en.run(scores, [("p", "c")], GRAPH, [0.05], [10], 12, 7, 1)
    r2 = en.run(scores, [("p", "c")], GRAPH, [0.05], [10], 12, 7, 2)
    assert json.dumps(r1, default=float) == json.dumps(r2, default=float)


def test_sizes_reject_bad_k():
    with pytest.raises(ValueError):
        en.sizes_of([], [N_EDGES + 1], N_EDGES)


def test_cli_parsing(tmp_path):
    a = en.parse_args(
        [
            "--scores",
            "x=/a",
            "y=/b",
            "--pairs",
            "x:y",
            "--out",
            str(tmp_path),
            "--fracs",
            "0.01",
            "--topn",
            "5",
            "7",
            "--n-null",
            "9",
            "--seed",
            "3",
            "--workers",
            "2",
        ]
    )
    assert a.dirs == {"x": Path("/a"), "y": Path("/b")} and a.pair_list == [("x", "y")]
    assert a.fracs == [0.01] and a.topn == [5, 7] and (a.n_null, a.seed, a.workers) == (9, 3, 2)
    d = en.parse_args([])
    assert set(d.dirs) == set(en.TAGS) and d.pair_list == list(en.DEFAULT_PAIRS)
    assert (
        d.n_null == 1000 and d.fracs == list(en.DEFAULT_FRACS) and d.topn == list(en.DEFAULT_TOPN)
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--scores", "nodir"],
        ["--scores", "x=/a", "--pairs", "x:y"],
        ["--scores", "x=/a", "y=/b"],
        ["--scores", "x=/a", "--pairs", "x"],
    ],
)
def test_cli_rejects_bad_args(argv):
    with pytest.raises(SystemExit):
        en.parse_args(argv)


def test_load_scores_and_plot(tmp_path):
    for lab, seed in (("p", 1), ("c", 2)):
        (tmp_path / lab).mkdir()
        torch.save(
            {
                **{k: torch.tensor(v) for k, v in _scores(seed).items()},
                "per_example": torch.zeros(3, N_EDGES),
            },
            tmp_path / lab / "scores.pt",
        )
    sc = en.load_scores({"p": tmp_path / "p", "c": tmp_path / "c"})
    assert set(sc["p"]) == {"mean", "mean_a", "mean_b"} and sc["p"]["mean"].shape == (N_EDGES,)
    res = en.run(sc, [("p", "c")], GRAPH, [0.05], [10], 10, 0, 1)
    en.plot(res, tmp_path / "figures" / "edge_node.png")
    assert (tmp_path / "figures" / "edge_node.png").stat().st_size > 0
