"""Property tests for `experiments/eapig-circuit-check/cmap.py` (edge-level child -> parent
activation patching, HANDOFF-cmap.md "Correctness checks").

cmap.py does `import run` / `from run import et` by bare name, so the experiment directory is put
on sys.path just long enough to load it by file path; the `run` module it pulls in is then removed
from sys.modules again (and any previous `run` entry restored) so no other test is affected.
Everything runs on a tiny random-init Llama pair (2 layers, d=64, seeds 0 and 1, weights x4).
keep[u, r] == 0 means "edge carries the child's output of u into the parent's receiver r".
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from geode.circuits import eapig
from geode.circuits.eapig import EdgeGraph, build_graph, node_outputs, patched_forward

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eapig-circuit-check"


def _load_cmap():
    saved = sys.modules.pop("run", None)
    sys.path.insert(0, str(EXP))
    try:
        spec = importlib.util.spec_from_file_location("eapig_cmap", EXP / "cmap.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(EXP))
        sys.modules.pop("run", None)
        if saved is not None:
            sys.modules["run"] = saved
    return mod


cmap = _load_cmap()
run = cmap.run  # the experiment's run.py, held by reference

VOCAB = 50
T = 6
N_VAL = 10
BS = 4
FRACS = (0.1, 0.4)
N_RANDOM = 4


# ------------------------------------------------------------------ fixtures


def _tiny_model(seed: int) -> LlamaForCausalLM:
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, tie_word_embeddings=True)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = LlamaForCausalLM(cfg).float().eval()
        with torch.no_grad():  # std-0.02 init is near-uniform; scale up for real signal
            for p in model.parameters():
                if p.dim() > 1:
                    p.mul_(4.0)
    return model


@pytest.fixture(scope="module")
def parent():
    return _tiny_model(0)


@pytest.fixture(scope="module")
def child():
    return _tiny_model(1)


@pytest.fixture(scope="module")
def graph(parent):
    return build_graph(parent)


@pytest.fixture(scope="module")
def val(child):
    """Random prompts; for half of them c1/c2 are the child's argmax so acc2 is mixed."""
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(0, VOCAB - 1, (N_VAL, T), generator=g)
    with torch.no_grad():
        lg = child(ids).logits
    c1 = torch.randint(0, VOCAB, (N_VAL,), generator=g)
    c2 = torch.randint(0, VOCAB, (N_VAL,), generator=g)
    c1[::2], c2[::2] = lg[::2, -2].argmax(-1), lg[::2, -1].argmax(-1)
    return {"clean_ids": ids, "c1": c1, "c2": c2}


@pytest.fixture(scope="module")
def circuits(graph):
    g = torch.Generator().manual_seed(11)
    return {"child": torch.randn(graph.n_edges, generator=g),
            "ep": torch.randn(graph.n_edges, generator=g)}


@pytest.fixture(scope="module")
def full_run(parent, child, graph, val, circuits):
    res, per_ex = cmap.run_cmap(parent, child, graph, val, circuits, FRACS, N_RANDOM, BS)
    return res, per_ex


def _kl_hand(patched_logits, child_logits) -> torch.Tensor:
    """KL(child || patched) summed over the last two positions, float64, straight from torch."""
    p = torch.softmax(child_logits[:, -2:].double(), -1)
    lq = torch.log_softmax(patched_logits[:, -2:].double(), -1)
    return (p * (torch.log(p) - lq)).sum(-1).sum(-1)


def _random_keep(graph, seed: int, p_patch: float = 0.5) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return graph.valid & (torch.rand(graph.n_upstream, graph.n_receivers, generator=g) > p_patch)


# ------------------------------------------------- 1. all-ones keep == parent


def test_all_ones_keep_logits_equal_parent(parent, child, graph, val):
    ids = val["clean_ids"]
    with torch.no_grad():
        src = node_outputs(child, graph, ids)
        ref = parent(ids).logits
        for keep in (graph.valid, torch.ones_like(graph.valid)):
            out = patched_forward(parent, graph, ids, src, keep, last_n=2)
            assert torch.allclose(out, ref[:, -2:], atol=1e-5)


def test_eval_cmap_all_ones_kl_equals_kl_parent(parent, child, graph, val):
    ev = cmap.eval_cmap(parent, child, graph, val, [graph.valid, torch.zeros_like(graph.valid)], BS)
    assert torch.allclose(ev["kl"][0], ev["kl_parent"], atol=1e-5)
    assert (ev["kl_parent"] > 1e-3).all()  # parent and child really differ
    assert not torch.allclose(ev["kl"][1], ev["kl_parent"], atol=1e-3)
    assert torch.equal(ev["acc"][0], ev["acc_parent"])
    assert torch.allclose(ev["logp"][0], ev["logp_parent"], atol=1e-4)


# --------------------------------------- 2. all-zeros keep == lm_head(norm(sum child nodes))


def test_all_patched_logits_equal_parent_head_on_summed_child_nodes(parent, child, graph, val):
    ids = val["clean_ids"]
    with torch.no_grad():
        nodes_c = node_outputs(child, graph, ids)
        out = patched_forward(parent, graph, ids, nodes_c, torch.zeros_like(graph.valid), last_n=2)
        ref = parent.lm_head(parent.model.norm(nodes_c.sum(2)))[:, -2:]
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)
    # sanity: the sum of the child's nodes really is the child's final residual stream
    with torch.no_grad():
        assert torch.allclose(child.lm_head(child.model.norm(nodes_c.sum(2))),
                              child(ids).logits, rtol=1e-4, atol=1e-4)


# --------------------------------------------- 3. child == parent: nothing to recover


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_identical_models_patched_equals_parent_for_any_mask(parent, graph, val, seed):
    twin = copy.deepcopy(parent)
    ids = val["clean_ids"]
    keep = _random_keep(graph, seed, p_patch=0.2 + 0.3 * seed)
    with torch.no_grad():
        out = patched_forward(parent, graph, ids, node_outputs(twin, graph, ids), keep)
        assert torch.allclose(out, parent(ids).logits, atol=1e-5)


@pytest.mark.parametrize("kl_patched,kl_parent", [
    (0.3, 0.0), (0.0, 0.0), (1e-3, 1e-9), (5.0, 1e-8), (0.2, -1.0), (0.2, float("nan"))])
def test_recovery_is_nan_when_denominator_is_not_positive_enough(kl_patched, kl_parent):
    assert math.isnan(cmap.recovery(kl_patched, kl_parent))


@pytest.mark.parametrize("kl_patched,kl_parent,expect", [
    (0.0, 1.0, 1.0), (1.0, 1.0, 0.0), (0.25, 1.0, 0.75), (2.0, 1.0, -1.0), (0.1, 1e-4, -999.0)])
def test_recovery_formula(kl_patched, kl_parent, expect):
    assert cmap.recovery(kl_patched, kl_parent) == pytest.approx(expect)


def test_run_cmap_identical_models_gives_nan_not_finite_garbage(parent, graph, val, circuits):
    twin = copy.deepcopy(parent)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # nanpercentile on all-NaN slices
        res, _ = cmap.run_cmap(parent, twin, graph, val, circuits, (0.1,), 2, BS)
    assert res["baseline"]["kl_child_parent_mean"] < cmap.DENOM_TOL
    rows = [res["none"], res["all"], *res["sizes"]["0.1"]["masks"].values()]
    for row in rows:
        assert math.isnan(row["recovery"])
        assert all(math.isnan(x) for x in row["recovery_ci95"])
    s = res["sizes"]["0.1"]
    assert all(math.isnan(r) for r in s["random"]["recovery"] + s["random_nologits"]["recovery"])
    assert all(math.isnan(r["frac_of_all"]) for r in s["masks"].values())
    assert res["checks"]["child_self_patch_max_kl"] < cmap.CHECK_TOL


# -------------------------------------------------------------- 4. mask direction


@pytest.fixture(scope="module")
def last_mlp_child(parent):
    """Deep copy of the parent with ONLY the last layer's MLP weights perturbed."""
    ch = copy.deepcopy(parent)
    g = torch.Generator().manual_seed(77)
    with torch.no_grad():
        for p in ch.model.layers[-1].mlp.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.5)
    return ch


def test_patching_edges_from_unchanged_nodes_leaves_logits_unchanged(
        parent, last_mlp_child, graph, val):
    ids = val["clean_ids"][:3]
    m_last = graph.mlp_index(graph.n_layers - 1)
    with torch.no_grad():
        nodes_p = node_outputs(parent, graph, ids)
        nodes_c = node_outputs(last_mlp_child, graph, ids)
        same = (nodes_p - nodes_c).abs().amax(dim=(0, 1, 3))  # per node
        assert (same[:m_last] < 1e-6).all()  # embed, all heads, m0, a1.h*
        assert same[m_last] > 1e-2  # the perturbed node differs
        ref = parent(ids).logits[:, -2:]
        for e in range(graph.n_edges):
            u, r = graph.edge_ur[e].tolist()
            if u == m_last:
                continue
            keep = graph.valid.clone()
            keep[u, r] = False
            out = patched_forward(parent, graph, ids, nodes_c, keep, last_n=2)
            assert torch.allclose(out, ref, atol=1e-5), (u, r)


def test_patching_last_mlp_to_logits_changes_logits_as_hand_computed(
        parent, last_mlp_child, graph, val):
    ids = val["clean_ids"]
    m = graph.mlp_index(graph.n_layers - 1)
    # m_last has exactly one valid edge: into logits
    assert graph.valid[m].nonzero().squeeze(1).tolist() == [graph.logits_index]
    keep = graph.valid.clone()
    keep[m, graph.logits_index] = False
    with torch.no_grad():
        nodes_p = node_outputs(parent, graph, ids)
        nodes_c = node_outputs(last_mlp_child, graph, ids)
        out = patched_forward(parent, graph, ids, nodes_c, keep, last_n=2)
        ref = parent.lm_head(parent.model.norm(nodes_p.sum(2) - nodes_p[:, :, m] + nodes_c[:, :, m]))
        plain = parent(ids).logits[:, -2:]
    assert (out - plain).abs().max() > 1e-2
    assert torch.allclose(out, ref[:, -2:], rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------- 5. drop_logits_edges


def _drop_ref(graph, idx):
    return [i for i in idx if graph.edge_ur[i, 1].item() != graph.logits_index]


def test_drop_logits_edges_exact_filter_on_random_sets(graph):
    E = graph.n_edges
    all_logits = (graph.edge_ur[:, 1] == graph.logits_index).nonzero().squeeze(1)
    non_logits = (graph.edge_ur[:, 1] != graph.logits_index).nonzero().squeeze(1)
    assert 0 < len(all_logits) < E
    g = torch.Generator().manual_seed(3)
    sets = {"empty": torch.empty(0, dtype=torch.long), "all_edges": torch.arange(E),
            "only_logits": all_logits, "no_logits": non_logits,
            "single_logits": all_logits[:1], "single_other": non_logits[:1]}
    sets |= {f"rand{i}": torch.randperm(E, generator=g)[: int(torch.randint(1, E, (1,), generator=g))]
             for i in range(8)}
    for name, idx in sets.items():
        out = cmap.drop_logits_edges(graph, idx)
        assert out.tolist() == _drop_ref(graph, idx.tolist()), name  # same edges, order kept
        assert out.dtype == torch.long
        if name in ("only_logits", "empty"):
            assert len(out) == 0
        if name == "no_logits":
            assert torch.equal(out, idx)
    assert len(cmap.drop_logits_edges(graph, torch.arange(E))) == E - len(all_logits)


def test_drop_logits_edges_accepts_numpy_indices(graph):
    idx = np.array([5, 0, graph.n_edges - 1, 3])
    assert cmap.drop_logits_edges(graph, idx).tolist() == _drop_ref(graph, idx.tolist())


# ------------------------------------------------------ 6. sparse == dense on CMAP masks


@pytest.mark.parametrize("k", [1, 5, 20, 40, 70])
@pytest.mark.parametrize("nologits", [False, True])
def test_sparse_equals_dense_on_cmap_masks(parent, child, graph, val, circuits, monkeypatch, k, nologits):
    ids = val["clean_ids"]
    top = torch.as_tensor(cmap.et.topk_edges(circuits["child"].numpy(), k))
    if nologits:
        top = cmap.drop_logits_edges(graph, top)
    # patched edge set -> keep; also its complement (mostly-patched masks, other plan branch)
    keeps = [run.keep_without(graph, top, "cpu"),
             run.keep_only(graph, top, "cpu")]
    calls = []
    real_plan = eapig._binary_plan
    monkeypatch.setattr(eapig, "_binary_plan", lambda *a, **kw: calls.append(1) or real_plan(*a, **kw))
    with torch.no_grad():
        src = node_outputs(child, graph, ids)
        child_lp = eapig.answer_logprobs(child(ids).logits)
        sparse = [eapig.kl_2tok(patched_forward(parent, graph, ids, src, kp, last_n=2), child_lp)
                  for kp in keeps]
        assert calls  # the sparse path was exercised
        calls.clear()
        monkeypatch.setenv("GEODE_EAPIG_DENSE", "1")
        dense = [eapig.kl_2tok(patched_forward(parent, graph, ids, src, kp, last_n=2), child_lp)
                 for kp in keeps]
        assert not calls  # and the dense override really bypassed it
    for s, d in zip(sparse, dense):
        assert (s - d).abs().max() < 1e-4


def test_sparse_equals_dense_on_random_masks_of_many_sizes(parent, child, graph, val, monkeypatch):
    ids = val["clean_ids"]
    keeps = [_random_keep(graph, s, p) for s, p in enumerate((0.02, 0.1, 0.3, 0.5, 0.9, 0.98))]
    with torch.no_grad():
        src = node_outputs(child, graph, ids)
        sparse = [patched_forward(parent, graph, ids, src, kp, last_n=2) for kp in keeps]
        monkeypatch.setenv("GEODE_EAPIG_DENSE", "1")
        dense = [patched_forward(parent, graph, ids, src, kp, last_n=2) for kp in keeps]
    for s, d in zip(sparse, dense):
        assert torch.allclose(s, d, atol=1e-4)


def test_eval_cmap_checks_dict_values_tiny(parent, child, graph, val):
    keeps = [run.keep_without(graph, torch.arange(k), "cpu") for k in (3, 30)]
    ev = cmap.eval_cmap(parent, child, graph, val, keeps, BS, check_n=2)
    assert set(ev["checks"]) == {"child_self_patch_max_kl", "sparse_vs_dense_max_dkl"}
    assert all(0 <= v < cmap.CHECK_TOL for v in ev["checks"].values())
    assert cmap.eval_cmap(parent, child, graph, val, keeps, BS)["checks"] == {}


# ---------------------------------------- 7. recovery of the all-patched mask by hand


def test_all_patched_recovery_matches_hand_computation(parent, child, graph, val, circuits):
    ids = val["clean_ids"]
    with torch.no_grad():
        child_logits, parent_logits = child(ids).logits, parent(ids).logits
        allp = patched_forward(parent, graph, ids, node_outputs(child, graph, ids),
                               torch.zeros_like(graph.valid), last_n=2)
    kl_par = _kl_hand(parent_logits, child_logits).mean().item()
    kl_all = _kl_hand(allp, child_logits).mean().item()
    expected = 1.0 - kl_all / kl_par
    assert abs(expected) > 1e-3 and expected != 1.0
    res, _ = cmap.run_cmap(parent, child, graph, val, circuits, (0.1,), 2, BS)
    assert res["baseline"]["kl_child_parent_mean"] == pytest.approx(kl_par, rel=1e-4)
    assert res["all"]["kl_mean"] == pytest.approx(kl_all, rel=1e-4)
    assert res["all"]["recovery"] == pytest.approx(expected, abs=1e-4)
    assert res["none"]["recovery"] == pytest.approx(0.0, abs=1e-4)


# ------------------------------------------------ 8. eval_cmap per-example values


def test_eval_cmap_per_example_matches_direct_computation(parent, child, graph, val):
    ids, c1, c2 = val["clean_ids"], val["c1"], val["c2"]
    keeps = [run.keep_without(graph, torch.arange(k), "cpu") for k in (2, 25, 60)]
    keeps += [graph.valid, torch.zeros_like(graph.valid), _random_keep(graph, 9)]
    ev = cmap.eval_cmap(parent, child, graph, val, keeps, BS)  # bs=4 -> ragged last batch
    rows = torch.arange(N_VAL)

    def stats(logits):
        lp = torch.log_softmax(logits[:, -2:].double(), -1)
        acc = (lp[:, 0].argmax(-1) == c1) & (lp[:, 1].argmax(-1) == c2)
        return acc, lp[rows, 0, c1] + lp[rows, 1, c2]

    with torch.no_grad():
        child_logits, parent_logits = child(ids).logits, parent(ids).logits
        src = node_outputs(child, graph, ids)
        assert ev["kl"].shape == ev["acc"].shape == ev["logp"].shape == (len(keeps), N_VAL)
        for i, keep in enumerate(keeps):
            logits = patched_forward(parent, graph, ids, src, keep, last_n=2)
            acc, logp = stats(logits)
            assert torch.allclose(ev["kl"][i].double(), _kl_hand(logits, child_logits), atol=1e-4)
            assert torch.equal(ev["acc"][i], acc), i
            assert torch.allclose(ev["logp"][i].double(), logp, atol=1e-4)
        for name, lg in (("parent", parent_logits), ("child", child_logits)):
            acc, logp = stats(lg)
            assert torch.equal(ev[f"acc_{name}"], acc)
            assert torch.allclose(ev[f"logp_{name}"].double(), logp, atol=1e-4)
        assert torch.allclose(ev["kl_parent"].double(), _kl_hand(parent_logits, child_logits), atol=1e-4)
    # non-vacuous: acc is mixed across examples, and logp is a sum of two log-probs (<= 0)
    assert 0 < ev["acc_child"].sum() < N_VAL
    assert (ev["logp"] <= 1e-6).all()


# ------------------------------------------------------- 9. batch-size invariance


def _assert_tree_close(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            if k != "sec_total":
                _assert_tree_close(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_tree_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-4, abs=1e-5), path
    else:
        assert a == b, path


def test_run_cmap_batch_size_invariance(parent, child, graph, val, circuits, full_run):
    res4, per4 = full_run
    for bs in (3, 10):
        res, per = cmap.run_cmap(parent, child, graph, val, circuits, FRACS, N_RANDOM, bs)
        _assert_tree_close(res, res4)
        assert set(per) == set(per4)
        for k in per:
            if per[k].dtype == bool:
                assert np.array_equal(per[k], per4[k]), k
            else:
                assert np.allclose(per[k], per4[k], atol=1e-5), k


# ------------------------------------------------------------ 10. run_cmap schema


def test_run_cmap_schema(full_run, graph, circuits):
    res, per_ex = full_run
    E = graph.n_edges
    assert res["n_val"] == N_VAL and res["n_edges"] == E and res["n_random"] == N_RANDOM
    assert list(res["sizes"]) == [str(f) for f in FRACS]
    labels = ["child", "child_nologits", "ep", "ep_nologits", "child_and_ep_nologits",
              "child_minus_ep_nologits"]
    for frac in FRACS:
        s = res["sizes"][str(frac)]
        k = cmap.et.size_to_k(frac, E)
        assert s["k"] == k
        assert list(s["masks"]) == labels
        assert set(s["logits_edges_removed"]) == {"child", "ep"}
        for lab in ("child", "ep"):
            top = torch.as_tensor(cmap.et.topk_edges(circuits[lab].numpy(), k))
            n_logits = int((graph.edge_ur[top, 1] == graph.logits_index).sum())
            assert s["logits_edges_removed"][lab] == n_logits
            assert s["masks"][lab]["n_edges"] == k
            assert s["masks"][f"{lab}_nologits"]["n_edges"] == k - s["logits_edges_removed"][lab]
        assert (s["masks"]["child_and_ep_nologits"]["n_edges"]
                + s["masks"]["child_minus_ep_nologits"]["n_edges"]
                == s["masks"]["child_nologits"]["n_edges"])
        for lab, row in s["masks"].items():
            assert set(row) == {"kl_mean", "recovery", "recovery_ci95", "acc2", "logp_correct",
                                "n_edges", "frac_of_all"}
            assert row["frac_of_all"] == pytest.approx(row["recovery"] / res["all"]["recovery"])
            lo, hi = row["recovery_ci95"]
            assert lo <= hi and all(math.isfinite(x) for x in row["recovery_ci95"])
            assert 0 <= row["acc2"] <= 1 and row["logp_correct"] <= 0
            assert row["recovery"] == pytest.approx(1 - row["kl_mean"] / res["baseline"]["kl_child_parent_mean"])
            assert per_ex[f"kl_{lab}_{frac}"].shape == (N_VAL,)
        for band in ("random", "random_nologits", "random_typed", "random_typed_disjoint"):
            assert len(s[band]["recovery"]) == N_RANDOM
            assert len(s[band]["recovery_band"]) == len(s[band]["acc2_band"]) == 3
            assert s[band]["recovery_band"] == sorted(s[band]["recovery_band"])
        assert len(s["random_nologits"]["logits_edges_removed"]) == N_RANDOM
    assert {"kl_parent", "kl_all", "acc_parent", "acc_child"} <= set(per_ex)
    assert set(res["checks"]) == {"child_self_patch_max_kl", "sparse_vs_dense_max_dkl",
                                  "none_vs_parent_max_dkl", "embed_edges_vs_child_embeds_max_kl",
                                  "logits_edges_vs_all_max_kl"}
    assert all(0 <= v < cmap.CHECK_TOL for v in res["checks"].values())
    assert res["none"]["recovery"] == pytest.approx(0.0, abs=1e-4)
    assert math.isfinite(res["all"]["recovery"]) and res["all"]["recovery"] != 0
    assert res["baseline"]["kl_child_parent_mean"] == pytest.approx(float(per_ex["kl_parent"].mean()), rel=1e-5)


def test_run_cmap_random_sets_are_reconstructable_and_nologits_is_their_filter(
        parent, child, graph, val, full_run):
    res, _ = full_run
    E, kp = graph.n_edges, res["baseline"]["kl_child_parent_mean"]
    for frac in FRACS:
        k = cmap.et.size_to_k(frac, E)
        rnd = run.random_sets(E, k, N_RANDOM, seed=run.SEED + 202 + int(frac * 1e5))
        rnd_nl = [cmap.drop_logits_edges(graph, x) for x in rnd]
        for x, y in zip(rnd, rnd_nl):
            assert y.tolist() == [i for i in x.tolist() if graph.edge_ur[i, 1] != graph.logits_index]
        s = res["sizes"][str(frac)]
        assert s["random_nologits"]["logits_edges_removed"] == [k - len(y) for y in rnd_nl]
        keeps = [run.keep_without(graph, x, "cpu") for x in rnd + rnd_nl]
        ev = cmap.eval_cmap(parent, child, graph, val, keeps, BS)
        rec = [cmap.recovery(float(ev["kl"][i].mean()), kp) for i in range(2 * N_RANDOM)]
        assert s["random"]["recovery"] == pytest.approx(rec[:N_RANDOM], abs=1e-4)
        assert s["random_nologits"]["recovery"] == pytest.approx(rec[N_RANDOM:], abs=1e-4)
        assert s["random"]["recovery_band"] == pytest.approx(np.percentile(rec[:N_RANDOM], [5, 50, 95]), abs=1e-4)


# ------------------------------------- new: strata, typed nulls, split masks, direction


def _up_key(name: str, L: int):
    if name == "embed":
        return ("embed", -1)
    kind, rest = name.split(".") if "." in name else (name, "")
    return ("head" if rest.startswith("h") else "mlp", int(kind[1:]))


def _rc_key(name: str, L: int):
    if name == "logits":
        return ("logits", L)
    head, rest = name.split(".")
    return (rest.rstrip("0123456789") if rest != "in" else "mlp_in", int(head[1:]))


def test_edge_strata_is_a_bijection_with_upstream_and_receiver_kind_and_layer(graph):
    st = cmap.edge_strata(graph)
    L = graph.n_layers
    assert st.shape == (graph.n_edges,) and st.dtype == torch.long
    assert sorted(set(st.tolist())) == list(range(int(st.max()) + 1))  # contiguous ids
    pairs = [(_up_key(graph.upstream_names[u], L), _rc_key(graph.receiver_names[r], L))
             for u, r in graph.edge_ur.tolist()]
    by_stratum: dict = {}
    for s_, pr in zip(st.tolist(), pairs):
        by_stratum.setdefault(s_, set()).add(pr)
    assert all(len(v) == 1 for v in by_stratum.values())  # never mixes kinds/layers
    assert len({next(iter(v)) for v in by_stratum.values()}) == len(by_stratum)  # never splits


def test_edge_strata_logits_edges_group_by_upstream_kind_and_layer(graph):
    st = cmap.edge_strata(graph)
    names = graph.edge_names()

    def s(n):
        return st[names.index(n)].item()

    assert s("a0.h0->logits") == s("a0.h3->logits")
    assert s("a0.h0->logits") != s("a1.h0->logits")
    assert len({s("a0.h0->logits"), s("m0->logits"), s("embed->logits"), s("m1->logits")}) == 4
    assert s("a0.h0->a1.q0") == s("a0.h3->a1.q3") != s("a0.h0->a1.k0")
    assert s("a0.h0->a1.k0") == s("a0.h2->a1.k1") != s("a0.h0->a1.v0")
    assert s("embed->a1.q0") != s("a0.h0->a1.q0") != s("m0->a1.q0")
    assert s("embed->m0.in") != s("embed->m1.in") != s("embed->a1.q0")


def _counts(graph, strata, idx):
    return torch.bincount(strata[idx], minlength=int(strata.max()) + 1)


@pytest.mark.parametrize("kind", ["random20", "random_all_but_few", "all_edges", "single", "logits_only"])
def test_typed_random_sets_match_per_stratum_counts(graph, kind):
    E = graph.n_edges
    g = torch.Generator().manual_seed(4)
    idx = {"random20": torch.randperm(E, generator=g)[:20],
           "random_all_but_few": torch.randperm(E, generator=g)[: E - 3],
           "all_edges": torch.arange(E), "single": torch.tensor([7]),
           "logits_only": (graph.edge_ur[:, 1] == graph.logits_index).nonzero().squeeze(1)}[kind]
    st = cmap.edge_strata(graph)
    sets = cmap.typed_random_sets(graph, idx, 6, seed=3)
    assert len(sets) == 6
    want = _counts(graph, st, idx)
    for x in sets:
        assert len(x) == len(idx) and len(set(x.tolist())) == len(x)  # no duplicates
        assert x.min() >= 0 and x.max() < E  # valid edge indices only
        assert torch.equal(_counts(graph, st, x), want)
    assert [x.tolist() for x in cmap.typed_random_sets(graph, idx, 6, seed=3)] == [x.tolist() for x in sets]
    assert [x.tolist() for x in cmap.typed_random_sets(graph, idx, 6, seed=3, strata=st)] == [x.tolist() for x in sets]
    if kind == "random20":
        other = cmap.typed_random_sets(graph, idx, 6, seed=4)
        assert [x.tolist() for x in other] != [x.tolist() for x in sets]
        assert len({tuple(x.tolist()) for x in sets}) > 1  # draws differ within a call


def test_typed_random_sets_empty_idx_gives_empty_sets(graph):
    sets = cmap.typed_random_sets(graph, torch.empty(0, dtype=torch.long), 3, seed=0)
    assert len(sets) == 3 and all(len(x) == 0 for x in sets)


@pytest.mark.parametrize("kind", ["random20", "random_all_but_few", "single", "logits_only"])
def test_typed_random_sets_disjoint_never_reuses_idx_and_fills_free_strata(graph, kind):
    E = graph.n_edges
    g = torch.Generator().manual_seed(5)
    idx = {"random20": torch.randperm(E, generator=g)[:20],
           "random_all_but_few": torch.randperm(E, generator=g)[: E - 3],
           "single": torch.tensor([7]),
           "logits_only": (graph.edge_ur[:, 1] == graph.logits_index).nonzero().squeeze(1)}[kind]
    st = cmap.edge_strata(graph)
    need = _counts(graph, st, idx)
    free = _counts(graph, st, torch.arange(E)) - need
    sets = cmap.typed_random_sets(graph, idx, 6, seed=3, strata=st, disjoint=True)
    own = set(idx.tolist())
    for x in sets:
        assert not own & set(x.tolist())  # never an edge of idx
        assert len(set(x.tolist())) == len(x)
        assert torch.equal(_counts(graph, st, x), torch.minimum(need, free))  # min(need, free) per stratum
    again = cmap.typed_random_sets(graph, idx, 6, seed=3, strata=st, disjoint=True)
    assert [x.tolist() for x in again] == [x.tolist() for x in sets]
    if kind == "logits_only":  # every logits stratum fully taken: nothing left to draw
        assert all(len(x) == 0 for x in sets)


def test_random_typed_disjoint_band_reconstructable_and_disjoint(parent, child, graph, val, circuits,
                                                               full_run):
    res, _ = full_run
    kp = res["baseline"]["kl_child_parent_mean"]
    for frac in FRACS:
        k = cmap.et.size_to_k(frac, graph.n_edges)
        top = cmap.drop_logits_edges(graph, torch.as_tensor(cmap.et.topk_edges(circuits["child"].numpy(), k)))
        sets = cmap.typed_random_sets(graph, top, N_RANDOM, seed=run.SEED + 404 + int(frac * 1e5),
                                      disjoint=True)
        assert all(not set(top.tolist()) & set(x.tolist()) for x in sets)
        ev = cmap.eval_cmap(parent, child, graph, val, [run.keep_without(graph, x, "cpu") for x in sets], BS)
        rec = [cmap.recovery(float(ev["kl"][i].mean()), kp) for i in range(N_RANDOM)]
        band = res["sizes"][str(frac)]["random_typed_disjoint"]
        assert band["recovery"] == pytest.approx(rec, abs=1e-4)
        assert band["n_edges"] == [len(x) for x in sets]
        share = res["sizes"][str(frac)]["random_typed"]["share_in_child_nologits"]
        assert 0.0 <= share <= 1.0


def test_split_masks_partition_child_nologits(full_run, graph, circuits, parent, child, val):
    res, per_ex = full_run
    for frac in FRACS:
        k = cmap.et.size_to_k(frac, graph.n_edges)
        top = {lab: cmap.drop_logits_edges(graph, torch.as_tensor(cmap.et.topk_edges(circuits[lab].numpy(), k)))
               for lab in ("child", "ep")}
        c, e = set(top["child"].tolist()), set(top["ep"].tolist())
        both = torch.tensor(sorted(c & e), dtype=torch.long)
        only = torch.tensor(sorted(c - e), dtype=torch.long)
        assert not (c & e) & (c - e) and (c & e) | (c - e) == c
        m = res["sizes"][str(frac)]["masks"]
        assert m["child_and_ep_nologits"]["n_edges"] == len(both)
        assert m["child_minus_ep_nologits"]["n_edges"] == len(only)
        keeps = [run.keep_without(graph, x, "cpu") for x in (both, only)]
        ev = cmap.eval_cmap(parent, child, graph, val, keeps, BS)
        assert np.allclose(per_ex[f"kl_child_and_ep_nologits_{frac}"], ev["kl"][0].numpy(), atol=1e-5)
        assert np.allclose(per_ex[f"kl_child_minus_ep_nologits_{frac}"], ev["kl"][1].numpy(), atol=1e-5)


def test_split_masks_overlap_extremes(parent, child, graph, val):
    """Identical scores: child_and == child_nologits, child_minus empty (patching nothing == parent)."""
    sc = torch.randn(graph.n_edges, generator=torch.Generator().manual_seed(2))
    res, _ = cmap.run_cmap(parent, child, graph, val, {"child": sc, "ep": sc.clone()}, (0.4,), 2, BS)
    m = res["sizes"]["0.4"]["masks"]
    assert m["child_and_ep_nologits"]["n_edges"] == m["child_nologits"]["n_edges"]
    assert m["child_minus_ep_nologits"]["n_edges"] == 0
    assert m["child_and_ep_nologits"]["recovery"] == pytest.approx(m["child_nologits"]["recovery"], abs=1e-5)
    assert m["child_minus_ep_nologits"]["recovery"] == pytest.approx(0.0, abs=1e-4)


def test_random_typed_recovery_reconstructable_from_seed(parent, child, graph, val, circuits, full_run):
    res, _ = full_run
    kp = res["baseline"]["kl_child_parent_mean"]
    for frac in FRACS:
        k = cmap.et.size_to_k(frac, graph.n_edges)
        top = cmap.drop_logits_edges(graph, torch.as_tensor(cmap.et.topk_edges(circuits["child"].numpy(), k)))
        sets = cmap.typed_random_sets(graph, top, N_RANDOM, seed=run.SEED + 303 + int(frac * 1e5))
        ev = cmap.eval_cmap(parent, child, graph, val, [run.keep_without(graph, x, "cpu") for x in sets], BS)
        rec = [cmap.recovery(float(ev["kl"][i].mean()), kp) for i in range(N_RANDOM)]
        band = res["sizes"][str(frac)]["random_typed"]
        assert band["recovery"] == pytest.approx(rec, abs=1e-4)
        assert band["recovery_band"] == pytest.approx(np.percentile(rec, [5, 50, 95]), abs=1e-4)


def test_direction_checks_tiny_on_pair_and_large_when_mask_is_inverted(
        parent, child, graph, val, monkeypatch):
    ids = val["clean_ids"][:BS]
    good = cmap.direction_checks(parent, child, graph, ids)
    assert set(good) == {"embed_edges_vs_child_embeds_max_kl", "logits_edges_vs_all_max_kl"}
    assert all(0 <= v < cmap.CHECK_TOL for v in good.values())
    monkeypatch.setattr(run, "keep_without", run.keep_only)  # deliberately inverted mask
    bad = cmap.direction_checks(parent, child, graph, ids)
    assert all(v > 1e-2 for v in bad.values()), bad


# ------------------------------------------------------------------ 11. boot_ci


def test_boot_ci_brackets_point_estimate_deterministic_finite():
    rng = np.random.default_rng(0)
    kp = rng.uniform(0.5, 2.0, 60)
    kl = kp * rng.uniform(0.1, 0.6, 60)
    point = 1 - kl.mean() / kp.mean()
    ci = cmap.boot_ci(kl, kp, seed=3)
    assert len(ci) == 2 and all(math.isfinite(x) for x in ci)
    assert ci[0] < point < ci[1]
    assert cmap.boot_ci(kl, kp, seed=3) == ci
    assert cmap.boot_ci(kl, kp, seed=4) != ci
    assert cmap.boot_ci(kl, kp, seed=3, n=50) != ci
    wide = cmap.boot_ci(kl[:10], kp[:10], seed=3)  # fewer prompts -> wider interval
    assert wide[1] - wide[0] > ci[1] - ci[0]


def test_boot_ci_is_nan_when_denominator_is_zero():
    z = np.zeros(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert all(math.isnan(x) for x in cmap.boot_ci(z + 0.1, z, seed=0, n=20))


# ------------------------------------------------------------ 12. shards and merge


def _shard(res: dict, per_ex: dict, path: Path, route: str = "elicit") -> Path:
    """Write a shard the way main() does."""
    d = {"route": route, "parent": "P", "child": "C",
         "circuits": {"child": "dirC", "ep": "dirE"},
         "checks_pass": all(v < cmap.CHECK_TOL for v in res["checks"].values())} | res
    run.dump(d, path)
    np.savez_compressed(path.with_suffix(".npz"), **per_ex)
    return path


@pytest.fixture(scope="module")
def shards(tmp_path_factory, parent, child, graph, val, circuits):
    tmp = tmp_path_factory.mktemp("shards")
    out = {}
    for frac in FRACS[::-1]:  # written large-first: merge must sort
        res, per = cmap.run_cmap(parent, child, graph, val, circuits, (frac,), N_RANDOM, BS)
        out[frac] = _shard(res, per, tmp / f"elicit_{frac}.json")
    return out


def test_merge_unions_sizes_sorted_and_npz_keys(shards, full_run, tmp_path):
    out = tmp_path / "merged" / "elicit.json"
    out.parent.mkdir()
    paths = [str(shards[f]) for f in FRACS[::-1]]
    ret = cmap.merge(paths, out)
    merged = json.loads(out.read_text())
    assert list(merged["sizes"]) == [str(f) for f in FRACS]  # sorted ascending numerically
    assert ret["shards"] == paths and merged["checks_pass"] is True
    full, _ = full_run
    for frac in FRACS:  # shards reproduce the all-sizes run
        _assert_tree_close(merged["sizes"][str(frac)], full["sizes"][str(frac)])
    assert merged["checks"].keys() == full["checks"].keys()
    sh = [json.loads(Path(p).read_text()) for p in paths]
    assert merged["sec_total"] == pytest.approx(sum(s["sec_total"] for s in sh))
    for key in ("kl_child_parent_mean", "acc2_parent"):
        assert merged["baseline"][key] == pytest.approx(full["baseline"][key], rel=1e-4)
    with np.load(out.with_suffix(".npz")) as z, np.load(shards[FRACS[0]].with_suffix(".npz")) as z0, \
            np.load(shards[FRACS[1]].with_suffix(".npz")) as z1:
        assert set(z.files) == set(z0.files) | set(z1.files)
        assert f"kl_child_{FRACS[0]}" in z.files and f"kl_child_{FRACS[1]}" in z.files
        assert np.array_equal(z[f"kl_child_{FRACS[1]}"], z1[f"kl_child_{FRACS[1]}"])


def test_merge_via_main_cli(shards, tmp_path):
    out = tmp_path / "m.json"
    cmap.main(["--merge", str(shards[FRACS[0]]), str(shards[FRACS[1]]), "--out", str(out)])
    assert list(json.loads(out.read_text())["sizes"]) == [str(f) for f in FRACS]
    with pytest.raises(SystemExit):
        cmap.main(["--merge", str(shards[FRACS[0]])])  # --out missing


def test_merge_duplicate_size_raises(shards, tmp_path):
    dup = tmp_path / "dup.json"
    dup.write_text(shards[FRACS[0]].read_text())
    dup.with_suffix(".npz").write_bytes(shards[FRACS[0]].with_suffix(".npz").read_bytes())
    with pytest.raises(AssertionError, match="size in two shards"):
        cmap.merge([str(shards[FRACS[0]]), str(dup)], tmp_path / "o.json")


@pytest.mark.parametrize("field,edit", [
    ("baseline", lambda d: d["baseline"].__setitem__("kl_child_parent_mean", d["baseline"]["kl_child_parent_mean"] + 1)),
    ("baseline_acc", lambda d: d["baseline"].__setitem__("acc2_child", d["baseline"]["acc2_child"] + 0.5)),
    ("all", lambda d: d["all"].__setitem__("recovery", d["all"]["recovery"] + 0.1)),
    ("route", lambda d: d.__setitem__("route", "teach")),
    ("n_val", lambda d: d.__setitem__("n_val", d["n_val"] + 1)),
    ("n_random", lambda d: d.__setitem__("n_random", 99)),
])
def test_merge_mismatched_shared_fields_raise(shards, tmp_path, field, edit):
    bad = tmp_path / "bad.json"
    d = json.loads(shards[FRACS[1]].read_text())
    edit(d)
    bad.write_text(json.dumps(d))
    bad.with_suffix(".npz").write_bytes(shards[FRACS[1]].with_suffix(".npz").read_bytes())
    with pytest.raises(AssertionError):
        cmap.merge([str(shards[FRACS[0]]), str(bad)], tmp_path / "o.json")


def test_merge_checks_pass_requires_every_shard(shards, tmp_path):
    bad = tmp_path / "bad.json"
    d = json.loads(shards[FRACS[1]].read_text())
    d["checks_pass"] = False
    d["checks"]["child_self_patch_max_kl"] = 0.5
    bad.write_text(json.dumps(d))
    bad.with_suffix(".npz").write_bytes(shards[FRACS[1]].with_suffix(".npz").read_bytes())
    out = tmp_path / "o.json"
    cmap.merge([str(shards[FRACS[0]]), str(bad)], out)
    merged = json.loads(out.read_text())
    assert merged["checks_pass"] is False and merged["checks"]["child_self_patch_max_kl"] == 0.5


# ---------------------------------------------------------------- 13. summarize


def test_summarize_one_row_per_size_per_route(shards, tmp_path, capsys):
    merged = tmp_path / "elicit.json"
    cmap.merge([str(shards[FRACS[0]]), str(shards[FRACS[1]])], merged)
    teach = tmp_path / "teach.json"
    d = json.loads(merged.read_text())
    d["route"] = "teach"
    teach.write_text(json.dumps(d))
    md = cmap.summarize([str(merged), str(teach)])
    rows = [ln for ln in md.splitlines() if re.match(r"^\| [\d.]+% \|", ln)]
    assert len(rows) == 2 * len(FRACS)
    assert "**elicit**" in md and "**teach**" in md
    assert md.count("| size | k |") == 2
    for lab in ("child", "child_nologits", "ep", "ep_nologits", "child_and_ep_nologits",
                "child_minus_ep_nologits"):
        assert lab in md
    assert "typed" in md
    assert f"| {FRACS[0] * 100:g}% |" in md and f"| {FRACS[1] * 100:g}% |" in md
    cmap.main(["--summarize", str(merged)])
    assert "all-patched recovery" in capsys.readouterr().out


# --------------------------------------------------------------------- 14. main


class _FakeGraph(EdgeGraph):
    """Reports the production edge count on its FIRST n_edges read (main's assert), then the real one."""

    _reads = 0

    @property
    def n_edges(self) -> int:
        if type(self)._reads == 0:
            type(self)._reads = 1
            return 195_865
        return int(self.valid.sum())


@pytest.fixture
def main_env(monkeypatch, tmp_path, parent, child, val, circuits):
    models = {"P": parent, "C": child}
    monkeypatch.setattr(run, "load_model", lambda path, device: models[path])
    monkeypatch.setattr(run, "load_data", lambda d, task="word": {"val": val})
    for lab in ("child", "ep"):
        (tmp_path / lab).mkdir()
        torch.save({"mean": circuits[lab]}, tmp_path / lab / "scores.pt")
    _FakeGraph._reads = 0
    monkeypatch.setattr(run, "build_graph", lambda m: _FakeGraph.from_config(m.config))
    return tmp_path


def test_main_end_to_end_smoke(main_env, full_run):
    out = main_env / "res" / "elicit.json"
    cmap.main(["--route", "elicit", "--parent", "P", "--child", "C",
               "--child-scores", str(main_env / "child"), "--extra-circuit", f"ep={main_env / 'ep'}",
               "--sizes", *map(str, FRACS), "--n-random", str(N_RANDOM), "--batch-size", str(BS),
               "--device", "cpu", "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["route"] == "elicit" and r["checks_pass"] is True
    assert r["parent"] == "P" and r["child"] == "C"
    assert r["circuits"] == {"child": str(main_env / "child"), "ep": str(main_env / "ep")}
    assert list(r["sizes"]) == [str(f) for f in FRACS]
    with np.load(out.with_suffix(".npz")) as z:
        assert "kl_parent" in z.files and f"kl_ep_nologits_{FRACS[0]}" in z.files
    full, _ = full_run
    _assert_tree_close({k: r[k] for k in ("baseline", "none", "all", "sizes")},
                       {k: full[k] for k in ("baseline", "none", "all", "sizes")})


def test_main_asserts_production_edge_count_on_the_tiny_graph(main_env, monkeypatch):
    """Without the fake graph, main refuses anything but the 195,865-edge production graph."""
    monkeypatch.setattr(run, "build_graph", build_graph)
    with pytest.raises(AssertionError):
        cmap.main(["--route", "elicit", "--parent", "P", "--child", "C",
                   "--child-scores", str(main_env / "child"), "--device", "cpu",
                   "--out", str(main_env / "o.json")])


@pytest.mark.parametrize("missing", ["--route", "--parent", "--child", "--child-scores", "--out"])
def test_main_requires_flags(main_env, missing):
    args = {"--route": "elicit", "--parent": "P", "--child": "C",
            "--child-scores": str(main_env / "child"), "--out": str(main_env / "o.json")}
    argv = [x for k, v in args.items() if k != missing for x in (k, v)]
    with pytest.raises(SystemExit):
        cmap.main(argv)
