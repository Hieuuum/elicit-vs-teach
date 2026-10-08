"""Tests for `experiments/eapig-circuit-check/run.py`: the --task/--metric switches, the
probe stage, the KL test path (HANDOFF-kl-symbol-nodeedge.md, step 6), the results-dir
guard, and an LD-path key regression.

run.py lives under a hyphenated experiment directory, so it is loaded by file path (as in
`test_eapig_compare.py`). Everything runs on a tiny random-init Llama (2 layers, 82 edges)
with synthetic pairs dicts; the 195,865-edge `main()` path is never touched.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from geode.circuits.eapig import build_graph, eap_ig_scores, kl_2tok, node_outputs, patched_forward

ROOT = Path(__file__).resolve().parents[2]
RUN_PATH = ROOT / "experiments" / "eapig-circuit-check" / "run.py"
_spec = importlib.util.spec_from_file_location("eapig_run", RUN_PATH)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)
et = run.et
HERE = run.HERE

VOCAB = 50
EOS = VOCAB - 1
T = 6
BS = 4
N_DISC, N_VAL, N_COPY = 8, 10, 6  # sanity reads 10 val token ids
SIZES = ("0.02", "0.05", "0.1")
SMOKE = ["--smoke-draws", "6", "2"]

LD_TOP_KEYS = ["m_full", "m_empty", "n_draws", "n_draws_ts", "sizes", "f_log_mean",
               "validity_gate", "tests", "selected_size"]
LD_SIZE_KEYS = ["k", "m_circuit", "m_circuit_terms", "f", "f_random", "sufficiency",
                "f_removed", "f_removed_random", "equivalence", "partial_necessity",
                "consistency", "specificity"]
LD_SANITY_KEYS = ["n_edges", "n_val", "exact_match", "em_per_example",
                  "nonstandard_split_share", "nonstandard_per_example", "m_full",
                  "m_full_terms", "m_empty", "m_empty_terms", "identity_err_full",
                  "identity_err_empty", "sec_per_circuit_eval", "token_ids_10"]


# ------------------------------------------------------------------ fixtures


class FakeTok:
    """Just enough tokenizer for stage_sanity: ids decode to digits, EOS is special."""

    eos_token_id = EOS

    def decode(self, ids, skip_special_tokens=False):
        return "".join(str(i % 10) for i in ids if not (skip_special_tokens and i == EOS))

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [int(c) for c in text if c.isdigit()]}


def _tiny_model() -> LlamaForCausalLM:
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, tie_word_embeddings=True)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = LlamaForCausalLM(cfg).float().eval()
        with torch.no_grad():  # std-0.02 init is near-uniform; scale up for real signal
            for p in model.parameters():
                if p.dim() > 1:
                    p.mul_(4.0)
    return model


def _pairs(n: int, seed: int, with_half: bool = False) -> dict:
    g = torch.Generator().manual_seed(seed)
    clean = torch.randint(0, EOS, (n, T), generator=g)
    corrupt = clean.clone()
    corrupt[:, 1:4] = torch.randint(0, EOS, (n, 3), generator=g)
    corrupt[:, -1] = (clean[:, -1] + 1) % EOS  # counterfactual first answer token
    c2 = torch.randint(0, EOS, (n,), generator=g)
    p = {"clean_ids": clean, "corrupt_ids": corrupt, "c1": clean[:, -1], "k1": corrupt[:, -1],
         "c2": c2, "k2": (c2 + 3) % EOS, "answer_text": [f"{i}{i}" for i in range(n)]}
    if with_half:
        p["half"] = torch.arange(n) % 2
    return p


@pytest.fixture(scope="module")
def model():
    return _tiny_model()


@pytest.fixture(scope="module")
def graph(model):
    return build_graph(model)


@pytest.fixture(scope="module")
def data():
    g = torch.Generator().manual_seed(9)
    return {"disc": _pairs(N_DISC, 1, with_half=True), "val": _pairs(N_VAL, 2),
            "copy": _pairs(N_COPY, 3), "stories": torch.randint(0, EOS, (4, 8), generator=g)}


def _args(results: Path, tag: str, *extra: str):
    return run.parse_args(["--tag", tag, "--model", "unused", "--device", "cpu",
                           "--batch-size", str(BS), "--results", str(results), *extra])


def _run(model, graph, data, results: Path, tag: str, *extra: str) -> Path:
    run.run_stages(_args(results, tag, *extra), model, graph, FakeTok(), data)
    return results / tag


@pytest.fixture(scope="module")
def scored(tmp_path_factory, model, graph, data):
    """sanity + score for a parent and a child, per metric: {metric: results dir}."""
    dirs = {}
    for metric in ("ld", "kl"):
        res = tmp_path_factory.mktemp(f"scores_{metric}")
        for tag in ("elicit_parent", "elicit_child"):
            _run(model, graph, data, res, tag, "--metric", metric, "--stage", "sanity", "score")
        dirs[metric] = res
    return dirs


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _force_pass(monkeypatch):
    """Make every test verdict pass, so any size passes all five tests."""
    def passing(fn):
        def wrapped(*a, **k):
            return fn(*a, **k) | {"pass": True}
        return wrapped

    for name in ("random_baseline_test", "tost_equivalence", "kl_equivalence", "specificity",
                 "kl_specificity"):
        monkeypatch.setattr(et, name, passing(getattr(et, name)))
    monkeypatch.setattr(et, "consistency_pass", lambda *a, **k: True)


# --------------------------------------------------------------- eval_kl / logprobs


def test_clean_logprobs_shape_normalized_on_cpu(model, data):
    lp = run.clean_logprobs(model, data["val"], BS)
    assert lp.shape == (N_VAL, 2, VOCAB) and lp.dtype == torch.float32
    assert lp.device.type == "cpu"
    assert torch.allclose(lp.logsumexp(-1), torch.zeros(N_VAL, 2), atol=1e-5)
    # batching does not change the result
    assert torch.allclose(lp, run.clean_logprobs(model, data["val"], N_VAL), atol=1e-6)


def test_eval_kl_full_keep_zero_and_empty_is_plain_corrupt_kl(model, graph, data):
    val = data["val"]
    lp = run.clean_logprobs(model, val, BS)
    full = graph.valid
    empty = torch.zeros_like(full)
    kl = run.eval_kl(model, graph, val, [full, empty], BS, lp)
    assert kl.shape == (2, N_VAL)
    assert kl[0].abs().max() < 1e-4
    with torch.no_grad():
        ref = kl_2tok(model(val["corrupt_ids"]).logits, lp)
    assert torch.allclose(kl[1], ref, rtol=1e-4, atol=1e-5)
    assert (kl[1] > 1e-3).all()  # the corrupt run really differs


def test_eval_kl_matches_direct_patched_forward_any_mask(model, graph, data):
    val = data["val"]
    lp = run.clean_logprobs(model, val, BS)
    keeps = [run.keep_only(graph, torch.tensor(r), "cpu")
             for r in ([0, 5, 9, 30], list(range(0, graph.n_edges, 3)))]
    kl = run.eval_kl(model, graph, val, keeps, BS, lp)
    with torch.no_grad():
        acts = node_outputs(model, graph, val["corrupt_ids"])
        for i, keep in enumerate(keeps):
            ref = kl_2tok(patched_forward(model, graph, val["clean_ids"], acts, keep), lp)
            assert torch.allclose(kl[i], ref, rtol=1e-4, atol=1e-5)
    assert (kl >= -1e-6).all()
    # batch-outer layout: a different batch size gives the same numbers
    assert torch.allclose(kl, run.eval_kl(model, graph, val, keeps, 5, lp), atol=1e-5)


def test_eval_kl_empty_mask_list(model, graph, data):
    lp = run.clean_logprobs(model, data["val"], BS)
    assert run.eval_kl(model, graph, data["val"], [], BS, lp).shape == (0, N_VAL)


def test_make_judge_kl_f_is_one_minus_ratio(model, graph, data):
    sanity = {"kl_empty_mean": 2.0, "m_full": 1.0, "m_empty": 0.0}
    _, main, f_of = run.make_judge(model, graph, data, BS, "kl", sanity)
    assert f_of(0.0) == 1.0 and f_of(2.0) == 0.0 and f_of(0.5) == pytest.approx(0.75)
    x = torch.rand(3, 4)
    assert main(x) is x
    assert np.isnan(run.make_judge(model, graph, data, BS, "kl", {"kl_empty_mean": 0.0})[2](1.0))
    _, main_ld, f_ld = run.make_judge(model, graph, data, BS, "ld", sanity)
    assert f_ld(0.25) == pytest.approx(0.25)
    assert main_ld(torch.rand(3, 4, 3)).shape == (3, 4)


# ------------------------------------------------------------------- sanity / score


def test_sanity_kl_fields(scored):
    s = _load(scored["kl"] / "elicit_parent" / "sanity.json")
    assert list(s)[: len(LD_SANITY_KEYS)] == LD_SANITY_KEYS  # LD fields unchanged, first
    assert list(s)[len(LD_SANITY_KEYS):] == ["kl_empty_mean", "kl_empty_per_example",
                                             "kl_identity_err_full"]
    assert len(s["kl_empty_per_example"]) == N_VAL
    assert s["kl_empty_mean"] == pytest.approx(np.mean(s["kl_empty_per_example"]), rel=1e-5)
    assert s["kl_empty_mean"] > 0
    assert 0 <= s["kl_identity_err_full"] < run.KL_IDENTITY_TOL
    # sanity is metric-independent
    s_ld = _load(scored["ld"] / "elicit_parent" / "sanity.json")
    for key in ("m_full", "m_empty", "kl_empty_mean", "exact_match"):
        assert s_ld[key] == pytest.approx(s[key], rel=1e-6)


def test_score_kl_keys_shapes_finite(scored, graph):
    sc = torch.load(scored["kl"] / "elicit_child" / "scores.pt")
    assert set(sc) == {"mean", "mean_a", "mean_b", "per_example"}
    E = graph.n_edges
    assert sc["mean"].shape == sc["mean_a"].shape == sc["mean_b"].shape == (E,)
    assert sc["per_example"].shape == (N_DISC, E) and sc["per_example"].dtype == torch.float16
    for v in sc.values():
        assert torch.isfinite(v.float()).all()
    assert sc["mean"].abs().sum() > 0
    ld = torch.load(scored["ld"] / "elicit_child" / "scores.pt")
    assert set(ld) == set(sc)
    assert not torch.allclose(ld["mean"], sc["mean"])  # the metric switch takes effect


def test_score_kl_is_eapig_of_negative_kl(scored, model, graph, data):
    disc = data["disc"]
    lp = run.clean_logprobs(model, disc, BS)
    ref = torch.cat([eap_ig_scores(model, graph, disc["clean_ids"][s : s + BS],
                                   disc["corrupt_ids"][s : s + BS],
                                   lambda lg, lp=lp[s : s + BS]: -kl_2tok(lg, lp), steps=5)
                     for s in range(0, N_DISC, BS)])
    sc = torch.load(scored["kl"] / "elicit_child" / "scores.pt")
    assert torch.allclose(sc["mean"], ref.mean(0), rtol=1e-4, atol=1e-6)
    half = disc["half"]
    assert torch.allclose(sc["mean_a"], ref[half == 0].mean(0), rtol=1e-4, atol=1e-6)
    assert torch.allclose(sc["per_example"].float(), ref.half().float())


# ------------------------------------------------------------------------- probe


@pytest.mark.parametrize("tag,metric,n", [("elicit_parent", "kl", 20), ("elicit_child", "kl", 10),
                                          ("elicit_parent", "ld", 20), ("elicit_child", "ld", 10)])
def test_probe_schema_and_draw_counts(tmp_path, scored, model, graph, data, tag, metric, n):
    out = _run(model, graph, data, tmp_path, tag, "--metric", metric, "--stage", "probe",
               "--scores-dir", str(scored[metric]))
    p = _load(out / "probe.json")
    assert p["metric"] == metric and p["task"] == "word" and p["n_draws"] == n
    if metric == "kl":
        assert "kl_empty_mean" in p and "m_full" not in p
        sanity = _load(scored[metric] / tag / "sanity.json")
        assert p["kl_empty_mean"] == sanity["kl_empty_mean"]
        assert p["performing"]["signal"] == (sanity["kl_empty_mean"] > run.KL_GATE_NATS)
    else:
        assert "m_full" in p and "m_empty" in p and "kl_empty_mean" not in p
        assert p["performing"]["signal"] == (p["m_full"] - p["m_empty"] > 0)
    assert list(p["sizes"]) == list(SIZES)
    for frac, r in p["sizes"].items():
        assert set(r) == {"k", "f", "f_random", "random_p5", "random_p50", "random_p95",
                          "above_p95"}
        assert r["k"] == et.size_to_k(float(frac), graph.n_edges)
        assert len(r["f_random"]) == n
        assert r["random_p5"] <= r["random_p50"] <= r["random_p95"]
        assert r["random_p95"] == pytest.approx(np.percentile(r["f_random"], 95))
        assert r["above_p95"] == (r["f"] > r["random_p95"])
    perf = p["performing"]
    assert perf["f10_above_p95"] == p["sizes"]["0.1"]["above_p95"]
    assert perf["pass"] == (perf["signal"] and perf["f10_above_p95"])
    assert not (out / "evaluate.json").exists() and not (out / "scores.pt").exists()


def test_probe_kl_f_matches_eval_kl(tmp_path, scored, model, graph, data):
    out = _run(model, graph, data, tmp_path, "elicit_child", "--metric", "kl", "--stage", "probe",
               "--scores-dir", str(scored["kl"]), "--probe-sizes", "0.05")
    p = _load(out / "probe.json")["sizes"]["0.05"]
    mean = torch.load(scored["kl"] / "elicit_child" / "scores.pt")["mean"]
    circ = torch.as_tensor(et.topk_edges(mean.numpy(), p["k"]))
    lp = run.clean_logprobs(model, data["val"], BS)
    kl_c = run.eval_kl(model, graph, data["val"], [run.keep_only(graph, circ, "cpu")], BS, lp)
    kl_empty = _load(scored["kl"] / "elicit_child" / "sanity.json")["kl_empty_mean"]
    assert p["f"] == pytest.approx(1 - kl_c.mean().item() / kl_empty, rel=1e-5)


def test_probe_without_10pct_leaves_performing_undecided(tmp_path, scored, model, graph, data):
    out = _run(model, graph, data, tmp_path, "elicit_child", "--metric", "kl", "--stage", "probe",
               "--scores-dir", str(scored["kl"]), "--probe-sizes", "0.02", "0.05")
    perf = _load(out / "probe.json")["performing"]
    assert perf["f10_above_p95"] is None and perf["pass"] is None


@pytest.mark.parametrize("metric", ["ld", "kl"])
def test_probe_draws_are_evaluates_first_draws(tmp_path, scored, model, graph, data, metric):
    """Seeded like evaluate: the probe's random sets are evaluate's first n draws."""
    a, b = tmp_path / "a", tmp_path / "b"
    probe = _run(model, graph, data, a, "elicit_child", "--metric", metric, "--stage", "probe",
                 "--scores-dir", str(scored[metric]))
    ev = _run(model, graph, data, b, "elicit_child", "--metric", metric, "--stage", "evaluate",
              "--scores-dir", str(scored[metric]), "--sizes", *SIZES,
              "--smoke-draws", "12", "2")
    p, e = _load(probe / "probe.json"), _load(ev / "evaluate.json")
    for frac in SIZES:
        assert p["sizes"][frac]["f"] == pytest.approx(e["sizes"][frac]["f"], rel=1e-6)
        assert np.allclose(p["sizes"][frac]["f_random"], e["sizes"][frac]["f_random"][:10],
                           rtol=1e-6, equal_nan=True)


# ------------------------------------------------------------------------- guard


@pytest.mark.parametrize("frozen", ["results", "results_large"])
@pytest.mark.parametrize("extra", [
    ["--metric", "kl"],
    ["--task", "symbol", "--stage", "sanity", "score"],
    ["--stage", "sanity", "score", "probe"],
    ["--stage", "probe"],
    ["--metric", "kl", "--stage", "score"],
])
def test_guard_refuses_frozen_dirs(frozen, extra, capsys):
    with pytest.raises(SystemExit) as e:
        _args(HERE / frozen, "elicit_child", *extra)
    assert e.value.code == 2
    assert "frozen" in capsys.readouterr().err


def test_guard_resolves_paths():
    with pytest.raises(SystemExit) as e:
        _args(HERE / "results_large" / ".." / "results", "elicit_child", "--metric", "kl")
    assert e.value.code == 2


@pytest.mark.parametrize("frozen", ["results", "results_large"])
def test_guard_allows_frozen_path(frozen):
    a = _args(HERE / frozen, "elicit_child", "--task", "word", "--metric", "ld",
              "--stage", "sanity", "score", "evaluate")
    assert (a.task, a.metric, a.stage) == ("word", "ld", ["sanity", "score", "evaluate"])
    # all defaults: the frozen path into the default results dir
    a = run.parse_args(["--tag", "teach_child", "--model", "m"])
    assert a.results == str(HERE / "results") and a.scores_dir is None
    assert (a.task, a.metric, a.stage) == ("word", "ld", ["sanity", "score", "evaluate"])
    assert a.probe_sizes == [0.02, 0.05, 0.1]


def test_guard_allows_other_dirs(tmp_path):
    a = _args(tmp_path / "results_klsym", "elicit_child", "--task", "symbol", "--metric", "kl",
              "--stage", "sanity", "score", "probe")
    assert (a.task, a.metric) == ("symbol", "kl")
    assert _args(HERE / "results_kltests", "fmt_parent", "--metric", "kl").metric == "kl"


@pytest.mark.parametrize("tag", ["fmt_parent", "teach_child"])
def test_symbol_refused_for_teach_route(tmp_path, tag, capsys):
    with pytest.raises(SystemExit) as e:
        _args(tmp_path, tag, "--task", "symbol", "--stage", "sanity", "score")
    assert e.value.code == 2 and "elicit route" in capsys.readouterr().err


@pytest.mark.parametrize("tag", ["elicit_parent", "elicit_child"])
def test_symbol_allowed_for_elicit_route(tmp_path, tag):
    assert _args(tmp_path, tag, "--task", "symbol", "--stage", "sanity", "score").task == "symbol"


def test_evaluate_refused_on_symbol(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        _args(tmp_path, "elicit_child", "--task", "symbol", "--metric", "kl",
              "--stage", "evaluate")
    assert e.value.code == 2 and "word task only" in capsys.readouterr().err


def test_unknown_stage_refused(tmp_path):
    with pytest.raises(SystemExit):
        _args(tmp_path, "elicit_child", "--stage", "scores")


# ---------------------------------------------------------------- KL evaluate


def _evaluate(model, graph, data, results, tag, metric, scores_dir, *extra):
    parent = ["--parent-tag", "elicit_parent"] if tag == "elicit_child" else []
    out = _run(model, graph, data, results, tag, "--metric", metric, "--stage", "evaluate",
               "--scores-dir", str(scores_dir), "--sizes", *SIZES, *parent, *(extra or SMOKE))
    return _load(out / "evaluate.json"), np.load(out / "per_example.npz")


def test_kl_evaluate_schema(tmp_path, scored, model, graph, data):
    e, px = _evaluate(model, graph, data, tmp_path, "elicit_child", "kl", scored["kl"])
    assert e["metric"] == "kl" and e["task"] == "word"
    assert e["n_draws"] == 6 and e["n_draws_ts"] == 2
    sanity = _load(scored["kl"] / "elicit_child" / "sanity.json")
    assert e["kl_empty_mean"] == sanity["kl_empty_mean"]
    assert set(LD_TOP_KEYS) - {"m_full", "m_empty"} <= set(e)
    assert {"parent_in_child", "real_patching_top20", "copy_kl_empty_mean"} <= set(e)
    assert list(e["tests"]) == list(SIZES)
    assert "selected_size" in e
    assert e["selected_size"] == et.stopping_rule({float(s): v for s, v in e["tests"].items()})
    for frac in SIZES:
        r = e["sizes"][frac]
        assert "m_circuit" not in r and "m_circuit_terms" not in r
        assert set(LD_SIZE_KEYS) - {"m_circuit", "m_circuit_terms"} | {"kl_circuit"} == set(r)
        assert r["f"] == pytest.approx(1 - r["kl_circuit"] / e["kl_empty_mean"], rel=1e-6)
        assert len(r["f_random"]) == len(r["f_removed_random"]) == 6
        eq = r["equivalence"]
        assert {"p", "mean_kl", "bound", "pass"} <= set(eq)
        assert eq["bound"] == pytest.approx(run.EQUIV_EPS_FRAC * e["kl_empty_mean"])
        assert eq["mean_kl"] == pytest.approx(px[f"suff_{frac}"].mean(), rel=1e-6)
        spec = r["specificity"]
        assert spec["rel_add"] == pytest.approx(px[f"nec_{frac}"].mean() / e["kl_empty_mean"],
                                                rel=1e-6)
        assert spec["rel_copy"] is not None  # copy has its own KL(full || empty) > 0
        assert r["consistency"]["n_shared"] >= 0 and "mean_coverage" in r["consistency"]
        assert set(e["tests"][frac]) == {"sufficiency", "equivalence", "partial_necessity",
                                         "consistency", "specificity"}
        assert px[f"suff_{frac}"].shape == (N_VAL,)  # one KL column, not LD's 3 terms
    assert np.abs(px["full"]).max() < 1e-4 and np.abs(px["copy_full"]).max() < 1e-4
    assert e["copy_kl_empty_mean"] == pytest.approx(px["copy_empty"].mean(), rel=1e-6)
    rp = e["real_patching_top20"]
    assert len(rp["edges"]) == len(rp["actual_drop"]) == len(rp["eapig_score"]) == 20
    assert list(e["parent_in_child"]) == list(SIZES)
    assert e["validity_gate"]["kl_empty_above"] == (e["kl_empty_mean"] > run.KL_GATE_NATS)


def test_kl_evaluate_runs_all_sizes_without_break(tmp_path, scored, model, graph, data,
                                                  monkeypatch):
    """With every test passing at the first size, LD stops there and KL still runs all."""
    _force_pass(monkeypatch)
    e_kl, _ = _evaluate(model, graph, data, tmp_path / "kl", "elicit_child", "kl", scored["kl"])
    assert list(e_kl["tests"]) == list(SIZES)
    assert all(all(t.values()) for t in e_kl["tests"].values())
    assert e_kl["selected_size"] == 0.02
    assert "tinystories" in e_kl  # unchanged check at the selected size
    assert len(e_kl["tinystories"]["loss_random"]) == 2
    e_ld, _ = _evaluate(model, graph, data, tmp_path / "ld", "elicit_child", "ld", scored["ld"])
    assert list(e_ld["tests"]) == ["0.02"] and e_ld["selected_size"] == 0.02


def test_kl_real_patching_drop_is_kl_increase(tmp_path, scored, model, graph, data):
    e, _ = _evaluate(model, graph, data, tmp_path, "elicit_child", "kl", scored["kl"])
    rp = e["real_patching_top20"]
    mean = torch.load(scored["kl"] / "elicit_child" / "scores.pt")["mean"]
    top = torch.as_tensor(et.topk_edges(mean.numpy(), 20))
    lp = run.clean_logprobs(model, data["disc"], BS)
    keeps = [graph.valid, run.keep_without(graph, top[:1], "cpu")]
    kl = run.eval_kl(model, graph, data["disc"], keeps, BS, lp).mean(1)
    assert rp["actual_drop"][0] == pytest.approx((kl[1] - kl[0]).item(), rel=1e-4, abs=1e-6)


def test_kl_parent_gate_fails_stops_after_suff_and_nec(tmp_path, scored, model, graph, data):
    src = tmp_path / "scores"
    shutil.copytree(scored["kl"] / "elicit_parent", src / "elicit_parent")
    s = _load(src / "elicit_parent" / "sanity.json")
    s["kl_empty_mean"] = 0.05  # below the 0.1-nat gate
    (src / "elicit_parent" / "sanity.json").write_text(json.dumps(s))
    e, px = _evaluate(model, graph, data, tmp_path / "out", "elicit_parent", "kl", src)
    assert e["validity_gate"]["pass"] is False and e["validity_gate"]["kl_empty_above"] is False
    for key in ("tests", "selected_size", "tinystories", "parent_in_child",
                "real_patching_top20", "copy_kl_empty_mean"):
        assert key not in e
    for frac in SIZES:
        r = e["sizes"][frac]
        assert "sufficiency" in r and "partial_necessity" in r
        assert r["partial_necessity"] == et.random_baseline_test(
            r["f_removed"], np.array(r["f_removed_random"]), False)
        for key in ("equivalence", "consistency", "specificity"):
            assert key not in r
    assert "full" not in px and "copy_full" not in px


def test_kl_parent_gate_passes_runs_tests(tmp_path, scored, model, graph, data, monkeypatch):
    _force_pass(monkeypatch)
    s = _load(scored["kl"] / "elicit_parent" / "sanity.json")
    assert s["kl_empty_mean"] > run.KL_GATE_NATS  # the tiny model has real signal
    e, _ = _evaluate(model, graph, data, tmp_path, "elicit_parent", "kl", scored["kl"])
    assert e["validity_gate"]["pass"] is True
    assert list(e["tests"]) == list(SIZES)
    assert "parent_in_child" not in e and "real_patching_top20" not in e


def test_kl_parent_gate_needs_sufficiency(tmp_path, scored, model, graph, data, monkeypatch):
    orig = et.random_baseline_test
    monkeypatch.setattr(et, "random_baseline_test", lambda *a, **k: orig(*a, **k) | {"pass": False})
    e, _ = _evaluate(model, graph, data, tmp_path, "elicit_parent", "kl", scored["kl"])
    assert e["validity_gate"]["kl_empty_above"] is True and e["validity_gate"]["pass"] is False
    assert "tests" not in e


def test_kl_evaluate_frozen_draw_counts(tmp_path, scored, model, graph, data):
    """No smoke override: the frozen counts (fast box: 100 / 20; slow parent: 50 / 10)."""
    src = tmp_path / "scores"
    shutil.copytree(scored["kl"] / "elicit_parent", src / "elicit_parent")
    s = _load(src / "elicit_parent" / "sanity.json")
    s["kl_empty_mean"] = 0.05  # fail the gate: only sufficiency + necessity draws run
    for slow, n in ((0.1, 100), (5.0, 50)):
        s["sec_per_circuit_eval"] = slow
        (src / "elicit_parent" / "sanity.json").write_text(json.dumps(s))
        out = _run(model, graph, data, tmp_path / f"o{n}", "elicit_parent", "--metric", "kl",
                   "--stage", "evaluate", "--scores-dir", str(src), "--sizes", "0.05")
        e = _load(out / "evaluate.json")
        assert e["n_draws"] == n and len(e["sizes"]["0.05"]["f_random"]) == n
        assert e["n_draws_ts"] == (10 if slow > 2 else 20)


def test_scores_dir_reads_there_writes_results(tmp_path, scored, model, graph, data):
    res = tmp_path / "results_kltests"
    e, _ = _evaluate(model, graph, data, res, "elicit_child", "kl", scored["kl"])
    out = res / "elicit_child"
    assert sorted(p.name for p in out.iterdir()) == ["evaluate.json", "per_example.npz"]
    assert not (res / "elicit_parent").exists()
    assert not (scored["kl"] / "elicit_child" / "evaluate.json").exists()
    # the parent's circuit came from <scores-dir>/elicit_parent/scores.pt
    pmean = torch.load(scored["kl"] / "elicit_parent" / "scores.pt")["mean"]
    k = et.size_to_k(0.05, graph.n_edges)
    pc = torch.as_tensor(et.topk_edges(pmean.numpy(), k))
    lp = run.clean_logprobs(model, data["val"], BS)
    kl_pc = run.eval_kl(model, graph, data["val"], [run.keep_only(graph, pc, "cpu")], BS, lp)
    assert e["parent_in_child"]["0.05"]["f_parent_circuit"] == pytest.approx(
        1 - kl_pc.mean().item() / e["kl_empty_mean"], rel=1e-5)


def test_scores_dir_missing_scores_fails_loudly(tmp_path, model, graph, data):
    with pytest.raises(FileNotFoundError):
        _evaluate(model, graph, data, tmp_path / "out", "elicit_parent", "kl", tmp_path / "none")


# ------------------------------------------------------------- LD regression


def test_ld_evaluate_keys_unchanged_child(tmp_path, scored, model, graph, data):
    e, px = _evaluate(model, graph, data, tmp_path, "elicit_child", "ld", scored["ld"])
    keys = LD_TOP_KEYS + (["tinystories"] if e["selected_size"] is not None else []) + [
        "parent_in_child", "real_patching_top20"]
    assert list(e) == keys
    for frac in e["tests"]:
        assert list(e["sizes"][frac]) == LD_SIZE_KEYS
        assert set(e["sizes"][frac]["equivalence"]) == {"p_lower", "p_upper", "p", "mean_diff",
                                                         "pass", "n", "eps", "alpha"}
    assert list(e["validity_gate"]) == ["m_full_pos", "pass", "f_degenerate"]
    assert px[f"suff_{SIZES[0]}"].shape == (N_VAL, 3)
    assert "copy_empty" not in px


def test_ld_evaluate_keys_unchanged_failing_parent(tmp_path, scored, model, graph, data,
                                                   monkeypatch):
    orig = et.random_baseline_test
    monkeypatch.setattr(et, "random_baseline_test", lambda *a, **k: orig(*a, **k) | {"pass": False})
    e, _ = _evaluate(model, graph, data, tmp_path, "elicit_parent", "ld", scored["ld"])
    assert list(e) == ["m_full", "m_empty", "n_draws", "n_draws_ts", "sizes", "f_log_mean",
                       "validity_gate"]
    for frac in SIZES:  # a failing LD parent gets no partial-necessity verdict (frozen)
        assert list(e["sizes"][frac]) == LD_SIZE_KEYS[:8]


def test_ld_score_stage_unchanged(scored, model, graph, data):
    disc = data["disc"]
    ref = torch.cat([eap_ig_scores(
        model, graph, disc["clean_ids"][s : s + BS], disc["corrupt_ids"][s : s + BS],
        lambda lg, s=s: run.logit_diff_2tok(lg, *(disc[k][s : s + BS] for k in
                                                  ("c1", "k1", "c2", "k2")))[0], steps=5)
        for s in range(0, N_DISC, BS)])
    sc = torch.load(scored["ld"] / "elicit_child" / "scores.pt")
    assert torch.allclose(sc["mean"], ref.mean(0), rtol=1e-5, atol=1e-7)
