"""Tests for `fold_runner.py` and `hidden_pref.py` (10-fold robustness check).

Tiny random-init Llama + synthetic pairs dirs saved with `data.save_pairs`; CPU only, no
network. `main()` is exercised by monkeypatching the model/tokenizer loaders.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from geode.circuits.eapig import build_graph

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eapig-circuit-check"
sys.path.insert(0, str(EXP))  # fold_runner/hidden_pref `import run`; run.load_data -> data
import fold_runner as fr  # noqa: E402
import hidden_pref as hp  # noqa: E402
import run  # noqa: E402
from data import (  # noqa: E402
    ADDITION_PAIRS_FILENAME,
    COPY_PAIRS_FILENAME,
    SYMBOL_PAIRS_FILENAME,
    save_pairs,
)

_spec = importlib.util.spec_from_file_location(
    "eapig_run_helpers", ROOT / "tests/experiments/test_eapig_run.py"
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)

N_DISC, N_VAL, N_COPY, BS = 8, 10, 6, 4
EOS = helpers.EOS


@pytest.fixture(scope="module")
def model():
    return helpers._tiny_model()


@pytest.fixture(scope="module")
def n_edges(model):
    return build_graph(model).n_edges


def _write_fold(d: Path, seed: int) -> None:
    split = ["discovery"] * N_DISC + ["validation"] * N_VAL
    for fname, s in ((ADDITION_PAIRS_FILENAME, seed), (SYMBOL_PAIRS_FILENAME, seed + 50)):
        p = helpers._pairs(N_DISC + N_VAL, s, with_half=True)
        save_pairs(p | {"meta": pd.DataFrame({"split": split})}, d / fname)
    save_pairs(helpers._pairs(N_COPY, seed + 7), d / COPY_PAIRS_FILENAME)


@pytest.fixture(scope="module")
def folds_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("folds")
    for k in range(3):
        _write_fold(root / f"fold_{k:02d}", 10 + k)
    return root


def _args(*extra: str):
    return fr.parse_args(["--model", "unused", "--device", "cpu", "--batch-size", str(BS), *extra])


def _fold_args(folds_root, out, tag="elicit_child", folds=(0, 1, 2), *extra):
    return _args("--tag", tag, "--metric", "kl", "--folds-root", str(folds_root),
                 "--folds", *map(str, folds), "--results-root", str(out),
                 "--subdir", "klword", *extra)  # fmt: skip


def _go(model, a):
    g = build_graph(model)
    return fr.run_folds(model, g, helpers.FakeTok(), a, fr.plan_jobs(a))


@pytest.fixture(scope="module")
def fold_out(tmp_path_factory, model, folds_root):
    out = tmp_path_factory.mktemp("res")
    _go(model, _fold_args(folds_root, out))
    return out


# ------------------------------------------------------------------ fold_runner


def test_fold_mode_writes_expected_tree(fold_out):
    for k in range(3):
        d = fold_out / f"fold_{k:02d}" / "klword" / "elicit_child"
        assert sorted(p.name for p in d.iterdir()) == ["per_example.pt", "sanity.json", "scores.pt"]


def test_scores_split_keys_and_per_example_mean(fold_out, n_edges):
    d = fold_out / "fold_00" / "klword" / "elicit_child"
    sc = torch.load(d / "scores.pt")
    assert set(sc) == {"mean", "mean_a", "mean_b"}
    per = torch.load(d / "per_example.pt")
    assert per.dtype == torch.float16 and per.shape == (N_DISC, n_edges)
    assert sc["mean"].shape == (n_edges,)
    # fp16 per-example mean matches the fp32 stored mean up to fp16 rounding
    scale = sc["mean"].abs().max().item()
    assert (per.float().mean(0) - sc["mean"]).abs().max().item() < 2e-3 * scale + 1e-6


def test_fold_result_independent_of_order(model, folds_root, fold_out, tmp_path):
    _go(model, _fold_args(folds_root, tmp_path, folds=(2,)))
    a = torch.load(tmp_path / "fold_02/klword/elicit_child/scores.pt")["mean"]
    b = torch.load(fold_out / "fold_02/klword/elicit_child/scores.pt")["mean"]
    assert torch.equal(a, b)


def test_different_folds_differ(fold_out):
    m = [
        torch.load(fold_out / f"fold_{k:02d}/klword/elicit_child/scores.pt")["mean"] for k in (0, 1)
    ]
    assert not torch.equal(m[0], m[1])


def test_single_mode_equals_direct_run_stages(model, folds_root, tmp_path):
    g = build_graph(model)
    d = folds_root / "fold_01"
    a = _args("--tag", "elicit_child", "--metric", "kl", "--data-dir", str(d),
              "--results", str(tmp_path / "single"))  # fmt: skip
    (out,) = fr.run_folds(model, g, helpers.FakeTok(), a, fr.plan_jobs(a))
    assert out == tmp_path / "single" / "elicit_child"
    ns = run.parse_args(["--tag", "elicit_child", "--model", "x", "--metric", "kl",
                         "--device", "cpu", "--batch-size", str(BS),
                         "--results", str(tmp_path / "direct")])  # fmt: skip
    torch.manual_seed(run.SEED)
    run.run_stages(ns, model, g, helpers.FakeTok(), run.load_data(d, "word"))
    mine = torch.load(out / "scores.pt")["mean"]
    direct = torch.load(tmp_path / "direct/elicit_child/scores.pt")["mean"]
    assert torch.equal(mine, direct)
    s1 = json.loads((out / "sanity.json").read_text())
    s2 = json.loads((tmp_path / "direct/elicit_child/sanity.json").read_text())
    assert s1["exact_match"] == s2["exact_match"] and s1["m_full"] == s2["m_full"]


def test_symbol_task_uses_symbol_file(model, folds_root, tmp_path):
    a = _args("--tag", "elicit_parent", "--task", "symbol", "--metric", "kl",
              "--data-dir", str(folds_root / "fold_00"), "--results", str(tmp_path))  # fmt: skip
    (out,) = fr.run_folds(model, build_graph(model), helpers.FakeTok(), a, fr.plan_jobs(a))
    w = _args("--tag", "elicit_parent", "--metric", "kl",
              "--data-dir", str(folds_root / "fold_00"), "--results", str(tmp_path / "w"))  # fmt: skip
    (ow,) = fr.run_folds(model, build_graph(model), helpers.FakeTok(), w, fr.plan_jobs(w))
    assert not torch.equal(
        torch.load(out / "scores.pt")["mean"], torch.load(ow / "scores.pt")["mean"]
    )


def test_resume_skips_done_and_redo_recomputes(model, folds_root, tmp_path, capsys):
    a = _fold_args(folds_root, tmp_path, folds=(0, 1))
    _go(model, a)
    sc0 = tmp_path / "fold_00/klword/elicit_child/scores.pt"
    marker = sc0.stat().st_mtime_ns
    capsys.readouterr()
    _go(model, a)
    out = capsys.readouterr().out
    assert out.count("] skip (") == 2 and sc0.stat().st_mtime_ns == marker
    # remove one fold's scores -> only that fold reruns
    (tmp_path / "fold_01/klword/elicit_child/scores.pt").unlink()
    _go(model, a)
    out = capsys.readouterr().out
    assert out.count("] skip (") == 1 and "[fold_01] elicit_child" in out
    assert (tmp_path / "fold_01/klword/elicit_child/scores.pt").is_file()
    # --redo recomputes everything
    _go(model, _fold_args(folds_root, tmp_path, "elicit_child", (0, 1), "--redo"))
    assert "] skip (" not in capsys.readouterr().out
    assert sc0.stat().st_mtime_ns != marker


def test_sanity_only_stage_skip_and_no_scores(model, folds_root, tmp_path):
    a = _fold_args(folds_root, tmp_path, "elicit_child", (0,), "--stage", "sanity")
    (out,) = _go(model, a)
    assert (out / "sanity.json").is_file() and not (out / "scores.pt").exists()


@pytest.mark.parametrize("bad", ["results", "results_large"])
def test_frozen_results_refused(bad):
    with pytest.raises(SystemExit):
        _args("--tag", "elicit_child", "--data-dir", "d", "--results", str(EXP / bad))
    with pytest.raises(SystemExit):
        _args("--tag", "elicit_child", "--folds-root", "f", "--folds", "0",
              "--results-root", str(EXP / bad), "--subdir", "x")  # fmt: skip


def test_frozen_refused_for_kl_too():
    with pytest.raises(SystemExit):
        _args("--tag", "elicit_child", "--metric", "kl", "--data-dir", "d",
              "--results", str(EXP / "results"))  # fmt: skip


@pytest.mark.parametrize("tag", ["base", "fmt_parent", "teach_child"])
def test_symbol_refused_for_non_elicit(tag):
    with pytest.raises(SystemExit):
        _args("--tag", tag, "--task", "symbol", "--data-dir", "d", "--results", "/tmp/x")


def test_stage_choices_exclude_evaluate():
    with pytest.raises(SystemExit):
        _args("--tag", "base", "--stage", "evaluate", "--data-dir", "d", "--results", "/tmp/x")


@pytest.mark.parametrize(
    "extra",
    [
        [],  # neither mode
        ["--data-dir", "d"],  # single without results
        ["--folds-root", "f", "--folds", "0"],  # fold without results-root/subdir
        ["--folds-root", "f", "--results-root", "r", "--subdir", "s"],  # no folds
        ["--folds-root", "f", "--folds", "0", "--results-root", "r", "--subdir", "s",
         "--data-dir", "d"],  # mixed
        ["--data-dir", "d", "--results", "r", "--folds", "0"],  # mixed
        ["--data-dir", "d", "--results", "r", "--repro-ref", "p", "--stage", "sanity"],
    ],
)  # fmt: skip
def test_mode_flag_validation(extra):
    with pytest.raises(SystemExit):
        _args("--tag", "base", *extra)


def test_plan_jobs_paths(tmp_path):
    a = _fold_args("FR", tmp_path, folds=(0, 9))
    jobs = fr.plan_jobs(a)
    assert [j[0] for j in jobs] == ["fold_00", "fold_09"]
    assert jobs[1][1] == Path("FR/fold_09")
    assert jobs[1][2] == tmp_path / "fold_09" / "klword"


def _main(monkeypatch, model, argv):
    import transformers

    monkeypatch.setattr(run, "load_model", lambda p, d: model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *a, **k: helpers.FakeTok()))  # fmt: skip
    return fr.main(argv)


def test_main_repro_passes_against_own_output_and_fails_perturbed(
    monkeypatch, model, n_edges, folds_root, tmp_path
):
    base = ["--tag", "elicit_child", "--model", "unused", "--metric", "kl", "--device", "cpu",
            "--batch-size", str(BS), "--expect-edges", str(n_edges),
            "--data-dir", str(folds_root / "fold_00")]  # fmt: skip
    ref_dir = tmp_path / "ref"
    assert _main(monkeypatch, model, [*base, "--results", str(ref_dir)]) == 0
    ref = ref_dir / "elicit_child" / "scores.pt"
    rc = _main(monkeypatch, model, [*base, "--results", str(tmp_path / "ok"), "--redo",
                                    "--repro-ref", str(ref)])  # fmt: skip
    rep = json.loads((tmp_path / "ok/elicit_child/repro.json").read_text())
    assert rc == 0 and rep["pass"] is True and rep["max_abs_diff"] < 1e-4 and rep["tol"] == 1e-4
    # perturbed reference
    sc = torch.load(ref)
    sc["mean"] = sc["mean"] + 1e-2
    bad = tmp_path / "bad_scores.pt"
    torch.save(sc, bad)
    rc = _main(monkeypatch, model, [*base, "--results", str(tmp_path / "ko"),
                                    "--repro-ref", str(bad)])  # fmt: skip
    rep = json.loads((tmp_path / "ko/elicit_child/repro.json").read_text())
    assert rc == 1 and rep["pass"] is False and rep["max_abs_diff"] == pytest.approx(1e-2, rel=1e-2)
    # a custom tolerance flips the verdict
    rc = _main(monkeypatch, model, [*base, "--results", str(tmp_path / "tol"),
                                    "--repro-ref", str(bad), "--repro-tol", "0.1"])  # fmt: skip
    assert rc == 0


def test_main_wrong_edge_count_asserts(monkeypatch, model, folds_root, tmp_path):
    with pytest.raises(AssertionError):
        _main(monkeypatch, model, ["--tag", "base", "--model", "u", "--device", "cpu",
                                   "--data-dir", str(folds_root / "fold_00"),
                                   "--results", str(tmp_path)])  # fmt: skip


def test_main_fold_mode_end_to_end(monkeypatch, model, n_edges, folds_root, tmp_path):
    rc = _main(monkeypatch, model, [
        "--tag", "base", "--model", "u", "--device", "cpu", "--batch-size", str(BS),
        "--expect-edges", str(n_edges), "--metric", "kl", "--folds-root", str(folds_root),
        "--folds", "0", "2", "--results-root", str(tmp_path), "--subdir", "klword",
    ])  # fmt: skip
    assert rc == 0
    assert (tmp_path / "fold_00/klword/base/sanity.json").is_file()
    assert (tmp_path / "fold_02/klword/base/per_example.pt").is_file()
    assert not (tmp_path / "fold_01").exists()


# ------------------------------------------------------------------ hidden_pref


def _manual(model, val):
    """Reference computation, one row at a time, straight from model logits."""
    paper, full, top1, lpc1, lpfull = [], [], [], [], []
    with torch.no_grad():
        for i in range(val["clean_ids"].shape[0]):
            ids = val["clean_ids"][i : i + 1]
            c1, c2, k1, k2 = (int(val[k][i]) for k in ("c1", "c2", "k1", "k2"))
            lg = model(ids).logits[0].float()
            lp = lg.log_softmax(-1)
            alt = torch.cat([ids[:, :-1], torch.tensor([[k1]])], 1)
            lpk = model(alt).logits[0].float().log_softmax(-1)
            paper.append((lg[-2, c1] - lg[-2, k1]).item())
            f = lp[-2, c1] + lp[-1, c2] - (lpk[-2, k1] + lpk[-1, k2])
            full.append(f.item())
            top1.append(float(lg[-2].argmax().item() == c1))
            lpc1.append(lp[-2, c1].item())
            lpfull.append((lp[-2, c1] + lp[-1, c2]).item())
    return paper, full, top1, lpc1, lpfull


def test_hidden_pref_matches_manual(model):
    val = helpers._pairs(7, 21)  # 7 rows, bs 3 -> ragged last batch
    r = hp.hidden_pref(model, val, bs=3)
    paper, full, top1, lpc1, lpfull = map(torch.tensor, _manual(model, val))
    assert r["n"] == 7
    assert r["paper_logit_diff_mean"] == pytest.approx(paper.mean().item(), abs=1e-4)
    assert r["paper_logit_diff_sd"] == pytest.approx(paper.std().item(), abs=1e-4)
    assert r["paper_pos_frac"] == pytest.approx((paper > 0).float().mean().item())
    assert r["full_answer_logp_diff_mean"] == pytest.approx(full.mean().item(), abs=1e-4)
    assert r["full_answer_logp_diff_sd"] == pytest.approx(full.std().item(), abs=1e-4)
    assert r["full_answer_pos_frac"] == pytest.approx((full > 0).float().mean().item())
    assert r["top1_c1"] == pytest.approx(top1.mean().item())
    assert r["logp_c1_mean"] == pytest.approx(lpc1.mean().item(), abs=1e-4)
    assert r["logp_full_correct_mean"] == pytest.approx(lpfull.mean().item(), abs=1e-4)


def test_hidden_pref_batch_size_invariant(model):
    val = helpers._pairs(9, 22)
    a, b = hp.hidden_pref(model, val, bs=2), hp.hidden_pref(model, val, bs=64)
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-4)


def test_k2_changes_full_answer_but_not_paper(model):
    val = helpers._pairs(8, 23)
    r0 = hp.hidden_pref(model, val)
    val2 = dict(val, k2=(val["k2"] + 5) % (EOS))
    r1 = hp.hidden_pref(model, val2)
    assert r1["paper_logit_diff_mean"] == r0["paper_logit_diff_mean"]
    assert r1["full_answer_logp_diff_mean"] != pytest.approx(
        r0["full_answer_logp_diff_mean"], abs=1e-6
    )
    # c-side quantities untouched
    assert r1["logp_full_correct_mean"] == r0["logp_full_correct_mean"]


def test_k1_conditioning_uses_second_forward(model):
    """Changing k1 changes log p(k2 | P, k1) through the k1-prefixed forward."""
    val = helpers._pairs(8, 24)
    r0 = hp.hidden_pref(model, val)
    val2 = dict(val, k1=(val["k1"] + 2) % EOS)
    val2["k1"] = torch.where(val2["k1"] == val["c1"], (val2["k1"] + 1) % EOS, val2["k1"])
    r1 = hp.hidden_pref(model, val2)
    assert r1["full_answer_logp_diff_mean"] != pytest.approx(
        r0["full_answer_logp_diff_mean"], abs=1e-6
    )


class _Favour(torch.nn.Module):
    """Stub LM: always puts huge mass on a fixed token pair, ignoring context."""

    def __init__(self, vocab, t1, t2):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.vocab, self.t1, self.t2 = vocab, t1, t2

    def forward(self, ids):
        lg = torch.zeros(*ids.shape, self.vocab) + self.p
        lg[:, -2, self.t1] = 20.0  # answer position 1
        lg[:, -1, self.t2] = 20.0  # answer position 2
        return type("O", (), {"logits": lg})()


def test_sign_sanity_favouring_correct_is_positive():
    val = helpers._pairs(6, 25)
    val["c1"][:] = 3
    val["clean_ids"][:, -1] = 3
    val["c2"][:] = 4
    val["k1"][:] = 5
    val["k2"][:] = 6
    good = hp.hidden_pref(_Favour(EOS + 1, 3, 4), val)
    assert good["paper_logit_diff_mean"] == pytest.approx(20.0)
    assert good["paper_pos_frac"] == 1.0 and good["full_answer_pos_frac"] == 1.0
    assert good["full_answer_logp_diff_mean"] > 10 and good["top1_c1"] == 1.0
    assert good["logp_c1_mean"] > -1e-3 and good["paper_logit_diff_sd"] == pytest.approx(
        0, abs=1e-5
    )
    bad = hp.hidden_pref(_Favour(EOS + 1, 5, 6), val)  # favours the counterfactual
    assert bad["paper_logit_diff_mean"] == pytest.approx(-20.0)
    assert bad["paper_pos_frac"] == 0.0 and bad["full_answer_pos_frac"] == 0.0
    assert bad["full_answer_logp_diff_mean"] < -10 and bad["top1_c1"] == 0.0


def test_single_row_sd_is_zero_not_nan(model):
    r = hp.hidden_pref(model, helpers._pairs(1, 26))
    assert r["n"] == 1 and r["paper_logit_diff_sd"] == 0.0 and r["full_answer_logp_diff_sd"] == 0.0


def test_hidden_pref_no_grad_and_float32(model):
    hp.hidden_pref(model, helpers._pairs(3, 27))
    assert all(p.grad is None for p in model.parameters())


def _hp_args(*extra):
    return hp.parse_args(["--tag", "elicit_parent", "--model", "u", "--device", "cpu", *extra])


def test_hidden_pref_fold_mode_writes_jsons(model, folds_root, tmp_path):
    a = _hp_args("--folds-root", str(folds_root), "--folds", "0", "2",
                 "--results-root", str(tmp_path))  # fmt: skip
    hp.run_jobs(model, a, hp.plan_jobs(a))
    for k in (0, 2):
        j = json.loads((tmp_path / f"fold_{k:02d}/hidden_pref/elicit_parent.json").read_text())
        assert set(j) == {
            "tag", "model", "form", "n", "paper_logit_diff_mean", "paper_logit_diff_sd",
            "paper_pos_frac", "full_answer_logp_diff_mean", "full_answer_logp_diff_sd",
            "full_answer_pos_frac", "top1_c1", "logp_c1_mean", "logp_full_correct_mean",
            "data_dir",
        }  # fmt: skip
        assert (
            j["n"] == N_VAL
            and j["tag"] == "elicit_parent"
            and j["data_dir"].endswith(f"fold_{k:02d}")
        )
    assert not (tmp_path / "fold_01").exists()


def test_hidden_pref_word_is_default_and_recorded(model, folds_root, tmp_path):
    a = _hp_args("--data-dir", str(folds_root / "fold_00"), "--out", str(tmp_path / "w.json"))
    assert a.form == "word"
    (rec,) = hp.run_jobs(model, a, hp.plan_jobs(a))
    assert rec["form"] == "word"


def test_hidden_pref_symbol_form_reads_symbol_pairs(model, folds_root, tmp_path):
    """--form symbol scores the symbol-surface file, not the word file (different seeds
    in the fixture, so the values must differ) and writes to hidden_pref_symbol/."""
    a = _hp_args("--form", "symbol", "--folds-root", str(folds_root), "--folds", "0", "1",
                 "--results-root", str(tmp_path))  # fmt: skip
    recs = hp.run_jobs(model, a, hp.plan_jobs(a))
    for k, rec in zip((0, 1), recs):
        f = tmp_path / f"fold_{k:02d}/hidden_pref_symbol/elicit_parent.json"
        assert json.loads(f.read_text()) == rec and rec["form"] == "symbol"
        sym = run.load_data(folds_root / f"fold_{k:02d}", "symbol")["val"]
        word = run.load_data(folds_root / f"fold_{k:02d}", "word")["val"]
        assert rec["paper_logit_diff_mean"] == pytest.approx(
            hp.hidden_pref(model, sym)["paper_logit_diff_mean"]
        )
        assert rec["paper_logit_diff_mean"] != pytest.approx(
            hp.hidden_pref(model, word)["paper_logit_diff_mean"]
        )
    assert not (tmp_path / "fold_00/hidden_pref").exists()


def test_hidden_pref_rejects_unknown_form():
    with pytest.raises(SystemExit):
        _hp_args("--form", "nl", "--data-dir", "d", "--out", "o.json")


def test_hidden_pref_uses_validation_rows_only(model, folds_root, tmp_path):
    a = _hp_args("--data-dir", str(folds_root / "fold_00"), "--out", str(tmp_path / "o" / "x.json"))
    (rec,) = hp.run_jobs(model, a, hp.plan_jobs(a))
    val = run.load_data(folds_root / "fold_00", "word")["val"]
    assert rec["n"] == N_VAL
    assert rec["paper_logit_diff_mean"] == pytest.approx(
        hp.hidden_pref(model, val)["paper_logit_diff_mean"]
    )
    assert json.loads((tmp_path / "o/x.json").read_text())["n"] == N_VAL


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--data-dir", "d"],
        ["--out", "o"],
        ["--folds-root", "f", "--folds", "0"],
        ["--folds-root", "f", "--results-root", "r"],
        ["--folds-root", "f", "--folds", "0", "--results-root", "r", "--out", "o"],
        ["--data-dir", "d", "--out", "o", "--folds", "1"],
    ],
)
def test_hidden_pref_mode_validation(extra):
    with pytest.raises(SystemExit):
        _hp_args(*extra)
