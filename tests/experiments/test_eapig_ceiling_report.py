"""ceiling_report.py: split-half ceilings on the carry-pattern set (smoke + edge cases)."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest
import torch

from tests._scriptloader import repo_root

EXP = repo_root() / "experiments" / "eapig-circuit-check"
sys.path.insert(0, str(EXP))
_spec = importlib.util.spec_from_file_location("eapig_ceiling_report", EXP / "ceiling_report.py")
cr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cr
_spec.loader.exec_module(cr)

E = 20_000


def test_identical_halves_give_ceiling_one():
    s = torch.randn(E, generator=torch.Generator().manual_seed(0))
    assert all(v == pytest.approx(1.0) for v in cr.ceilings(s, s.clone()).values())


def test_independent_halves_sit_near_chance():
    g = torch.Generator().manual_seed(1)
    c = cr.ceilings(torch.randn(E, generator=g), torch.randn(E, generator=g), (0.05,))
    ch = cr.chance_jaccard(1000, E)
    assert abs(c["0.05"] - ch) < 0.02


def test_noisy_halves_between_chance_and_one_and_noise_lowers_it():
    g = torch.Generator().manual_seed(2)
    sig = torch.randn(E, generator=g)
    lo = cr.ceilings(
        sig + 0.3 * torch.randn(E, generator=g), sig + 0.3 * torch.randn(E, generator=g)
    )
    hi = cr.ceilings(
        sig + 0.05 * torch.randn(E, generator=g), sig + 0.05 * torch.randn(E, generator=g)
    )
    for f in lo:
        assert cr.chance_jaccard(max(1, round(float(f) * E)), E) < lo[f] < hi[f] <= 1.0


def test_ranking_is_signed_not_absolute():
    # half b negates every score: signed top-k sets are disjoint at small k
    s = torch.randn(E, generator=torch.Generator().manual_seed(3))
    assert cr.ceilings(s, -s, (0.01,))["0.01"] == 0.0


def test_mismatched_sizes_raise():
    with pytest.raises(ValueError, match="half sizes differ"):
        cr.ceilings(torch.zeros(10), torch.zeros(11))


def test_chance_formula():
    assert cr.chance_jaccard(E, E) == pytest.approx(1.0)
    assert cr.chance_jaccard(100, E) == pytest.approx(0.5 / (200 - 0.5))


def test_report_reads_scores_and_sanity_and_prints_table(tmp_path, capsys):
    g = torch.Generator().manual_seed(4)
    for sub, em in (("klword/base", 0.5), ("klword_hard_acc/elicit_child", 0.97)):
        d = tmp_path / sub
        d.mkdir(parents=True)
        a = torch.randn(E, generator=g)
        torch.save(
            {"mean": a, "mean_a": a, "mean_b": a + 0.1 * torch.randn(E, generator=g)},
            d / "scores.pt",
        )
        (d / "sanity.json").write_text(json.dumps({"exact_match": em}))
    out = tmp_path / "c.json"
    cr.main(
        [
            "--root",
            str(tmp_path),
            "base=klword/base",
            "ec_acc=klword_hard_acc/elicit_child",
            "--out",
            str(out),
        ]
    )
    rep = json.loads(out.read_text())
    assert set(rep["models"]) == {"base", "ec_acc"} and rep["n_edges"] == E
    assert rep["models"]["ec_acc"]["exact_match"] == 0.97
    assert set(rep["chance"]) == {str(f) for f in cr.FRACS}
    txt = capsys.readouterr().out
    assert "| ec_acc | 0.970 |" in txt and "| chance |" in txt
