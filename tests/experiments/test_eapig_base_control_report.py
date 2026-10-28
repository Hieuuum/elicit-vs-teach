"""Smoke tests for experiments/eapig-circuit-check/base_control_report.py (synthetic inputs)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[2] / "experiments/eapig-circuit-check/base_control_report.py"
_spec = importlib.util.spec_from_file_location("base_control_report", _P)
bcr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bcr)

K = 100_000  # large k so a 3-decimal Jaccard round-trips through an integer overlap


def _ov(j: float) -> int:
    return round(2 * K * j / (1 + j))


def _entry(js: tuple[float, ...]) -> dict:
    rows = {}
    for f, j in zip(bcr.FRACS, js, strict=True):
        rows[bcr.frac_key(f)] = {
            "overlap": _ov(j),
            "k": K,
            "null": {"n1": {"edge": {"p5": 0.99}}},
        }
    return {"rankings": {"signed": rows}}


def make(ref_scale: float = 1.0) -> tuple[dict, dict]:
    pairs = {}
    for pair, _, _ in bcr.PAIRS:
        exp = bcr.EXPECTED_REF.get(pair, (0.3,) * 6)
        pairs[pair] = _entry(tuple(x * ref_scale if pair in bcr.EXPECTED_REF else x for x in exp))
    for lab in bcr.LABELS:
        pairs[f"{lab}:halves"] = _entry((0.9,) * 6)
    en = {"pairs": pairs}
    boot = {"meta": {}, "pairs": {}}  # overlap_boot.json layout: pairs."A:B".str(frac)
    for pair in pairs:
        if pair.endswith(":halves"):
            continue
        boot["pairs"][pair] = {
            str(f): {
                "point": bcr.edge_jaccard(en, pair, f),
                "ci_lo": 0.1,
                "ci_hi": 0.9,
            }
            for f in bcr.FRACS
        }
    return en, boot


def test_jaccard_formula():
    assert bcr.jaccard(50, 100) == pytest.approx(1 / 3)
    assert bcr.jaccard(100, 100) == 1.0
    assert bcr.jaccard(0, 100) == 0.0


def test_chance_uses_n1_p5():
    en, _ = make()
    # overlap = K * (1 - 0.99) = 0.01 K  ->  J = 0.01 / 1.99
    assert bcr.chance_jaccard(en, 0.01) == pytest.approx(0.01 / 1.99)


def test_table_format():
    en, boot = make()
    t = bcr.build_table(en, boot)
    assert "| base → ec word | 0.300 [0.100, 0.900]" in t
    assert "| split-half ceiling ep op | 0.900 |" in t
    assert t.splitlines()[-1].startswith("| chance (95th pct) | 0.005 |")


def test_checks_pass_on_matching_input():
    en, boot = make()
    assert all(ok for ok, _ in bcr.run_checks(en, boot))


def test_checks_fail_on_wrong_reference():
    en, boot = make(ref_scale=0.9)
    assert not all(ok for ok, _ in bcr.run_checks(en, boot))


def test_checks_fail_on_boot_point_mismatch():
    en, boot = make()
    boot["pairs"]["base:ec_word"]["0.01"]["point"] += 0.01
    res = bcr.run_checks(en, boot)
    assert not all(ok for ok, _ in res)


def test_main_end_to_end(tmp_path, capsys):
    en, boot = make()
    (tmp_path / "en.json").write_text(json.dumps(en))
    (tmp_path / "b.json").write_text(json.dumps(boot))
    argv = [
        "--edge-node",
        str(tmp_path / "en.json"),
        "--boot",
        str(tmp_path / "b.json"),
        "--out-md",
        str(tmp_path / "t.md"),
        "--fig",
        str(tmp_path / "f.png"),
    ]
    assert bcr.main(argv) == 0
    assert (tmp_path / "t.md").exists() and (tmp_path / "f.png").stat().st_size > 0
    assert "FAIL" not in capsys.readouterr().out


# ---------------------------------------------------------------- nine sizes (tiny fractions)

TINY = (0.0001, 0.0002, 0.0005)
NINE = TINY + bcr.FRACS


def _entry_at(js_by_f: dict[float, float], k_by_f: dict[float, int] | None = None) -> dict:
    rows = {}
    for f, j in js_by_f.items():
        k = (k_by_f or {}).get(f, K)
        ov = round(2 * k * j / (1 + j))
        rows[bcr.frac_key(f)] = {"overlap": ov, "k": k, "null": {"n1": {"edge": {"p5": 0.99}}}}
    return {"rankings": {"signed": rows}}


def make_nine(fracs=NINE, tiny_j: float = 0.5) -> tuple[dict, dict]:
    """Like `make`, over `fracs`: PLAN values at the six old sizes, `tiny_j` elsewhere."""
    pairs = {}
    for pair, _, _ in bcr.PAIRS:
        exp = bcr.EXPECTED_REF.get(pair)
        pairs[pair] = _entry_at(
            {f: (exp[bcr.FRACS.index(f)] if exp and f in bcr.FRACS else tiny_j) for f in fracs}
        )
    for lab in bcr.LABELS:
        pairs[f"{lab}:halves"] = _entry_at(dict.fromkeys(fracs, 0.9))
    en = {"pairs": pairs}
    boot = {"meta": {}, "pairs": {}}
    for pair in pairs:
        if pair.endswith(":halves"):
            continue
        boot["pairs"][pair] = {
            str(f): {"point": bcr.edge_jaccard(en, pair, f), "ci_lo": 0.1, "ci_hi": 0.9}
            for f in fracs
        }
    return en, boot


def test_frac_key_and_str_never_use_exponent_for_the_nine_sizes():
    for f in NINE:
        assert "e" not in bcr.frac_key(f) and "e" not in str(f)
        assert float(bcr.frac_key(f)[len("frac_") :]) == f
    assert bcr.frac_key(0.0001) == "frac_0.0001"
    assert bcr.frac_key(0.0005) == "frac_0.0005"


def test_size_label_all_nine():
    got = [bcr.size_label(f) for f in sorted(NINE)]
    assert got == ["0.01%", "0.02%", "0.05%", "0.1%", "0.2%", "0.5%", "1%", "2%", "5%"]


def test_sizes_in_is_ascending_regardless_of_key_order():
    en, _ = make_nine(fracs=(0.05, 0.0002, 0.01, 0.0001))
    assert bcr.sizes_in(en) == [0.0001, 0.0002, 0.01, 0.05]


def test_sizes_in_is_the_intersection_across_pairs():
    en, _ = make_nine()
    del en["pairs"]["base:ec_word"]["rankings"]["signed"]["frac_0.0001"]
    assert bcr.sizes_in(en) == sorted(NINE)[1:]


def test_sizes_in_ignores_non_frac_keys_and_empty_input():
    en, _ = make_nine(fracs=(0.01, 0.02))
    for v in en["pairs"].values():
        v["rankings"]["signed"]["top_100"] = {"overlap": 1, "k": 100}
    assert bcr.sizes_in(en) == [0.01, 0.02]
    assert bcr.sizes_in({"pairs": {}}) == []


def test_six_size_input_unchanged_by_discovery():
    en, boot = make()
    assert bcr.sizes_in(en) == list(bcr.FRACS)
    assert all(ok for ok, _ in bcr.run_checks(en, boot))
    assert bcr.build_table(en, boot) == bcr.build_table(en, boot, list(bcr.FRACS))


def test_table_nine_columns_ascending_with_labels():
    en, boot = make_nine()
    lines = bcr.build_table(en, boot).splitlines()
    assert lines[0] == "| pair | 0.01% | 0.02% | 0.05% | 0.1% | 0.2% | 0.5% | 1% | 2% | 5% |"
    assert lines[1] == "|---|" + "---|" * 9
    for line in lines[2:]:
        assert line.count("|") == 11, line  # 1 name + 9 cells -> 11 pipes
    row = next(x for x in lines if x.startswith("| ep op → ec word"))
    cells = [c.strip() for c in row.strip("|").split("|")[1:]]
    assert cells[0].startswith("0.500") and cells[3].startswith("0.766")  # 0.01% , 0.1%
    assert cells[-1].startswith("0.497")


def test_table_chance_row_uses_each_sizes_own_k():
    en, boot = make_nine()
    for pair in en["pairs"].values():  # the N1 null carries k per size
        for f in TINY:
            pair["rankings"]["signed"][bcr.frac_key(f)]["k"] = 20
            pair["rankings"]["signed"][bcr.frac_key(f)]["overlap"] = 10
    # k=20, p5=0.99 -> overlap 0.2 -> J = 0.2 / 39.8
    assert bcr.chance_jaccard(en, 0.0001) == pytest.approx(0.2 / 39.8)
    assert bcr.chance_jaccard(en, 0.001) == pytest.approx(0.01 / 1.99)


def test_checks_nine_sizes_pass_and_name_skipped_sizes():
    en, boot = make_nine()
    res = bcr.run_checks(en, boot)
    assert all(ok for ok, _ in res), res
    ref_msgs = [m for _, m in res if m.startswith("reference")]
    assert len(ref_msgs) == len(bcr.EXPECTED_REF)
    for m in ref_msgs:
        assert "skipped (not in PLAN table): ['0.01%', '0.02%', '0.05%']" in m
        assert "got (" in m
    # six-size input has nothing to skip
    assert not any("skipped" in m for _, m in bcr.run_checks(*make()))


def test_checks_nine_sizes_still_fail_when_an_old_size_drifts():
    en, boot = make_nine()
    rows = en["pairs"]["ep_sym:ec_word"]["rankings"]["signed"]
    rows[bcr.frac_key(0.01)]["overlap"] = _ov(0.5)  # PLAN says 0.731
    res = bcr.run_checks(en, boot)
    bad = [m for ok, m in res if not ok]
    assert any(m.startswith("reference ep_sym:ec_word") for m in bad)  # PLAN check fires
    assert all("ep_sym:ec_word" in m for m in bad)  # and nothing else is blamed


def test_checks_tiny_sizes_never_checked_against_the_table():
    en, boot = make_nine(tiny_j=0.123)  # arbitrary values at the tiny sizes
    assert all(ok for ok, _ in bcr.run_checks(en, boot))


def test_checks_boot_point_mismatch_at_a_tiny_size_is_caught():
    en, boot = make_nine()
    boot["pairs"]["base:ec_word"]["0.0001"]["point"] += 0.01
    res = bcr.run_checks(en, boot)
    bad = [m for ok, m in res if not ok]
    assert len(bad) == 1 and "base:ec_word" in bad[0]


def test_checks_boot_missing_at_a_tiny_size_is_flagged():
    en, boot = make_nine()
    del boot["pairs"]["ep_word:ec_word"]["0.0005"]
    bad = [m for ok, m in bcr.run_checks(en, boot) if not ok]
    assert len(bad) == 1 and "missing sizes" in bad[0]


def test_checks_fail_when_no_plan_size_is_present():
    en, boot = make_nine(fracs=TINY)
    res = bcr.run_checks(en, boot)
    bad = [m for ok, m in res if not ok]
    assert len(bad) == len(bcr.EXPECTED_REF)
    assert all("no PLAN-table size present" in m for m in bad)


def test_checks_subset_of_old_sizes_compares_matching_columns():
    en, boot = make_nine(fracs=(0.001, 0.01, 0.05, 0.0002))
    res = bcr.run_checks(en, boot)
    assert all(ok for ok, _ in res), res
    msg = next(m for _, m in res if m.startswith("reference ep_sym:ec_word"))
    assert "(0.766, 0.731, 0.497)" in msg


def test_main_end_to_end_nine_sizes(tmp_path, capsys):
    en, boot = make_nine()
    (tmp_path / "en.json").write_text(json.dumps(en))
    (tmp_path / "b.json").write_text(json.dumps(boot))
    argv = ["--edge-node", str(tmp_path / "en.json"), "--boot", str(tmp_path / "b.json"),
            "--out-md", str(tmp_path / "t.md"), "--fig", str(tmp_path / "f.png")]
    assert bcr.main(argv) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and "skipped (not in PLAN table)" in out
    assert (tmp_path / "f.png").stat().st_size > 0
    assert (tmp_path / "t.md").read_text().splitlines()[0].startswith("| pair | 0.01% |")


def test_main_exits_nonzero_when_old_size_reference_drifts_in_nine_size_input(tmp_path):
    en, boot = make_nine()
    en["pairs"]["fp_word:tc_word"]["rankings"]["signed"][bcr.frac_key(0.002)]["overlap"] = _ov(0.9)
    (tmp_path / "en.json").write_text(json.dumps(en))
    (tmp_path / "b.json").write_text(json.dumps(boot))
    argv = ["--edge-node", str(tmp_path / "en.json"), "--boot", str(tmp_path / "b.json"),
            "--fig", str(tmp_path / "f.png")]
    assert bcr.main(argv) != 0


def test_main_rejects_input_without_common_sizes(tmp_path):
    (tmp_path / "en.json").write_text(json.dumps({"pairs": {}}))
    (tmp_path / "b.json").write_text(json.dumps({"pairs": {}}))
    with pytest.raises(SystemExit):
        bcr.main(["--edge-node", str(tmp_path / "en.json"), "--boot", str(tmp_path / "b.json"),
                  "--fig", str(tmp_path / "f.png")])
