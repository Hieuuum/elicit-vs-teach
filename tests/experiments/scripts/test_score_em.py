"""score_em.py ``breakdown``: per-(size, op) accuracy for the ts1b children.

Silent-failure risk: a miscount here would misreport where the two children
differ at matched overall accuracy (the per-size check the owner asked for).
"""

from __future__ import annotations

import json

from tests._scriptloader import load

se = load("score_em")


def test_breakdown_counts_and_accuracy():
    got = se.breakdown([True, False, True, True], ["1x4", "1x4", "4x4", "4x4"])
    assert got == {"1x4": {"acc": 0.5, "n": 2}, "4x4": {"acc": 1.0, "n": 2}}


def test_breakdown_sorted_and_json_safe():
    got = se.breakdown([True, False], [(4, 4), (1, 4)])
    assert list(got) == ["(1, 4)", "(4, 4)"]
    json.dumps(got)


def test_breakdown_all_wrong_is_zero():
    assert se.breakdown([False] * 3, ["+"] * 3) == {"+": {"acc": 0.0, "n": 3}}


def test_breakdown_weighted_mean_equals_overall():
    hits = [True, False, True, False, False, True, True]
    keys = ["a", "a", "b", "b", "b", "c", "c"]
    got = se.breakdown(hits, keys)
    overall = sum(v["acc"] * v["n"] for v in got.values()) / len(hits)
    assert abs(overall - sum(hits) / len(hits)) < 1e-12
