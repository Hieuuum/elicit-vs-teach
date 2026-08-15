"""``edl_converged_val_floor.py`` FAMILIES regexes and arm-label mappings.

Focused on the ts38/ts38mw pair (EXPERIMENTS §6.14/§6.15): ts38mw's base
arm is the SAME ``evt-ts38-base-n<size>`` id the ts38 family reads (reused,
not retrained), while its pretaught arm is a NEW ``evt-ts38mw-pretaught-n
<size>`` id. The regex-level guarantee that matters is that neither family
ever picks up the other's own pretaught arm, and that a stray "-mw-"
infix or missing size never slips through — exercised here as a match/
no-match matrix rather than by driving the full ``collect()`` pipeline
(which needs a populated store; see ``test_dataset_size_sweep.py`` for that
style of test on the sibling driver).

CPU-only, no store, no network.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests._scriptloader import load

ecvf = load("edl_converged_val_floor")

# (family, run_id, expected groups or None if the family's regex must NOT match)
_MATRIX = [
    ("ts38", "evt-ts38-base-n1000", ("base", "1000")),
    ("ts38", "evt-ts38-pretaught-n1000", ("pretaught", "1000")),
    ("ts38", "evt-ts38mw-pretaught-n1000", None),
    ("ts38", "evt-ts38mw-base-n1000", None),
    ("ts38", "evt-ts38-base-n1000-extra", None),
    ("ts38", "evt-ts38-parent-loraprobe-lr3e-4", None),
    ("ts38", "evt-ts38mw-parent-probe-lr3e-4", None),
    ("ts38mw", "evt-ts38-base-n1000", ("base", "1000")),
    ("ts38mw", "evt-ts38-pretaught-n1000", None),
    ("ts38mw", "evt-ts38mw-pretaught-n1000", ("pretaught", "1000")),
    ("ts38mw", "evt-ts38mw-base-n1000", None),
    ("ts38mw", "evt-ts38-base-n1000-extra", None),
    ("ts38mw", "evt-ts38-parent-loraprobe-lr3e-4", None),
    ("ts38mw", "evt-ts38mw-parent-probe-lr3e-4", None),
]


@pytest.mark.parametrize(("family", "run_id", "expected"), _MATRIX)
def test_family_regex_matrix(family: str, run_id: str, expected: tuple[str, str] | None) -> None:
    pattern = ecvf.FAMILIES[family][0]
    match = pattern.match(run_id)
    if expected is None:
        assert match is None, f"{family} must NOT match {run_id!r}"
    else:
        assert match is not None, f"{family} must match {run_id!r}"
        assert (match.group(1), match.group(2)) == expected


def test_ts38_and_ts38mw_regexes_agree_on_the_shared_base_arm() -> None:
    """The one id both families are meant to pick up must parse identically
    under each family's own pattern — that's what makes the base arm safe
    to reuse across families."""
    run_id = "evt-ts38-base-n21544"
    ts38_match = ecvf.FAMILIES["ts38"][0].match(run_id)
    ts38mw_match = ecvf.FAMILIES["ts38mw"][0].match(run_id)
    assert ts38_match is not None and ts38mw_match is not None
    assert (ts38_match.group(1), ts38_match.group(2)) == ("base", "21544")
    assert (ts38mw_match.group(1), ts38mw_match.group(2)) == ("base", "21544")


def test_all_family_stems_distinct() -> None:
    stems = [ecvf.FAMILIES[f][1] for f in ecvf.FAMILIES]
    assert len(stems) == len(set(stems))
    assert "ts38mw" in ecvf.FAMILIES
    assert ecvf.FAMILIES["ts38mw"][1] == "edl_converged_val_floor_ts38mw"
    assert ecvf.FAMILIES["ts38mw"][1] != ecvf.FAMILIES["ts38"][1]


def test_arm_label_mappings_are_honest_and_distinct() -> None:
    """ts38's pretaught arm reads 'pre-taught (elicit)'; ts38mw's own
    pretaught-mw arm reads 'pre-taught-mw (elicit)' — the two must never
    collide, and the shared base label must read the same in both."""
    assert ecvf.TS38_ARM["pretaught"][1] == "pre-taught (elicit)"
    assert ecvf.TS38MW_ARM["pretaught"][1] == "pre-taught-mw (elicit)"
    assert ecvf.TS38_ARM["pretaught"][1] != ecvf.TS38MW_ARM["pretaught"][1]
    assert ecvf.TS38_ARM["base"][1] == ecvf.TS38MW_ARM["base"][1] == "base (teach)"
    # Both map to the canonical noinst/inst condition every downstream
    # lookup (STYLE, groupby) keys on.
    assert ecvf.TS38_ARM["base"][0] == ecvf.TS38MW_ARM["base"][0] == "noinst"
    assert ecvf.TS38_ARM["pretaught"][0] == ecvf.TS38MW_ARM["pretaught"][0] == "inst"


def test_collect_ts38mw_on_empty_store_returns_empty_dataframe_not_raise(tmp_path: Path) -> None:
    """A store whose runs/ dir exists but holds nothing matching (e.g. before
    any ts38mw run has been pushed) must come back as an empty table, not a
    KeyError from sorting a columnless DataFrame."""
    (tmp_path / "runs").mkdir(parents=True)

    df = ecvf.collect("ts38mw", tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert df.empty
