"""Unit test for gates.py's record-path decision (V0.6 shared-parent guard).

``gate_record_decision`` is the pure function behind --no-record and
--record-only-pass (2026-07-30 fig2-installer review): a launcher scores a
shared parent with --no-record first, then re-runs the SAME gate without it
to persist the verdict. Without --record-only-pass that second call
recomputes its own pass/fail and writes it unconditionally, so a
near-threshold recompute divergence from the --no-record score could still
land a recorded FAIL on the shared parent's manifest — exactly what
require_parent_ready (V0.6) then reads as "refuse every child of this
parent". This is a smoke/property test on the pure decision function only:
no checkpoint, tokenizer, or manifest needed, matching the CLAUDE.md testing
policy's "code whose *silent* failure would waste GPU budget" bar (a poisoned
shared parent is exactly that).
"""

from __future__ import annotations

import pytest

from tests._scriptloader import load

gates = load("gates")


@pytest.mark.parametrize(
    ("passed", "no_record", "record_only_pass", "expect_write", "expect_code"),
    [
        # Default (both flags False): always records, whatever the verdict —
        # existing callers rely on this (e.g. a deliberately-recorded FAIL).
        (True, False, False, True, 0),
        (False, False, False, True, 1),
        # --no-record: never writes, whatever the verdict.
        (True, True, False, False, 0),
        (False, True, False, False, 1),
        # --record-only-pass: writes on pass, writes NOTHING on fail (the
        # divergence guard) and still reports failure via a nonzero exit.
        (True, False, True, True, 0),
        (False, False, True, False, 1),
    ],
)
def test_gate_record_decision(
    passed: bool, no_record: bool, record_only_pass: bool, expect_write: bool, expect_code: int
) -> None:
    should_record, exit_code = gates.gate_record_decision(passed, no_record, record_only_pass)
    assert should_record == expect_write
    assert exit_code == expect_code


def test_no_record_and_record_only_pass_are_mutually_exclusive_at_parse_time() -> None:
    parser = gates.build_parser()
    for gate_argv in (
        ["g2", "--run", "x", "--config", "y"],
        ["g4", "--run", "x", "--config", "y"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(gate_argv + ["--no-record", "--record-only-pass"])
        # Each flag alone still parses fine (regression guard on the group).
        parser.parse_args(gate_argv + ["--no-record"])
        parser.parse_args(gate_argv + ["--record-only-pass"])
