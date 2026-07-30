"""V5.74 — geode.edl G5 leak bar + fixed eval-protocol constants.

The leak bar is inclusive at 0.02 (a zero-shot exact-match rate at or below the
threshold has not leaked the target mapping); the eval-protocol constants are the
stopping-block split point and the G5 shot count.
"""

from __future__ import annotations

from geode.edl import EVAL_STOP_ROWS, G5_N_SHOTS, g5_leak_ok


def test_v5_74_leak_bar_boundary() -> None:
    assert g5_leak_ok(0.02) is True  # inclusive: exactly the threshold passes
    assert g5_leak_ok(0.0) is True
    assert g5_leak_ok(0.0199) is True
    assert g5_leak_ok(0.0201) is False
    assert g5_leak_ok(0.03) is False


def test_v5_74_protocol_constants() -> None:
    assert EVAL_STOP_ROWS == 2048
    assert G5_N_SHOTS == 16
