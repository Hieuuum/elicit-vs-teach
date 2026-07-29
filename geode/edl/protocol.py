"""Fixed eval-protocol constants + the G5 leak bar (spec 02 §5, §8; V5.74).

The frozen shared eval file (``D_target_eval``) is split into a fixed stopping
block and a reporting block; ``EVAL_STOP_ROWS`` is that split point, identical for
every run and every n, so no reported number ever touches the rows that drove a
stop decision. ``G5_N_SHOTS`` is the G5 few-shot count (protocol, not a knob).
``g5_leak_ok`` is the leak bar the random-label (G3) and warm-start zero-shot
checks share: a zero-shot exact-match rate at or below the threshold has not
leaked the target mapping.
"""

from __future__ import annotations

# Rows [0, EVAL_STOP_ROWS) of the frozen eval file are the in-loop ε/k stopping
# block; everything after is the reporting block (θ_T test loss + G5). Identical
# for every run and every n (spec 02 §5, owner 2026-07-22).
EVAL_STOP_ROWS = 2048

# G5 is zero/16-shot (spec 02 §8): the shot count is protocol, not a knob.
G5_N_SHOTS = 16

# Exact match on signed multi-digit integers has ~0 chance rate, so 0.02 is the
# margin under which a zero-shot rate counts as "no leak" (spec 02 §8 G3).
G5_LEAK_THRESHOLD = 0.02


def g5_leak_ok(zero_shot: float) -> bool:
    """True iff a zero-shot exact-match rate is within the leak bar (≤ 0.02; V5.74).

    The random-label installer (G3) and warm-start leak checks must NOT have
    taught the target mapping; a rate at or below the threshold passes, above it
    fails.
    """
    return zero_shot <= G5_LEAK_THRESHOLD
