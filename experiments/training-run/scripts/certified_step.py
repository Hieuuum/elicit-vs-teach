"""Selection rule for the ts38 certified LoRA parent (decisions.md 2026-08-15,
"certified-checkpoint parent" entry; owner pre-authorized 2026-08-15).

The pre-registered LoRA-parent protocol (launch_ts38_lora_parent.sh) HALTed:
every CONVERGED parent (full-FT or LoRA) crossed the G8 retention bar. The
probe (launch_ts38_lora_probe.sh) replays the 3e-4 run once with snapshots
every 1000 steps and scores each one G1 + G8 --no-record into
results/ts38_parent_lora_probe.json. This module is the pure selection
function over that table — no argv, no file I/O, no network — so it is
testable without a probe run. launch_ts38_certified_parent.sh calls it via a
python heredoc (`from certified_step import select_certified_step`, resolved
via cwd=scripts/ the same way launch_ts38_mini.sh's heredocs already do
`from train import load_config`).

Rule: the EARLIEST step S with g1_pass and g8_pass at S, AND g1_pass at
S+1000 (persistence — the val curve is noisy at 1k granularity, so a lone
both-pass reading is not trusted on its own). A both-pass step with no
S+1000 row (e.g. the final probed step) cannot satisfy persistence and does
not qualify. Neither bar is touched here; this module only picks a step
among rows that already carry pass/fail verdicts computed elsewhere.
"""

from __future__ import annotations


def select_certified_step(rows: list[dict]) -> dict:
    """The earliest persistently-certified step, plus diagnostic fields.

    ``rows`` is the probe table (a list of dicts each carrying at least
    ``step``, ``g1_pass``, ``g8_pass``) in any order — duplicated steps are
    not expected (the probe overwrites in place) and are not de-duplicated
    here. Returns:

    - ``step``: the winning step, or ``None`` if no step satisfies the rule.
    - ``first_g1_pass_step``: the earliest step with ``g1_pass`` true, or
      ``None`` if no row passes G1.
    - ``last_g8_pass_step``: the latest step with ``g8_pass`` true, or
      ``None`` if no row passes G8.
    - ``both_pass_steps``: every step where both ``g1_pass`` and ``g8_pass``
      are true, sorted ascending (independent of the persistence rule —
      diagnostic only).
    """
    by_step = {row["step"]: row for row in rows}
    steps = sorted(by_step)

    g1_pass_steps = [s for s in steps if by_step[s].get("g1_pass")]
    g8_pass_steps = [s for s in steps if by_step[s].get("g8_pass")]
    both_pass_steps = [s for s in steps if by_step[s].get("g1_pass") and by_step[s].get("g8_pass")]

    step = None
    for s in steps:
        row = by_step[s]
        if not (row.get("g1_pass") and row.get("g8_pass")):
            continue
        next_row = by_step.get(s + 1000)
        if next_row is not None and next_row.get("g1_pass"):
            step = s
            break

    return {
        "step": step,
        "first_g1_pass_step": min(g1_pass_steps) if g1_pass_steps else None,
        "last_g8_pass_step": max(g8_pass_steps) if g8_pass_steps else None,
        "both_pass_steps": both_pass_steps,
    }
