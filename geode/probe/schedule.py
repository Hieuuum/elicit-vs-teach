"""Snapshot-step scheduler for runs 5-6 (specs/02 §6 "Runs 5-6", §7).

Spec words: "1024 steps, log-spaced early (every step through ~30, then
stretching), uniform later" / "strictly increasing, includes first and final
step, dense unit-stride prefix, log-then-uniform tail". Exact parameters are
OPEN(4); where the spec is silent this module picks the simplest deterministic
scheme honoring those words — see ``snapshot_steps``.

The schedule is written into the run manifest before launch; a silently wrong
schedule means missing checkpoints on a paid GPU box, hence tested core
(V5.55-V5.61).
"""

from __future__ import annotations


def snapshot_steps(total_steps: int, n: int = 1024, dense_until: int = 30) -> list[int]:
    """Return exactly ``min(n, total_steps)`` strictly increasing snapshot steps.

    Every returned step is an int in ``[1, total_steps]``. Exact scheme:

    * ``n >= total_steps``: every step ``1..total_steps`` (the budget covers the
      whole run; dense prefix, first and final step all follow trivially).
    * Otherwise, with ``d = min(dense_until, n - 1)`` (the clamp always reserves
      at least one slot for the final step, which outranks the prefix): the
      dense unit-stride prefix ``1..d``, then the remaining ``k = n - d`` slots
      fill ``(d, total_steps]`` greedily. With ``prev`` the last placed step and
      ``q`` slots remaining, the next step is the **earlier** of

      - the geometric proposal ``round(prev * ratio)`` where
        ``ratio = (total_steps / max(d, 1)) ** (1 / k)`` — the ideal log-spaced
        schedule over the region — and
      - the uniform completion ``prev + ceil((total_steps - prev) / q)``,

      never below ``prev + 1``, and the last slot is ``total_steps`` exactly.

    Early on the geometric proposal wins (near-unit stride, gaps stretching
    multiplicatively); once its gap exceeds the uniform completion's, the
    completion takes over and the tail advances in ~equal gaps to
    ``total_steps`` — the spec's log-then-uniform tail. Gaps never shrink by
    more than 1 (rounding jitter).

    Purely deterministic (integer/float arithmetic only, no randomness, no
    seed). Degenerate corner: ``n == 1`` with ``total_steps > 1`` returns
    ``[total_steps]`` — the final snapshot outranks step 1.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if dense_until < 0:
        raise ValueError(f"dense_until must be >= 0, got {dense_until}")

    m = min(n, total_steps)
    if m == total_steps:
        return list(range(1, total_steps + 1))

    d = min(dense_until, m - 1)  # reserve >= 1 slot for the final step
    k = m - d
    ratio = (total_steps / max(d, 1)) ** (1.0 / k)

    steps = list(range(1, d + 1))
    prev = d
    for q in range(k, 0, -1):  # q = slots remaining, current one included
        if q == 1:
            nxt = total_steps
        else:
            remaining = total_steps - prev
            geometric = round(prev * ratio)
            uniform = prev + -(-remaining // q)  # ceil division, int-exact
            nxt = max(prev + 1, min(geometric, uniform))
        steps.append(nxt)
        prev = nxt
    return steps
