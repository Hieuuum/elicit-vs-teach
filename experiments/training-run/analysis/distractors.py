"""Distractor choice for the hidden-preference and lens reads (2026-11-19).

The original reads compared the answer's first token with that of the next problem's answer (cyclic), whatever its
length. A model that only knows the answer's magnitude (its number of digits, from the operands) then already
prefers the right first token. The matched distractor is the nearest following problem whose answer has the same
sign and number of digits and a different first token, so only knowledge of the value itself can win.
"""

from __future__ import annotations

from collections.abc import Sequence


def answer_kind(ans: int) -> tuple[bool, int]:
    return ans < 0, len(str(abs(ans)))


def matched_distractors(answers: Sequence[int], targets: Sequence[int]) -> list[int | None]:
    """For item i: the index j of the nearest following item (cyclic) with the same answer kind (sign, digit
    count) and a different scored token; None if there is none."""
    n, out = len(answers), []
    for i in range(n):
        j = next(
            (
                (i + k) % n
                for k in range(1, n)
                if answer_kind(answers[(i + k) % n]) == answer_kind(answers[i])
                and targets[(i + k) % n] != targets[i]
            ),
            None,
        )
        out.append(j)
    return out


def next_distractors(targets: Sequence[int]) -> list[int | None]:
    """The original choice: the nearest following item with a different scored token."""
    n = len(targets)
    return [
        next(((i + k) % n for k in range(1, n) if targets[(i + k) % n] != targets[i]), None)
        for i in range(n)
    ]
