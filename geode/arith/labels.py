"""Random-label rule for the format-installer runs (spec 02 §5; OPEN(6)).

Runs 3/4 install the operator-notation *format* using **wrong** labels, so the
model cannot learn arithmetic from them — that is what keeps Arm B's later
learning attributable to the target run alone. If a "random" label leaked
operand information, the installer would secretly teach arithmetic and the
elicit-vs-teach comparison would be invalid, with nothing crashing. Hence this
rule is tested core.

Default (OPEN(6)): the label is uniform over integers with the **same digit
count and sign as the true answer**, and depends on the operands *only* through
that digit count and sign — never on the specific ``a``/``b``. Seeding is by
``(seed, index)`` so the label sequence is reproducible and, at a fixed index,
invariant to which operands produced a same-shape true answer.
"""

from __future__ import annotations

import hashlib
import random
from typing import Sequence

from geode.arith.formats import digits


def _rng(seed: int, index: int) -> random.Random:
    h = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def permute_labels(true_answers: Sequence[int], seed: int) -> list[int]:
    """The true answers shuffled across positions (spec 02 §5, permuted mode).

    Position ``i`` is shown some other position's true answer: individually
    wrong up to chance collisions (two questions sharing an answer), while the
    *multiset* of shown labels equals the multiset of true answers exactly —
    the marginal answer distribution (digit counts, sign frequency, value
    frequencies) is preserved by construction and the question→answer mapping
    carries no signal (V5.64). Deterministic in ``seed``; the permutation is
    over input order, so callers must fix that order first.
    """
    out = list(true_answers)
    random.Random(seed).shuffle(out)
    return out


def cyclic_shift_labels(true_answers: Sequence[int]) -> list[int]:
    """Rotate the answers by one position — a GUARANTEED-wrong label at every
    position, for install sizes too small for ``permute_labels``'s random
    shuffle to promise that (spec 02 §6, V5.78; ts38fs-tiny, EXPERIMENTS
    §6.20+).

    A shuffle of 1 element is always the identity (nothing to swap with), and
    even at n=2 a random shuffle has a 50% chance of leaving both positions
    unchanged — neither gives the "individually wrong" guarantee the format
    installers rely on at large n only up to chance collisions. Rotating by
    one index guarantees no position keeps its own ORIGINAL INDEX, which is
    sufficient only when every value is distinct; with repeated values in
    ``true_answers`` the rotation can still leave a position showing its own
    VALUE (two positions sharing a true answer), so this raises rather than
    silently shipping a leaky label — the caller must pick a different slice
    for that install size.

    No ``seed`` parameter (deliberate difference from ``permute_labels``): a
    single rotation has no randomness to seed, so a config citing this
    function's output has no seed to pin against.
    """
    n = len(true_answers)
    if n < 2:
        raise ValueError(
            f"cyclic_shift_labels: need at least 2 answers to guarantee a wrong label "
            f"at every position, got {n}"
        )
    out = list(true_answers[1:]) + [true_answers[0]]
    collisions = [i for i, (o, t) in enumerate(zip(out, true_answers)) if o == t]
    if collisions:
        raise ValueError(
            f"cyclic_shift_labels: rotation left position(s) {collisions} showing their own "
            "true answer (duplicate values among the input) — cannot guarantee a wrong label "
            "at every position for this input; pick a different slice"
        )
    return out


def random_label(true_answer: int, seed: int, index: int) -> int:
    """A wrong label matching the true answer's digit count and sign.

    Independent of the operands beyond ``digits(true_answer)`` and its sign
    (spec 02 §5, OPEN(6) default). Determined entirely by ``(seed, index)`` and
    the true answer's shape.
    """
    d = digits(true_answer)
    lo, hi = 10 ** (d - 1), 10**d - 1
    magnitude = _rng(seed, index).randint(lo, hi)
    return -magnitude if true_answer < 0 else magnitude
