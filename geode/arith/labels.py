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

from geode.arith.formats import digits


def _rng(seed: int, index: int) -> random.Random:
    h = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


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
