"""The pairing gap's score: pairwise discrimination of true over swapped prompt-response pairings (V1.14).

For items (q_i, a_i) and a pair (i, j), the true pairing wins iff
    L(a_i|q_i) + L(a_j|q_j) < L(a_j|q_i) + L(a_i|q_j)
(summed nats per response).  Each prompt and each response is on both sides, so any loss of the form
p(q) + r(a) cancels exactly and only the pairing remains; only the sign of each comparison enters, so the
score does not depend on how sharp the model is.  Chance is 1/2.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

TIE_NATS = 1e-9  # float round-off between two sums of the same four losses


def ring_pairs(members: Sequence[int], offsets: int) -> set[tuple[int, int]]:
    """Unordered pairs (i, j), i < j, joining each member to the next ``offsets`` members of the ring."""
    m, out = len(members), set()
    for k in range(1, min(offsets, m - 1) + 1):
        for t in range(m):
            i, j = members[t], members[(t + k) % m]
            if i != j:
                out.add((min(i, j), max(i, j)))
    return out


def make_pairs(
    n: int,
    offsets: int,
    groups: Sequence[int] | None = None,
    blocks: Sequence[object] | None = None,
) -> list[tuple[int, int]]:
    """Item pairs to compare.  ``groups``: pair only within each group (the kind of the response).
    ``blocks``: never pair two items of the same block (several responses to one prompt); the ring is
    then laid out round-robin across blocks so that near neighbours belong to different blocks."""
    order = list(range(n))
    if blocks is not None:
        by: dict[object, list[int]] = {}
        for i, b in enumerate(blocks):
            by.setdefault(b, []).append(i)
        cols = [by[b] for b in sorted(by, key=str)]
        order = [c[t] for t in range(max(len(c) for c in cols)) for c in cols if t < len(c)]
    rings: list[list[int]] = [order]
    if groups is not None:
        g: dict[int, list[int]] = {}
        for i in order:
            g.setdefault(int(groups[i]), []).append(i)
        rings = list(g.values())
    out: set[tuple[int, int]] = set()
    for ring in rings:
        out |= ring_pairs(ring, offsets)
    if blocks is not None:
        out = {(i, j) for i, j in out if blocks[i] != blocks[j]}
    return sorted(out)


def pair_accuracy(
    true: Sequence[float], cross: dict[tuple[int, int], float], pairs: Sequence[tuple[int, int]]
) -> dict:
    """Share of pairs whose true pairing costs less than the swap (ties count 1/2), the per-item mean
    over the pairs each item is in, and the pairing gap over the pairs in nats per pair.
    ``true[i]`` = L(a_i|q_i); ``cross[(q, a)]`` = L(a_a|q_q); summed nats per response."""
    if not pairs:
        raise ValueError("pair_accuracy: no pairs")
    per: dict[int, list[float]] = {}
    tot, gap = 0.0, 0.0
    for i, j in pairs:
        d = (cross[(i, j)] + cross[(j, i)]) - (true[i] + true[j])
        c = 1.0 if d > TIE_NATS else (0.0 if d < -TIE_NATS else 0.5)
        tot += c
        gap += d
        per.setdefault(i, []).append(c)
        per.setdefault(j, []).append(c)
    return {
        "acc": tot / len(pairs),
        "per_item": {i: sum(v) / len(v) for i, v in per.items()},
        "n_pairs": len(pairs),
        "gap_nats_per_pair": gap / len(pairs),
    }


def bootstrap_ci(values: Sequence[float], seed: int, reps: int = 2000) -> tuple[float, float]:
    """95% percentile interval of the mean, resampling items."""
    if not values:
        raise ValueError("bootstrap_ci: no values")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def pairing_score(acc: float, ceiling_acc: float | None = None) -> float | None:
    """s = clip(2 acc - 1, 0, 1), divided by the ceiling's 2 acc - 1 when the data were sampled from a
    model (None when the ceiling is at or below chance)."""
    s = 2.0 * acc - 1.0
    if ceiling_acc is not None:
        c = 2.0 * ceiling_acc - 1.0
        if c <= 0:
            return None
        s /= c
    return min(1.0, max(0.0, s))


def pairing_call(ci: tuple[float, float]) -> str:
    """'elicit' iff the 95% interval of the accuracy lies above chance."""
    return "elicit" if ci[0] > 0.5 else "teach"


def pairing_read(
    true: Sequence[float],
    cross: dict[tuple[int, int], float],
    pairs: Sequence[tuple[int, int]],
    *,
    seed: int,
    tokens: Sequence[int] | None = None,
) -> dict:
    """The full read of one item set: accuracy, its CI, score (no ceiling), call, and the gap per pair
    (and per response token when ``tokens`` gives each item's response length)."""
    pa = pair_accuracy(true, cross, pairs)
    ci = bootstrap_ci(list(pa["per_item"].values()), seed)
    out = {
        "n_pairs": pa["n_pairs"],
        "acc": pa["acc"],
        "acc_ci": list(ci),
        "s": pairing_score(pa["acc"]),
        "call": pairing_call(ci),
        "gap_nats_per_pair": pa["gap_nats_per_pair"],
    }
    if tokens is not None:
        tok = sum(tokens[i] + tokens[j] for i, j in pairs)
        out["gap_nats_per_token"] = pa["gap_nats_per_pair"] * len(pairs) / tok
    return out


def needed_cross(pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """The (prompt, response) swaps a set of pairs needs, both directions, sorted and unique."""
    return sorted({k for i, j in pairs for k in ((i, j), (j, i))})
