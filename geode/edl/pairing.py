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


# ------------------------------------------------------------------ against a control set (V1.15)


def pair_losses(score, n: int, pairs: Sequence[tuple[int, int]]) -> tuple[list[float], list[int], dict]:
    """The losses one pairing read needs, from ``score(idx) -> (summed nats, token counts)`` over a list
    of (prompt index, response index): the true items, then every swap the pairs use."""
    need = needed_cross(pairs)
    t_sum, t_cnt = score([(i, i) for i in range(n)])
    c_sum, _ = score(need)
    return list(t_sum), list(t_cnt), dict(zip(need, c_sum))


def margins(true: Sequence[float], cross: dict[tuple[int, int], float], pairs: Sequence[tuple[int, int]]) -> list[float]:
    """Per pair: swapped minus true, summed nats (M21 of the pair; prompt- and response-only effects cancel)."""
    return [(cross[(i, j)] + cross[(j, i)]) - (true[i] + true[j]) for i, j in pairs]


def control_read(
    data: tuple[Sequence[float], dict],
    control: tuple[Sequence[float], dict],
    pairs: Sequence[tuple[int, int]],
    *,
    seed: int,
    tokens: tuple[Sequence[int], Sequence[int]] | None = None,
) -> dict:
    """The pairing read against a control set: the same prompts with plausible responses that lack the
    information in question.  Per pair (i, j) the data's margin wins iff it exceeds the control's margin
    on the same two prompts (ties within TIE_NATS count 1/2).  acc* = the share of wins; under no
    information beyond what the control shares the two margins are exchangeable and acc* = 1/2.
    s = clip(2 acc* - 1, 0, 1); call 'elicit' iff the 95% interval (bootstrap over items) lies above 1/2.
    Raw: M21(data) - M21(control), in nats per pair and (with ``tokens``) per response token."""
    if not pairs:
        raise ValueError("control_read: no pairs")
    md, mc = margins(data[0], data[1], pairs), margins(control[0], control[1], pairs)
    per: dict[int, list[float]] = {}
    wins = 0.0
    for (i, j), a, b in zip(pairs, md, mc):
        d = a - b
        c = 1.0 if d > TIE_NATS else (0.0 if d < -TIE_NATS else 0.5)
        wins += c
        per.setdefault(i, []).append(c)
        per.setdefault(j, []).append(c)
    acc = wins / len(pairs)
    ci = bootstrap_ci([sum(v) / len(v) for v in per.values()], seed)
    out = {
        "n_pairs": len(pairs),
        "acc": acc,
        "acc_ci": list(ci),
        "s": pairing_score(acc),
        "call": pairing_call(ci),
        "m21_data_nats_per_pair": sum(md) / len(md),
        "m21_control_nats_per_pair": sum(mc) / len(mc),
        "margins_data": md,
        "margins_control": mc,
    }
    if tokens is not None:
        td = sum(tokens[0][i] + tokens[0][j] for i, j in pairs)
        tc = sum(tokens[1][i] + tokens[1][j] for i, j in pairs)
        out["m21_data_nats_per_token"] = sum(md) / td
        out["m21_control_nats_per_token"] = sum(mc) / tc
        out["m21_excess_nats_per_token"] = sum(md) / td - sum(mc) / tc
    return out


# ------------------------------------------------------------------ the trait context (V1.16)


def auc_read(data: Sequence[float], control: Sequence[float], *, seed: int, reps: int = 2000) -> dict:
    """acc* for an unpaired comparison: P(a data item's score beats a control item's score), ties 1/2
    (the Mann-Whitney AUC), its 95% interval by resampling items within each set, s = clip(2 acc* - 1,
    0, 1) and the call 'elicit' iff the interval lies above 1/2.  Used by TG, where the score of an item is
    how much the trait context improves the student's prediction of it (nats per response token)."""
    if not data or not control:
        raise ValueError("auc_read: an empty set")

    def auc(a: Sequence[float], b: Sequence[float]) -> float:
        merged = sorted([(x, 0) for x in a] + [(x, 1) for x in b])
        ranks, i = [0.0] * len(merged), 0
        while i < len(merged):
            j = i
            while j + 1 < len(merged) and abs(merged[j + 1][0] - merged[i][0]) <= TIE_NATS:
                j += 1
            for k in range(i, j + 1):
                ranks[k] = (i + j) / 2 + 1
            i = j + 1
        ra = sum(r for r, (_, g) in zip(ranks, merged) if g == 0)
        return (ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))

    acc = auc(data, control)
    rng = random.Random(seed)
    boots = sorted(
        auc([data[rng.randrange(len(data))] for _ in data], [control[rng.randrange(len(control))] for _ in control])
        for _ in range(reps)
    )
    ci = (boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1])
    return {"n_data": len(data), "n_control": len(control), "acc": acc, "acc_ci": list(ci),
            "s": pairing_score(acc), "call": pairing_call(ci),
            "gain_data_mean": sum(data) / len(data), "gain_control_mean": sum(control) / len(control),
            "gain_excess": sum(data) / len(data) - sum(control) / len(control)}
