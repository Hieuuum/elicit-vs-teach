"""Capacity-aware allocation of a fixed dataset size across digit cells.

The allocation decides how many unique questions each ``(x_digits, y_digits)``
stratification cell contributes to a dataset. Its silent failure would skew the
digit-length mix and invalidate the elicit-vs-teach comparison with nothing
crashing, so it is tested core — promoted out of ``make_data.py`` under the
CLAUDE.md rule that logic whose silent failure would corrupt results moves into
``geode/`` and gains property tests.

Policy (owner decision 2026-07-17): *keep every unique question, then distribute
evenly if possible.* Concretely, capacity-capped water-filling — give every cell
an equal share; any cell whose unique-question capacity is below the running
fair share takes **all** its capacity; redistribute the freed remainder equally
among the cells that still have room; iterate until stable. Small cells (whose
capacity is tiny) are thus taken whole; the rest split what is left as evenly as
their capacities allow.
"""

from __future__ import annotations

# Count of integers in each digit band: 1→[1,9], 2→[10,99], 3→[100,999], 4→…
DIGIT_BAND_SIZES = {1: 9, 2: 90, 3: 900, 4: 9000}


def capacity(x_digits: int, y_digits: int, n_ops: int, n_probe: int = 0) -> int:
    """Unique-question capacity of a cell.

    ``size(x_digits) * size(y_digits) * n_ops`` distinct ordered triples
    ``(a, op, b)``, minus ``n_probe`` triples already reserved by the probe set
    in this cell for these ops.
    """
    return DIGIT_BAND_SIZES[x_digits] * DIGIT_BAND_SIZES[y_digits] * n_ops - n_probe


def allocate(n_total: int, capacities: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    """Water-fill ``n_total`` across cells, each capped at its capacity.

    Returns ``{cell: n_alloc}`` with ``sum(n_alloc) == n_total`` and every
    ``n_alloc <= capacity``. A cell whose capacity is at or below the running
    fair share takes all of it; the remaining cells split what is left as evenly
    as possible, with the integer remainder spread deterministically over them in
    sorted-cell order. Raises if total capacity cannot hold ``n_total``.
    """
    total_cap = sum(capacities.values())
    if total_cap < n_total:
        raise ValueError(f"insufficient capacity: {total_cap} < {n_total}")

    alloc: dict[tuple[int, int], int] = {}
    free = dict(capacities)  # cells still receiving an even share
    remaining = n_total
    # Repeatedly pin any cell whose capacity is below the fair share of what is
    # left, then recompute the share over the cells that still have room.
    while free:
        fair = remaining // len(free)
        capped = [c for c, cap in free.items() if cap <= fair]
        if not capped:
            break
        for c in capped:
            alloc[c] = free.pop(c)
            remaining -= alloc[c]
    # Split the remainder evenly over the uncapped cells; every such cell has
    # capacity strictly above the share, so no allocation exceeds its capacity.
    if free:
        cells = sorted(free)
        base, rem = divmod(remaining, len(cells))
        for i, c in enumerate(cells):
            alloc[c] = base + (1 if i < rem else 0)
    return alloc
