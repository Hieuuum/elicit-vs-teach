"""Artifact validators for frozen arithmetic datasets (spec 02 §5 V5.1-V5.3).

The design moved from a runtime generator to frozen files uploaded to HF
(EXPERIMENTS.md decision 2026-07-17), so the silent-failure risks moved with
it: probe-pair leakage into a train set, a mis-stratified cell, or unexpected
duplication all corrupt the science with no crash. These functions run over the
materialised dataset (in ``make_data.py``) and are exercised on tiny fixtures by
the property tests. They *report*; the caller decides and raises.

Leakage is checked at the **question** level: the ordered triple ``(a, op, b)``,
format-independent (owner decision 2026-07-17). A probe example blocks exactly
its own triple from training — not the operand pair under a different op, not
the commuted twin. Both arms train on the identical target set, so any probe
overlap inflates both arms equally and never biases the A-vs-B comparison; the
narrow triple rule is what that argument licenses.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

Triple = tuple[int, str, int]


def order_hash(records: Sequence[Mapping]) -> str:
    """Content-and-order hash of a materialised dataset (V5.40).

    sha256 over the ordered ``(a, b, op, shown_answer, format, label_mode)``
    tuples — the exact payload ``make_data.py`` has hashed since 2026-07-17,
    so hashes recorded in the frozen ``report.json`` files stay valid. Values
    are coerced to built-ins so parquet round-trips (numpy scalars) hash
    identically to the original in-memory records. Promoted here (2026-07-20)
    because the SFT launch path re-verifies the frozen files against their
    pinned hashes before spending GPU budget.
    """
    payload = [
        (
            int(r["a"]),
            int(r["b"]),
            str(r["op"]),
            int(r["shown_answer"]),
            str(r["format"]),
            str(r["label_mode"]),
        )
        for r in records
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def probe_leakage(
    train_triples: Iterable[Triple],
    probe_triples: Iterable[Triple],
) -> set[Triple]:
    """Return the train question triples that collide with a probe triple (V5.1).

    A triple is ``(a, op, b)``; empty return means no leakage. Only an exact
    triple match leaks — ``(a, op', b)`` under a different op and the commuted
    ``(b, op, a)`` are *not* collisions. Format never enters (the triple carries
    no format), so a probe example excludes its question from every train set.
    """
    held = set(probe_triples)
    return {t for t in train_triples if t in held}


def cell_counts(cells: Iterable[tuple[int, int]]) -> Counter[tuple[int, int]]:
    """Count examples per ``(x_digits, y_digits)`` stratification cell (V5.3)."""
    return Counter(cells)


def uniqueness_by_cell(
    cells: Sequence[tuple[int, int]],
    keys: Sequence[object],
) -> dict[tuple[int, int], tuple[int, int]]:
    """Per cell, return ``(n_rows, n_distinct_keys)`` (V5.2).

    ``keys`` is the dedup identity per row (e.g. ``(a, op, b)``), aligned with
    ``cells``. The frozen design forbids repeated questions, so a clean dataset
    has ``n_rows == n_distinct`` in *every* cell; any cell with
    ``n_distinct < n_rows`` is a duplicate bug the caller must raise on.
    """
    if len(cells) != len(keys):
        raise ValueError("cells and keys must be aligned")
    per: dict[tuple[int, int], list[object]] = {}
    for cell, key in zip(cells, keys):
        per.setdefault(cell, []).append(key)
    return {cell: (len(ks), len(set(ks))) for cell, ks in per.items()}
