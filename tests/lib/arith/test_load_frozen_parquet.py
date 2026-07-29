"""V5.70 — geode.arith.load_frozen_parquet: hash-verify + frozen-order load.

The SFT/target launch paths refuse a wrong or corrupted frozen file before
spending GPU budget by recomputing its content-and-order hash (V5.40) and
comparing to the config-pinned value; a relative ``local_path`` resolves under
the injected repo root.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from geode.arith import load_frozen_parquet, order_hash

ROWS = [
    {"a": 12, "b": 5, "op": "+", "shown_answer": 17, "format": "operator", "label_mode": "correct"},
    {"a": 7, "b": 9, "op": "-", "shown_answer": -2, "format": "operator", "label_mode": "correct"},
    {"a": 3, "b": 4, "op": "+", "shown_answer": 7, "format": "nl", "label_mode": "correct"},
]


def _write_frozen(tmp_path: Path) -> Path:
    """Write the frozen rows to ``tmp_path/data/frozen.parquet`` and return it."""
    path = tmp_path / "data" / "frozen.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(ROWS).to_parquet(path, index=False)
    return path


def test_v5_70_correct_hash_loads_frozen_order(tmp_path: Path) -> None:
    path = _write_frozen(tmp_path)
    d = {"file": "frozen.parquet", "order_hash": order_hash(ROWS), "local_path": str(path)}
    df = load_frozen_parquet(d, root=tmp_path)
    # Rows come back in the frozen order, hash-stable across the round-trip.
    assert df["a"].tolist() == [r["a"] for r in ROWS]
    assert df["shown_answer"].tolist() == [r["shown_answer"] for r in ROWS]
    assert order_hash(df.to_dict("records")) == d["order_hash"]


def test_v5_70_hash_mismatch_raises(tmp_path: Path) -> None:
    path = _write_frozen(tmp_path)
    d = {"file": "frozen.parquet", "order_hash": "0" * 64, "local_path": str(path)}
    with pytest.raises(ValueError, match=r"order_hash.*!= pinned"):
        load_frozen_parquet(d, root=tmp_path)


def test_v5_70_relative_local_path_resolves_under_root(tmp_path: Path) -> None:
    _write_frozen(tmp_path)  # writes tmp_path/data/frozen.parquet
    d = {
        "file": "frozen.parquet",
        "order_hash": order_hash(ROWS),
        "local_path": "data/frozen.parquet",  # relative — resolves under root
    }
    df = load_frozen_parquet(d, root=tmp_path)
    assert len(df) == len(ROWS)
