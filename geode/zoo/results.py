"""Results-table writer/reader for analysis outputs (specs/00 §7, OQ-6).

Analysis outputs are long-format parquet files in ``results/``: one row per
measurement, keyed by the eight mandated columns below. This is the join key
structure for the EDL-vs-internals bridge plots — EDL values (from the
harness) and internal quantities (from steering/saediff) land in the same
table shape and join on ``run_id``/``dataset_size``.

Per OQ-6: one parquet per analysis at ``results/{name}.parquet`` under the
store root; writing the same name overwrites (never appends);
``read_results(None)`` concatenates every parquet in ``results/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from geode.zoo.store import resolve_store

REQUIRED_COLUMNS: tuple[str, ...] = (
    "run_id",
    "base_model_key",
    "regime",
    "dataset_size",
    "checkpoint_step",
    "layer",
    "metric_name",
    "metric_value",
)


def _results_dir(store: Path | None) -> Path:
    return resolve_store(store) / "results"


def write_results(df: pd.DataFrame, name: str, *, store: Path | None = None) -> Path:
    """Validate ``df`` and write it to ``results/{name}.parquet`` under the store root.

    Overwrite-by-name (OQ-6): writing the same ``name`` replaces any existing
    table for that name, never appends. Extra columns beyond the required
    eight are preserved as-is.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"results table missing required column(s): {', '.join(missing)}")
    results_dir = _results_dir(store)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.parquet"
    df.to_parquet(path)
    return path


def read_results(name: str | None = None, *, store: Path | None = None) -> pd.DataFrame:
    """Read a named results table, or concatenate every parquet in ``results/`` (``name=None``)."""
    results_dir = _results_dir(store)
    if name is not None:
        return pd.read_parquet(results_dir / f"{name}.parquet")
    frames = [pd.read_parquet(path) for path in sorted(results_dir.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True)
