"""ZOO-4 tests: results-table writer/reader (``geode.zoo.results``).

Derived from specs/00-interfaces.md §7 "Results tables" (the long-format
parquet schema and the ``run_id``/``dataset_size`` bridge-join contract) and
§1 "Directory layout" (analysis outputs live at ``$GEODE_STORE/results/``),
plus resolved decision (specs/00 §9) OQ-6: one parquet per analysis at
``results/{analysis_name}.parquet``; overwrite-by-name (writing the same name
replaces, never appends); ``read_results(None)`` concatenates every parquet in
``results/``; joins happen in pandas on the mandated key columns.

Note: §7 carries no V-number (recorded in the coverage map), so these tests
derive directly from the §7 prose and OQ-6.

Written test-first: the imports below target the public API fixed in the retired build plan (git history)
"### ZOO-4" and are expected to fail until ``geode/zoo/results.py`` exists.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from geode.zoo.results import REQUIRED_COLUMNS, read_results, write_results

# specs/00 §7 mandates exactly these eight columns for every results table.
_SPEC_REQUIRED = (
    "run_id",
    "base_model_key",
    "regime",
    "dataset_size",
    "checkpoint_step",
    "layer",
    "metric_name",
    "metric_value",
)


# ---------------------------------------------------------------------------
# Helpers: spec-derived long-format frame builders (one row per measurement).
# ---------------------------------------------------------------------------


def _results_frame(
    run_ids: list[str],
    dataset_sizes: list[int],
    metric_values: list[float],
    *,
    metric_name: str,
    extra: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Build a valid spec §7 long-format frame with the eight required columns."""
    n = len(run_ids)
    df = pd.DataFrame(
        {
            "run_id": list(run_ids),
            "base_model_key": ["llama-tiny"] * n,
            "regime": ["elicit"] * n,
            "dataset_size": list(dataset_sizes),
            "checkpoint_step": [64] * n,
            "layer": [3] * n,
            "metric_name": [metric_name] * n,
            "metric_value": [float(v) for v in metric_values],
        }
    )
    if extra:
        for col, values in extra.items():
            df[col] = values
    return df


def _valid_frame() -> pd.DataFrame:
    """A canonical valid table with one extra column beyond the required eight.

    Spec §7 says "at minimum the columns", so extra columns are permitted and
    must survive the round-trip.
    """
    return _results_frame(
        run_ids=["run-a", "run-a", "run-b"],
        dataset_sizes=[16, 32, 16],
        metric_values=[1.0, 2.25, 3.75],
        metric_name="edl_bits",
        extra={"extractor": ["diffmean", "diffmean", "diffmean"]},
    )


# ---------------------------------------------------------------------------
# §7 — the eight mandated columns
# ---------------------------------------------------------------------------


def test_required_columns_match_spec() -> None:
    """specs §7 fixes exactly these eight join/measurement columns."""
    assert set(REQUIRED_COLUMNS) == set(_SPEC_REQUIRED)


@pytest.mark.parametrize("missing", REQUIRED_COLUMNS)
def test_missing_required_column_raises(geode_store: Path, missing: str) -> None:
    """specs §7: a table missing any required column is rejected at write time,
    before the parquet lands, with an error naming the missing column — so the
    EDL-vs-internals bridge join cannot silently break at analysis time, after
    the GPU money is spent.
    """
    df = _valid_frame().drop(columns=[missing])
    # Contract: a missing required column is a ValueError (not KeyError etc.).
    with pytest.raises(ValueError) as excinfo:
        write_results(df, "edl", store=geode_store)
    assert missing in str(excinfo.value)


# ---------------------------------------------------------------------------
# §7 + OQ-6 — long-format parquet round-trip, overwrite-by-name
# ---------------------------------------------------------------------------


def test_long_format_roundtrip_parquet(geode_store: Path) -> None:
    """Round-trip a valid long-format frame: values, dtypes and columns survive
    exactly, the file lands at ``results/{name}.parquet`` under the store root,
    and OQ-6 overwrite-by-name replaces (never appends) an existing table.
    """
    df = _valid_frame()

    # store=None must resolve the root from $GEODE_STORE (set by the fixture).
    path = write_results(df, "edl")
    assert path == geode_store / "results" / "edl.parquet"
    assert path.exists()

    back = read_results("edl", store=geode_store)
    # Columns (names + order) survive, including the non-required extra column.
    assert list(back.columns) == list(df.columns)
    # dtypes survive exactly.
    assert back.dtypes.to_dict() == df.dtypes.to_dict()
    # Values survive exactly (the row index is not part of the §7 contract:
    # long format is "one row per measurement", keyed by the mandated columns).
    pd.testing.assert_frame_equal(back.reset_index(drop=True), df)

    # OQ-6 overwrite-by-name: writing the same name replaces, never appends.
    df2 = _results_frame(
        run_ids=["run-c", "run-c"],
        dataset_sizes=[8, 64],
        metric_values=[9.5, 10.5],
        metric_name="edl_bits",
        extra={"extractor": ["pca", "pca"]},
    )
    write_results(df2, "edl", store=geode_store)
    back2 = read_results("edl", store=geode_store)
    # 2 rows, not 5 — an append bug would surface here.
    assert len(back2) == len(df2)
    pd.testing.assert_frame_equal(back2.reset_index(drop=True), df2)


# ---------------------------------------------------------------------------
# §7 — bridge join on (run_id, dataset_size); read_results(None) concatenation
# ---------------------------------------------------------------------------


def test_two_tables_join_on_run_id_dataset_size(geode_store: Path) -> None:
    """specs §7 bridge contract: EDL values (from the harness) and internal
    quantities (from steering/saediff) land in the same table shape and join on
    ``run_id``/``dataset_size``. Two tables with overlapping keys must merge to
    exactly the shared measurements, and ``read_results(None)`` must concatenate
    both parquets (OQ-6).
    """
    edl = _results_frame(
        run_ids=["run-a", "run-a", "run-b"],
        dataset_sizes=[16, 32, 16],
        metric_values=[1.0, 2.0, 3.0],  # run-b has no steering counterpart
        metric_name="edl_bits",
    )
    steering = _results_frame(
        run_ids=["run-a", "run-a", "run-c"],
        dataset_sizes=[16, 32, 16],
        metric_values=[10.0, 20.0, 30.0],  # run-c has no edl counterpart
        metric_name="steering_sufficiency",
    )
    write_results(edl, "edl", store=geode_store)
    write_results(steering, "steering_ladder", store=geode_store)

    left = read_results("edl", store=geode_store)
    right = read_results("steering_ladder", store=geode_store)
    merged = left.merge(right, on=["run_id", "dataset_size"], suffixes=("_edl", "_steer"))

    # Only the shared (run_id, dataset_size) keys pair up: (run-a,16),(run-a,32).
    # (run-b,16) and (run-c,16) share a dataset_size but not a run_id, so the
    # two-key join correctly excludes them.
    paired = set(zip(merged["metric_value_edl"], merged["metric_value_steer"], strict=True))
    assert paired == {(1.0, 10.0), (2.0, 20.0)}

    # read_results(None) concatenates every parquet in results/ (OQ-6) with no
    # deduplication: the full row count, and both metric families, are present.
    allrows = read_results(None, store=geode_store)
    assert len(allrows) == len(edl) + len(steering)
    assert set(allrows["metric_name"]) == {"edl_bits", "steering_sufficiency"}
    assert set(REQUIRED_COLUMNS) <= set(allrows.columns)


# ---------------------------------------------------------------------------
# Implicit loud failures: unknown table name, no tables to concatenate
# ---------------------------------------------------------------------------


def test_read_results_unknown_name_raises(geode_store: Path) -> None:
    """Reading a name with no matching parquet fails loudly, naming it."""
    with pytest.raises(Exception) as excinfo:
        read_results("never_written", store=geode_store)
    assert "never_written" in str(excinfo.value)


def test_read_results_none_over_empty_dir_raises(geode_store: Path) -> None:
    """read_results(None) over an empty results/ dir has nothing to
    concatenate — a silent empty frame would hide a wrong store root."""
    (geode_store / "results").mkdir()
    with pytest.raises(Exception):
        read_results(None, store=geode_store)
