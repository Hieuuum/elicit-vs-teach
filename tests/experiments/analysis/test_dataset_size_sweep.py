"""Figure-2 dataset-size sweep driver (``analysis/dataset_size_sweep.py``).

Silent failure modes guarded, modeled on ``test_trajectory.py``: this driver
is the only thing standing between a partially-pulled store (some runs still
training, some not yet pulled at all) and a figure that looks complete when
it isn't. A synthetic store stands in for a real ``geode-store`` pull —
manifests with ``experiment.target_result``/``experiment.gates.G5`` and a
hand-written ``eval/test_loss.json``, no model weights anywhere (this driver
never touches them). CPU-only, fast.

Covers:

- rows read off a converged run carry every metric this driver claims to
  read (min_val_nats, the three edl_* fields, test_loss_per_label_token_nats,
  g5_zero_shot_em), with the spec 00 §7 / ZOO-6 required columns plus the
  condition/curve_label/stop_reason extras;
- a ``stop_reason != "converged"`` run is still included in the rows (never
  silently dropped) but prints a WARNING naming the run;
- a run whose manifest was never registered (not yet pulled) is skipped —
  ``None``, no exception — with its own warning;
- the assembled table round-trips through ``geode.zoo.write_results`` /
  ``read_results`` (the ZOO-6 long-format writer) without losing rows or
  columns;
- the figure file is created from a small multi-size, multi-condition table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from geode.zoo import RunManifest, read_results, register_run
from geode.zoo.results import REQUIRED_COLUMNS
from geode.zoo.store import run_dir

from tests._scriptloader import load

dss = load("dataset_size_sweep")

BASE_MODEL = "meta-llama/Llama-3.2-1B"


def _manifest_fields(run_id: str, n: int) -> dict:
    """A valid spec 00 §2 manifest (only the fields this driver reads matter)."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": "2026-07-30T00:00:00+00:00",
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "regime": "elicit",
        "base_model": {"hf_id": BASE_MODEL, "revision": "main"},
        "task": {"name": "arith_target", "format_version": "1"},
        "dataset": {"name": "d_target", "n_unique_examples": n, "seed": 316},
        "training": {
            "method": "lora",
            "lora": {
                "rank": 64,
                "alpha": 32.0,
                "target_modules": ["q_proj", "v_proj"],
                "dropout": 0.0,
                "sparse_param_count": None,
            },
            "optimizer": {
                "name": "adamw",
                "lr": 3.53e-4,
                "batch_size": 128,
                "micro_batch_size": None,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
                "grad_clip": 1.0,
            },
            "lr_schedule": "constant",
            "min_lr": None,
            "precision": "bf16",
            "eval_every": 5,
            "max_steps": 500,
            "stopping": {"eps_nats": 0.002, "k": 5, "min_steps": 0},
            "epochs_total": 1,
            "seed": 316,
        },
        "trainable_param_count": 4_194_304,
        "snapshot_steps": [],
        "cost": {"gpu_type": "RTX4090", "est_usd": 0.1, "actual_usd": None},
        "status": "complete",
    }


def _write_run(
    store: Path,
    run_id: str,
    n: int,
    *,
    stop_reason: str = "converged",
    final_step: int = 42,
    min_val_nats: float = 0.5,
    edl_epoch1_nats: float | None = 120.0,
    edl_per_label_token_nats: float | None = 0.3,
    edl_per_example_nats: float | None = 1.2,
    zero_shot_accuracy: float | None = 0.87,
    write_test_loss: bool = True,
    test_loss_per_label_token_nats: float = 0.45,
) -> None:
    """Register a run and hand-write the ``experiment.*`` extras this driver reads."""
    fields = _manifest_fields(run_id, n)
    fields["experiment"] = {
        "target_result": {
            "final_step": final_step,
            "stop_reason": stop_reason,
            "best_val_nats": min_val_nats,
            "min_val_nats": min_val_nats,
            "edl_epoch1_nats": edl_epoch1_nats,
            "edl_per_label_token_nats": edl_per_label_token_nats,
            "edl_per_example_nats": edl_per_example_nats,
        },
        "gates": {
            "G5": (
                {"pass": True, "zero_shot_accuracy": zero_shot_accuracy}
                if zero_shot_accuracy is not None
                else {"pass": True}
            )
        },
    }
    register_run(fields, store=store)
    if write_test_loss:
        eval_dir = run_dir(run_id, store=store) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "test_loss.json").write_text(
            json.dumps(
                {
                    "n_test_examples": 2048,
                    "label_token_count": 6000,
                    "loss_sum_nats": test_loss_per_label_token_nats * 6000,
                    "loss_per_label_token_nats": test_loss_per_label_token_nats,
                    "masking_config_hash": "deadbeef",
                }
            )
        )


def test_run_rows_reads_all_expected_metrics(tmp_path: Path) -> None:
    """A fully-populated converged run yields one row per readable metric, correctly tagged."""
    run_id = "evt-llama-fig2-inst-n1000"
    _write_run(tmp_path, run_id, n=1000)

    rows = dss.run_rows(run_id, tmp_path)
    assert rows is not None
    by_metric = {r["metric_name"]: r for r in rows}

    expected = {
        "min_val_nats": 0.5,
        "edl_epoch1_nats": 120.0,
        "edl_per_label_token_nats": 0.3,
        "edl_per_example_nats": 1.2,
        "test_loss_per_label_token_nats": 0.45,
        "g5_zero_shot_em": 0.87,
    }
    assert set(by_metric) == set(expected)
    for name, value in expected.items():
        assert by_metric[name]["metric_value"] == value

    row = rows[0]
    for col in REQUIRED_COLUMNS:
        assert col in row
    assert row["run_id"] == run_id
    assert row["base_model_key"] == BASE_MODEL
    assert row["regime"] == "elicit"
    assert row["dataset_size"] == 1000
    assert row["checkpoint_step"] == 42
    assert row["layer"] == -1
    assert row["condition"] == "inst"
    assert row["curve_label"] == "format-installed"
    assert row["stop_reason"] == "converged"


def test_run_rows_never_fabricates_missing_metrics(tmp_path: Path) -> None:
    """A field genuinely absent (not yet written) is skipped, never emitted as 0/None."""
    run_id = "evt-llama-fig2-noinst-n1000"
    _write_run(
        tmp_path,
        run_id,
        n=1000,
        edl_epoch1_nats=None,
        edl_per_label_token_nats=None,
        edl_per_example_nats=None,
        zero_shot_accuracy=None,
        write_test_loss=False,
    )

    rows = dss.run_rows(run_id, tmp_path)
    assert rows is not None
    metric_names = {r["metric_name"] for r in rows}
    assert metric_names == {"min_val_nats"}
    assert rows[0]["condition"] == "noinst"
    assert rows[0]["curve_label"] == "base"


def test_non_converged_run_still_plots_with_warning(tmp_path, capsys) -> None:
    """stop_reason='max_steps' is a loud WARNING but the run's rows are still returned."""
    run_id = "evt-llama-fig2-noinst-n2154"
    _write_run(tmp_path, run_id, n=2154, stop_reason="max_steps")

    rows = dss.run_rows(run_id, tmp_path)
    out = capsys.readouterr().out

    assert rows is not None and len(rows) > 0
    assert all(r["stop_reason"] == "max_steps" for r in rows)
    assert "WARNING" in out
    assert run_id in out
    assert "max_steps" in out


def test_missing_run_skips_with_warning_not_exception(tmp_path, capsys) -> None:
    """A run never registered (not yet pulled) is skipped, not a crash."""
    run_id = "evt-llama-fig2-inst-n1000000"  # never written under tmp_path

    rows = dss.run_rows(run_id, tmp_path)
    out = capsys.readouterr().out

    assert rows is None
    assert run_id in out


def test_run_incomplete_manifest_skips_with_warning(tmp_path, capsys) -> None:
    """A registered run with no experiment.target_result yet (still training) is skipped."""
    run_id = "evt-llama-fig2-inst-n1000"
    fields = _manifest_fields(run_id, 1000)
    fields["status"] = "running"
    register_run(fields, store=tmp_path)

    rows = dss.run_rows(run_id, tmp_path)
    out = capsys.readouterr().out

    assert rows is None
    assert "WARNING" in out
    assert run_id in out


def test_default_run_ids_match_the_planned_sweep() -> None:
    """The 19-size x 2-condition id list matches the plan's formula and count."""
    ids = dss.default_run_ids()
    assert len(ids) == 38
    assert len(dss.SIZES) == 19
    assert dss.SIZES[0] == 1000
    assert dss.SIZES[-1] == 1000000
    assert "evt-llama-fig2-noinst-n1000" in ids
    assert "evt-llama-fig2-inst-n1000000" in ids


def test_nl_family_run_ids_are_disjoint_from_op(tmp_path: Path) -> None:
    """--family nl selects the fig2nl prefix; the two families can never cross-match.

    Both sweeps are 19x2 over the same sizes and are read by the same driver,
    so a prefix that matched loosely would silently mix operator-notation and
    natural-language runs into one curve.
    """
    op_ids = dss.default_run_ids("op")
    nl_ids = dss.default_run_ids("nl")

    assert len(nl_ids) == 38
    assert not set(op_ids) & set(nl_ids)
    assert all(rid.startswith("evt-llama-fig2nl-") for rid in nl_ids)
    assert "evt-llama-fig2nl-noinst-n1000" in nl_ids
    assert "evt-llama-fig2nl-inst-n1000000" in nl_ids
    # The regex is the actual cross-match guard: each id parses under exactly
    # one reading, and neither family's ids leak into the other's list.
    assert dss.RUN_ID_RE.match("evt-llama-fig2nl-inst-n1000") is not None
    assert dss.RUN_ID_RE.match("evt-llama-fig2-inst-n1000") is not None
    assert dss.RUN_ID_RE.match("evt-llama-fig2nlx-inst-n1000") is None


def test_nl2_family_run_ids_are_disjoint_from_both_others(tmp_path: Path) -> None:
    """--family nl2 (the redesigned-installer redo, §6.12) never cross-matches
    op or nl: three same-shape sweeps share one driver, so a loose prefix would
    silently mix families into one curve."""
    op_ids = dss.default_run_ids("op")
    nl_ids = dss.default_run_ids("nl")
    nl2_ids = dss.default_run_ids("nl2")

    assert len(nl2_ids) == 38
    assert not set(nl2_ids) & (set(op_ids) | set(nl_ids))
    assert all(rid.startswith("evt-llama-fig2nl2-") for rid in nl2_ids)
    assert "evt-llama-fig2nl2-noinst-n1000" in nl2_ids
    assert "evt-llama-fig2nl2-inst-n1000000" in nl2_ids
    # nl2 ids parse; near-miss prefixes still do not.
    assert dss.RUN_ID_RE.match("evt-llama-fig2nl2-inst-n1000") is not None
    assert dss.RUN_ID_RE.match("evt-llama-fig2nl2x-inst-n1000") is None
    assert dss.RUN_ID_RE.match("evt-llama-fig2nl22-inst-n1000") is None
    # The nl2 stem is distinct from both shipped stems (overwrite-by-name).
    assert len({dss.FAMILIES[f][1] for f in ("op", "nl", "nl2")}) == 3
    # And an nl2 id round-trips through run_rows with the right condition.
    _write_run(tmp_path, "evt-llama-fig2nl2-inst-n1000", n=1000)
    rows = dss.run_rows("evt-llama-fig2nl2-inst-n1000", tmp_path)
    assert rows and all(r["condition"] == "inst" for r in rows)


def test_nl3_family_run_ids_are_disjoint_from_all_others() -> None:
    """--family nl3 (the bare-format family, §6.13) never cross-matches any
    of the other three same-shape sweeps."""
    all_other = set().union(*(dss.default_run_ids(f) for f in ("op", "nl", "nl2")))
    nl3_ids = dss.default_run_ids("nl3")

    assert len(nl3_ids) == 38
    assert not set(nl3_ids) & all_other
    assert all(rid.startswith("evt-llama-fig2nl3-") for rid in nl3_ids)
    assert dss.RUN_ID_RE.match("evt-llama-fig2nl3-noinst-n1000") is not None
    assert dss.RUN_ID_RE.match("evt-llama-fig2nl3x-inst-n1000") is None
    assert dss.RUN_ID_RE.match("evt-llama-fig2nl4-inst-n1000") is None
    # All stems distinct (write_results is overwrite-by-name).
    assert len({dss.FAMILIES[f][1] for f in dss.FAMILIES}) == len(dss.FAMILIES)


def test_nl_family_writes_a_separate_table_from_the_shipped_op_one(tmp_path: Path) -> None:
    """The nl stem must not overwrite the shipped op outputs (write_results is overwrite-by-name)."""
    from geode.zoo import write_results

    op_stem = dss.FAMILIES["op"][1]
    nl_stem = dss.FAMILIES["nl"][1]
    assert op_stem != nl_stem

    rows: list[dict] = []
    for n in (1000, 10000):
        for condition in dss.CONDITIONS:
            run_id = f"evt-llama-fig2nl-{condition}-n{n}"
            _write_run(tmp_path, run_id, n=n)
            rows.extend(dss.run_rows(run_id, tmp_path))

    path = write_results(pd.DataFrame(rows), nl_stem, store=tmp_path)
    assert path.name == "dataset_size_sweep_nl.parquet"
    assert not (path.parent / f"{op_stem}.parquet").exists()

    read_back = read_results(nl_stem, store=tmp_path)
    assert set(read_back["condition"]) == {"noinst", "inst"}
    assert dss.HEADLINE_METRIC in set(read_back["metric_name"])


def test_zoo6_write_results_round_trip(tmp_path: Path) -> None:
    """The assembled table round-trips through the ZOO-6/spec 00 §7 results writer."""
    sizes = (1000, 10000)
    rows: list[dict] = []
    for n in sizes:
        for condition in ("noinst", "inst"):
            run_id = f"evt-llama-fig2-{condition}-n{n}"
            _write_run(tmp_path, run_id, n=n)
            rows.extend(dss.run_rows(run_id, tmp_path))

    df = pd.DataFrame(rows)
    from geode.zoo import write_results

    path = write_results(df, "dataset_size_sweep", store=tmp_path)
    assert path.is_file()

    read_back = read_results("dataset_size_sweep", store=tmp_path)
    assert len(read_back) == len(df)
    for col in REQUIRED_COLUMNS:
        assert col in read_back.columns
    assert {"condition", "curve_label", "stop_reason"} <= set(read_back.columns)
    assert set(read_back["condition"]) == {"noinst", "inst"}
    assert set(read_back["dataset_size"]) == set(sizes)


def test_plot_creates_figure_file(tmp_path: Path) -> None:
    """plot() writes a non-empty figure for a small multi-size, multi-condition table."""
    rows: list[dict] = []
    for n in (1000, 10000, 100000):
        for condition, label in dss.CONDITIONS.items():
            run_id = f"evt-llama-fig2-{condition}-n{n}"
            _write_run(
                tmp_path, run_id, n=n, stop_reason="converged" if n != 10000 else "max_steps"
            )
            rows.extend(dss.run_rows(run_id, tmp_path))

    df = pd.DataFrame(rows)
    out = tmp_path / "figures" / "dataset_size_sweep.png"
    dss.plot(df, out)

    assert out.is_file()
    assert out.stat().st_size > 0


def test_manifest_round_trip_unaffected(tmp_path: Path) -> None:
    """Sanity: RunManifest still validates the hand-built experiment.* extras (unknown fields, V0.2)."""
    run_id = "evt-llama-fig2-inst-n1000"
    _write_run(tmp_path, run_id, n=1000)
    manifest = RunManifest.load(run_dir(run_id, store=tmp_path) / "manifest.json")
    manifest.validate()  # unknown "experiment" extra must not break validation
    assert manifest.data["experiment"]["target_result"]["stop_reason"] == "converged"


def test_ts38mw_default_run_ids_are_the_reused_base_plus_new_mw_pretaught() -> None:
    """default_run_ids('ts38mw') interleaves the SAME evt-ts38-base ids the ts38
    family points at with a NEW evt-ts38mw-pretaught arm, same order convention
    (per size: base then pretaught) as the ts38 family."""
    expected = [
        id_ for n in dss.TS38_SIZES for id_ in (f"evt-ts38-base-n{n}", f"evt-ts38mw-pretaught-n{n}")
    ]
    assert dss.default_run_ids("ts38mw") == expected
    assert len(expected) == 10


def test_ts38mw_ids_share_only_the_base_arm_with_ts38() -> None:
    """The ts38 and ts38mw sweeps share exactly the 5 base ids (reused, not
    retrained) and nothing else — the two families' own pretaught arms must
    never cross-match."""
    ts38_ids = set(dss.default_run_ids("ts38"))
    ts38mw_ids = set(dss.default_run_ids("ts38mw"))
    expected_shared = {f"evt-ts38-base-n{n}" for n in dss.TS38_SIZES}

    assert ts38_ids & ts38mw_ids == expected_shared
    assert len(expected_shared) == 5
    assert not any(rid.startswith("evt-ts38mw-") for rid in ts38_ids)
    assert not any(
        rid.startswith("evt-ts38-pretaught-") for rid in ts38mw_ids
    )  # ts38's own pretaught arm never leaks into ts38mw


def test_ts38mw_run_id_regex_matches_only_intended_ids() -> None:
    assert dss.RUN_ID_RE.match("evt-ts38mw-pretaught-n1000") is not None
    assert dss.RUN_ID_RE.match("evt-ts38-base-n1000") is not None
    assert dss.RUN_ID_RE.match("evt-ts38mw-base-n1000") is None
    assert dss.RUN_ID_RE.match("evt-ts38-pretaught-mw-n1000") is None


def test_parse_run_id_ts38mw_pretaught() -> None:
    assert dss._parse_run_id("evt-ts38mw-pretaught-n4642") == ("inst", "pre-taught-mw (elicit)")


def test_parse_run_id_ts38_base_shared_with_ts38mw_family() -> None:
    """evt-ts38-base-n1000 is the id BOTH families point at for the base arm;
    it must parse identically (same condition, same curve label) regardless
    of which family's sweep it was collected under — that's what makes reuse
    safe."""
    assert dss._parse_run_id("evt-ts38-base-n1000") == ("noinst", "base (teach)")


def test_parse_run_id_ts38_pretaught_regression() -> None:
    """The plain ts38 family's own pretaught arm must keep parsing exactly as
    before — adding the ts38mw branch to RUN_ID_RE must not touch this."""
    assert dss._parse_run_id("evt-ts38-pretaught-n1000") == ("inst", "pre-taught (elicit)")


def test_parse_run_id_rejects_ts38mw_base_and_malformed_ids() -> None:
    """There is no evt-ts38mw-base arm (base is only ever the shared
    evt-ts38- id), and a stray '-mw-' infix on the plain ts38 pretaught id
    must not be silently accepted."""
    for bad_id in ("evt-ts38mw-base-n1000", "evt-ts38-pretaught-mw-n1000"):
        with pytest.raises(ValueError):
            dss._parse_run_id(bad_id)


def test_ts38mw_pretaught_id_round_trips_through_run_rows(tmp_path: Path) -> None:
    run_id = "evt-ts38mw-pretaught-n1000"
    _write_run(tmp_path, run_id, n=1000)

    rows = dss.run_rows(run_id, tmp_path)
    assert rows and all(r["condition"] == "inst" for r in rows)
    assert rows[0]["curve_label"] == "pre-taught-mw (elicit)"


def test_ts38mw_stem_distinct_from_ts38() -> None:
    assert dss.FAMILIES["ts38mw"][1] == "dataset_size_sweep_ts38mw"
    assert dss.FAMILIES["ts38mw"][1] != dss.FAMILIES["ts38"][1]


def test_ts38pp_default_run_ids_are_the_reused_base_plus_new_pp_pretaught() -> None:
    """default_run_ids('ts38pp') interleaves the SAME evt-ts38-base ids the ts38
    family points at with a NEW evt-ts38pp-pretaught arm, same order convention
    (per size: base then pretaught) as the ts38 family."""
    expected = [
        id_ for n in dss.TS38_SIZES for id_ in (f"evt-ts38-base-n{n}", f"evt-ts38pp-pretaught-n{n}")
    ]
    assert dss.default_run_ids("ts38pp") == expected
    assert len(expected) == 10


def test_ts38pp_ids_share_only_the_base_arm_with_ts38() -> None:
    """The ts38 and ts38pp sweeps share exactly the 5 base ids (reused, not
    retrained) and nothing else — the two families' own pretaught arms must
    never cross-match."""
    ts38_ids = set(dss.default_run_ids("ts38"))
    ts38pp_ids = set(dss.default_run_ids("ts38pp"))
    expected_shared = {f"evt-ts38-base-n{n}" for n in dss.TS38_SIZES}

    assert ts38_ids & ts38pp_ids == expected_shared
    assert len(expected_shared) == 5
    assert not any(rid.startswith("evt-ts38pp-") for rid in ts38_ids)
    assert not any(
        rid.startswith("evt-ts38-pretaught-") for rid in ts38pp_ids
    )  # ts38's own pretaught arm never leaks into ts38pp


def test_ts38pp_ids_share_only_the_base_arm_with_ts38mw() -> None:
    """Same guarantee as above, between the two straddling-prefix families
    themselves: their own pretaught arms must never cross-match each other."""
    ts38mw_ids = set(dss.default_run_ids("ts38mw"))
    ts38pp_ids = set(dss.default_run_ids("ts38pp"))
    expected_shared = {f"evt-ts38-base-n{n}" for n in dss.TS38_SIZES}

    assert ts38mw_ids & ts38pp_ids == expected_shared
    assert not any(rid.startswith("evt-ts38mw-") for rid in ts38pp_ids)
    assert not any(rid.startswith("evt-ts38pp-") for rid in ts38mw_ids)


def test_ts38pp_run_id_regex_matches_only_intended_ids() -> None:
    assert dss.RUN_ID_RE.match("evt-ts38pp-pretaught-n1000") is not None
    assert dss.RUN_ID_RE.match("evt-ts38-base-n1000") is not None
    assert dss.RUN_ID_RE.match("evt-ts38pp-base-n1000") is None
    assert dss.RUN_ID_RE.match("evt-ts38-pretaught-pp-n1000") is None


def test_parse_run_id_ts38pp_pretaught() -> None:
    assert dss._parse_run_id("evt-ts38pp-pretaught-n4642") == ("inst", "pre-teach 4M full-FT")


def test_parse_run_id_ts38_base_shared_with_ts38pp_family() -> None:
    """evt-ts38-base-n1000 is the id BOTH families point at for the base arm;
    it must parse identically (same condition, same curve label) regardless
    of which family's sweep it was collected under — that's what makes reuse
    safe."""
    assert dss._parse_run_id("evt-ts38-base-n1000") == ("noinst", "base (teach)")


def test_parse_run_id_rejects_ts38pp_base_and_malformed_ids() -> None:
    """There is no evt-ts38pp-base arm (base is only ever the shared
    evt-ts38- id), and a stray '-pp-' infix on the plain ts38 pretaught id
    must not be silently accepted."""
    for bad_id in ("evt-ts38pp-base-n1000", "evt-ts38-pretaught-pp-n1000"):
        with pytest.raises(ValueError):
            dss._parse_run_id(bad_id)


def test_ts38pp_pretaught_id_round_trips_through_run_rows(tmp_path: Path) -> None:
    run_id = "evt-ts38pp-pretaught-n1000"
    _write_run(tmp_path, run_id, n=1000)

    rows = dss.run_rows(run_id, tmp_path)
    assert rows and all(r["condition"] == "inst" for r in rows)
    assert rows[0]["curve_label"] == "pre-teach 4M full-FT"


def test_ts38pp_stem_distinct_from_ts38() -> None:
    assert dss.FAMILIES["ts38pp"][1] == "dataset_size_sweep_ts38pp"
    assert dss.FAMILIES["ts38pp"][1] not in (dss.FAMILIES["ts38"][1], dss.FAMILIES["ts38mw"][1])


def test_all_family_stems_distinct() -> None:
    stems = [dss.FAMILIES[f][1] for f in dss.FAMILIES]
    assert len(stems) == len(set(stems))
    assert "ts38mw" in dss.FAMILIES
    assert "ts38pp" in dss.FAMILIES
