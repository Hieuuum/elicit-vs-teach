"""``ts38mt_mech_summary.py`` — folds the ten drivers' per-run tables into one
row per (arm, n). Silent failure modes guarded:

- picking the wrong snapshot as "first"/"final" (sort by step, not file
  order), or reporting a half-rise step fraction when the margin never rose;
- a grad-dynamics lookup returning the wrong run's row from the combined
  table, or a per-run file being shadowed by the combined one;
- the ΔW-mass weighting in ``weight_diff.summary_metrics`` degenerating to a
  plain mean, or silently reporting 0 where the column is absent;
- the residual-shift ratio dividing by a zero generic shift;
- cross-patch "best" picked across directions/scopes instead of within one;
- a missing table silently producing a full-looking row instead of NaNs plus
  a named entry in ``missing``.

CPU-only, synthetic tables only (no checkpoints, no network).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from tests._scriptloader import load

summ = load("ts38mt_mech_summary")
wd = load("weight_diff")


# ---------------------------------------------------------------- fixtures


def probe_rows(run: str, steps, margins, floor=0.25, n_layers=3, set_name="task"):
    """Per-layer probe rows where the best layer (= last) sits ``margin``
    above the layer-0 floor at each step."""
    rows = []
    for step, m in zip(steps, margins):
        for layer in range(n_layers):
            acc = floor if layer == 0 else floor + m * layer / (n_layers - 1)
            rows.append(
                {
                    "model": run,
                    "checkpoint_step": step,
                    "set": set_name,
                    "layer": layer,
                    "hook_name": f"h{layer}",
                    "probe_train_acc": acc,
                    "probe_test_acc": acc,
                    "majority_test_acc": floor,
                    "n_classes": 10,
                    "n_train": 100,
                    "n_test": 100,
                }
            )
    return pd.DataFrame(rows)


def shift_rows(task, generic, cos=0.9, evr=0.8):
    rows = []
    for layer, v in enumerate(task):
        rows.append(
            {
                "layer": layer,
                "set": "task",
                "rel_shift": v,
                "mean_cos_to_mean": cos,
                "top_pc_evr": evr,
                "n": 5,
            }
        )
    for layer, v in enumerate(generic):
        rows.append(
            {
                "layer": layer,
                "set": "generic",
                "rel_shift": v,
                "mean_cos_to_mean": 0.1,
                "top_pc_evr": 0.2,
                "n": 5,
            }
        )
    return pd.DataFrame(rows)


def jac_rows(cos, ratio, gain=2.0):
    rows = []
    for layer, (c, r) in enumerate(zip(cos, ratio)):
        rows.append(
            {
                "model": "a",
                "set": "task",
                "layer": layer,
                "n": 5,
                "mean_jac_norm": 1.0,
                "jac_norm_rel": 1.0,
                "mean_cos_to_mean": 0.5,
                "top_pc_evr": 0.5,
                "mean_cos_shift_vs_jac0": c,
                "mean_pred_gain_nats": r * gain,
                "actual_gain_nats": gain,
                "pred_gain_ratio": r,
            }
        )
    # theta_T rows carry no bridge columns
    for layer in range(len(cos)):
        rows.append(
            {
                "model": "b",
                "set": "task",
                "layer": layer,
                "n": 5,
                "mean_jac_norm": 1.0,
                "jac_norm_rel": 1.0,
                "mean_cos_to_mean": 0.5,
                "top_pc_evr": 0.5,
                "mean_cos_shift_vs_jac0": math.nan,
                "mean_pred_gain_nats": math.nan,
                "actual_gain_nats": math.nan,
                "pred_gain_ratio": math.nan,
            }
        )
    return pd.DataFrame(rows)


def cp_rows(rec: dict[tuple[str, str], list[float]]):
    rows = []
    for (direction, scope), vals in rec.items():
        for layer, v in enumerate(vals):
            rows.append(
                {
                    "layer": layer,
                    "scope": scope,
                    "direction": direction,
                    "metric_patched": v,
                    "top1_patched": v,
                    "recovery_frac": v,
                }
            )
    return pd.DataFrame(rows)


def wd_rows(modules, total_rel_fro=0.05):
    """``modules``: list of (fro_dw, effective_rank, overlap_32)."""
    rows = []
    for i, (fro_dw, er, ov) in enumerate(modules):
        rows.append(
            {
                "level": "module",
                "layer": i,
                "module": "q_proj",
                "rel_fro": 0.01,
                "fro_dw": fro_dw,
                "fro_w0": 100.0,
                "effective_rank": er,
                "overlap_32": ov,
            }
        )
    rows.append(
        {
            "level": "total",
            "layer": -2,
            "module": "all",
            "rel_fro": total_rel_fro,
            "fro_dw": 1.0,
            "fro_w0": 20.0,
            "effective_rank": math.nan,
            "overlap_32": math.nan,
        }
    )
    return pd.DataFrame(rows)


def gd_rows(run_ids, metric_base=0.1):
    rows = []
    for i, rid in enumerate(run_ids):
        rows.append({"level": "step", "run_id": rid, "step": 1, "grad_norm": 1.0})
        rows.append(
            {
                "level": "run",
                "run_id": rid,
                "n_steps": 100 * (i + 1),
                **{k: metric_base * (i + 1) for k in summ.GD_METRICS if k != "n_steps"},
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- test 1


class TestProbeRunSummary:
    def test_first_final_max_by_step_not_file_order(self):
        df = probe_rows("r", steps=[100, 1, 10], margins=[0.4, 0.0, 0.2])
        out = summ.probe_run_summary(df)
        assert out["probe_step_first"] == 1 and out["probe_step_final"] == 100
        assert out["probe_margin_first"] == pytest.approx(0.0)
        assert out["probe_margin_final"] == pytest.approx(0.4)
        assert out["probe_margin_max"] == pytest.approx(0.4)
        assert out["probe_best_layer_final"] == 2
        assert out["probe_floor"] == pytest.approx(0.25)

    def test_half_step_frac_is_first_crossing_over_final_step(self):
        df = probe_rows("r", steps=[1, 10, 100], margins=[0.0, 0.2, 0.4])
        out = summ.probe_run_summary(df)
        assert out["probe_margin_half_step_frac"] == pytest.approx(10 / 100)

    def test_half_step_frac_nan_when_margin_does_not_rise(self):
        df = probe_rows("r", steps=[1, 10], margins=[0.3, 0.3])
        assert math.isnan(summ.probe_run_summary(df)["probe_margin_half_step_frac"])
        df = probe_rows("r", steps=[1, 10], margins=[0.3, 0.1])
        assert math.isnan(summ.probe_run_summary(df)["probe_margin_half_step_frac"])

    def test_op_set_ignored_and_missing_task_refuses(self):
        task = probe_rows("r", steps=[1, 10], margins=[0.1, 0.2])
        op = probe_rows("r", steps=[1, 10], margins=[0.9, 0.9], set_name="op")
        out = summ.probe_run_summary(pd.concat([task, op]))
        assert out["probe_margin_final"] == pytest.approx(0.2)
        with pytest.raises(ValueError):
            summ.probe_run_summary(op)

    def test_missing_layer0_refuses(self):
        df = probe_rows("r", steps=[1], margins=[0.2])
        with pytest.raises(ValueError):
            summ.probe_run_summary(df[df["layer"] != 0])


# ---------------------------------------------------------------- test 8


class TestGradRunSummary:
    def test_picks_the_named_run_from_combined_table(self):
        df = gd_rows(["evt-a", "evt-b"])
        out = summ.grad_run_summary(df, "evt-b")
        assert out["gd_n_steps"] == 200
        assert out["gd_grad_early_mass_frac"] == pytest.approx(0.2)
        assert set(out) == {f"gd_{k}" for k in summ.GD_METRICS}

    def test_unknown_run_refuses(self):
        with pytest.raises(ValueError):
            summ.grad_run_summary(gd_rows(["evt-a"]), "evt-zzz")

    def test_per_run_file_takes_precedence_over_combined(self, tmp_path):
        rid = "evt-ts38mt-pp-n1000"
        gd_rows([rid], metric_base=0.5).to_csv(tmp_path / "grad_dynamics_all.csv", index=False)
        gd_rows([rid], metric_base=0.1).to_csv(tmp_path / f"grad_dynamics_{rid}.csv", index=False)
        assert summ._grad_table(tmp_path, rid).name == f"grad_dynamics_{rid}.csv"
        (tmp_path / f"grad_dynamics_{rid}.csv").unlink()
        assert summ._grad_table(tmp_path, rid).name == "grad_dynamics_all.csv"
        (tmp_path / "grad_dynamics_all.csv").unlink()
        assert summ._grad_table(tmp_path, rid) is None


# ---------------------------------------------------------------- test 9


class TestWeightDiffSummaryMetrics:
    def test_mass_weighted_by_fro_dw_squared(self):
        df = wd_rows([(1.0, 10.0, 0.1), (2.0, 20.0, 0.6)], total_rel_fro=0.07)
        m = wd.summary_metrics(df)
        # weights 1 and 4
        assert m["rel_fro"] == pytest.approx(0.07)
        assert m["effective_rank"] == pytest.approx((10 + 80) / 5)
        assert m["overlap_32"] == pytest.approx((0.1 + 2.4) / 5)

    def test_absent_overlap_column_is_not_reported_and_nan_erank_rows_dropped(self):
        df = wd_rows([(1.0, 10.0, 0.1), (3.0, math.nan, 0.9)]).drop(columns=["overlap_32"])
        m = wd.summary_metrics(df)
        assert "overlap_32" not in m
        assert m["effective_rank"] == pytest.approx(10.0)

    def test_zero_mass_gives_nan_not_zero(self):
        df = wd_rows([(0.0, 10.0, 0.1)])
        m = wd.summary_metrics(df)
        assert math.isnan(m["effective_rank"]) and math.isnan(m["overlap_32"])

    def test_print_summary_still_runs(self, capsys):
        wd.print_summary(wd_rows([(1.0, 10.0, 0.1)]).to_dict("records"))
        assert "rel_fro=" in capsys.readouterr().out

    def test_prefixed_in_aggregator(self):
        out = summ.weight_run_summary(wd_rows([(1.0, 10.0, 0.1)]))
        assert set(out) == {"wd_rel_fro", "wd_effective_rank", "wd_overlap_32"}


# ---------------------------------------------------------------- test 10


class TestShiftRunSummary:
    def test_peak_layer_ratio_and_consistency(self):
        df = shift_rows(task=[0.0, 0.2, 0.5, 0.3], generic=[0.0, 0.1, 0.1, 0.3], cos=0.95, evr=0.9)
        out = summ.shift_run_summary(df)
        assert out["shift_peak_layer"] == 2
        assert out["shift_task_at_peak"] == pytest.approx(0.5)
        assert out["shift_generic_at_peak"] == pytest.approx(0.1)
        assert out["shift_ratio_at_peak"] == pytest.approx(5.0)
        assert out["shift_task_cos_at_peak"] == pytest.approx(0.95)
        assert out["shift_task_evr_at_peak"] == pytest.approx(0.9)

    def test_zero_generic_gives_nan_ratio(self):
        df = shift_rows(task=[0.0, 0.4], generic=[0.0, 0.0])
        assert math.isnan(summ.shift_run_summary(df)["shift_ratio_at_peak"])

    def test_missing_generic_set_gives_nan_generic(self):
        df = shift_rows(task=[0.0, 0.4], generic=[])
        out = summ.shift_run_summary(df)
        assert math.isnan(out["shift_generic_at_peak"]) and math.isnan(out["shift_ratio_at_peak"])

    def test_no_task_rows_refuses(self):
        with pytest.raises(ValueError):
            summ.shift_run_summary(shift_rows(task=[], generic=[0.1]))


# ---------------------------------------------------------------- test 7


class TestJacobianRunSummary:
    def test_best_cos_layer_and_ratio_there(self):
        df = jac_rows(cos=[0.0, 0.3, 0.1], ratio=[0.1, 0.4, 0.9], gain=3.0)
        out = summ.jacobian_run_summary(df)
        assert out["jac_actual_gain_nats"] == pytest.approx(3.0)
        assert out["jac_best_cos_layer"] == 1
        assert out["jac_best_cos"] == pytest.approx(0.3)
        assert out["jac_pred_gain_ratio_at_best_cos"] == pytest.approx(0.4)
        assert out["jac_pred_gain_ratio_max"] == pytest.approx(0.9)

    def test_no_bridge_rows_gives_all_nan(self):
        df = jac_rows(cos=[0.2], ratio=[0.3])
        df = df[df["model"] == "b"]
        out = summ.jacobian_run_summary(df)
        assert set(out) == set(summ.JAC_KEYS) and all(math.isnan(v) for v in out.values())
        out = summ.jacobian_run_summary(df.drop(columns=["mean_cos_shift_vs_jac0"]))
        assert all(math.isnan(v) for v in out.values())


# ---------------------------------------------------------------- test 4


class TestCrossPatchRunSummary:
    def test_best_within_each_direction_scope_and_first_ge_half(self):
        df = cp_rows(
            {
                ("T_into_0", "answer"): [0.0, 0.2, 0.9, 0.6],
                ("T_into_0", "all"): [0.0, 0.6, 0.7, 1.0],
                ("0_into_T", "answer"): [1.0, 0.8, 0.1, 0.5],
                ("0_into_T", "all"): [1.0, 0.0, 0.2, 0.3],
            }
        )
        out = summ.cross_patch_run_summary(df)
        assert out["cp_T0_answer_best_layer"] == 2 and out[
            "cp_T0_answer_best_rec"
        ] == pytest.approx(0.9)
        assert out["cp_T0_answer_first_layer_ge_half"] == 2
        assert out["cp_T0_all_best_layer"] == 3 and out["cp_T0_all_first_layer_ge_half"] == 1
        assert out["cp_0T_answer_min_layer"] == 2 and out["cp_0T_answer_min_rec"] == pytest.approx(
            0.1
        )
        assert out["cp_0T_all_min_layer"] == 1

    def test_never_reaches_half_and_missing_groups_are_nan(self):
        df = cp_rows({("T_into_0", "answer"): [0.0, 0.1, 0.2]})
        out = summ.cross_patch_run_summary(df)
        assert math.isnan(out["cp_T0_answer_first_layer_ge_half"])
        assert out["cp_T0_answer_best_layer"] == 2
        assert math.isnan(out["cp_T0_all_best_rec"]) and math.isnan(out["cp_0T_answer_min_rec"])

    def test_nan_recovery_rows_ignored(self):
        df = cp_rows({("T_into_0", "answer"): [math.nan, 0.3, math.nan]})
        out = summ.cross_patch_run_summary(df)
        assert out["cp_T0_answer_best_layer"] == 1


# ---------------------------------------------------------------- assembly


def _write_cell(d, rid, with_cp=True):
    probe_rows(rid, steps=[1, 50], margins=[0.1, 0.3]).to_csv(
        d / f"resid_probe_{rid}.csv", index=False
    )
    wd_rows([(1.0, 10.0, 0.1)]).to_parquet(d / f"weight_diff_{rid}.parquet")
    shift_rows(task=[0.0, 0.4], generic=[0.0, 0.1]).to_csv(
        d / f"resid_shift_{rid}.csv", index=False
    )
    jac_rows(cos=[0.1, 0.2], ratio=[0.3, 0.4]).to_csv(d / f"jacobian_lens_{rid}.csv", index=False)
    if with_cp:
        cp_rows({("T_into_0", "answer"): [0.0, 0.8]}).to_csv(
            d / f"cross_patch_{rid}.csv", index=False
        )


class TestSummarizeGrid:
    def test_row_order_columns_and_missing_list(self, tmp_path):
        arms, sizes = ["base", "pp"], [1000, 2154]
        rids = [summ.run_id(a, n) for n in sizes for a in arms]
        _write_cell(tmp_path, rids[0])
        _write_cell(tmp_path, rids[1], with_cp=False)
        gd_rows(rids[:2]).to_csv(tmp_path / "grad_dynamics_all.csv", index=False)
        df, missing = summ.summarize_grid(tmp_path, arms, sizes)
        assert list(df["run_id"]) == rids  # n-major, then arm
        assert list(df.columns[:3]) == ["arm", "n", "run_id"]
        full = df.iloc[0]
        assert full["probe_margin_final"] == pytest.approx(0.3)
        assert full["gd_n_steps"] == 100
        assert full["wd_rel_fro"] == pytest.approx(0.05)
        assert full["shift_ratio_at_peak"] == pytest.approx(4.0)
        assert full["jac_best_cos"] == pytest.approx(0.2)
        assert full["cp_T0_answer_best_rec"] == pytest.approx(0.8)
        # second cell: cross_patch absent -> NaN + named
        assert math.isnan(df.iloc[1]["cp_T0_answer_best_rec"])
        assert f"cross_patch_{rids[1]}.csv" in missing
        # cells 3/4 entirely absent: 6 tables each
        for rid in rids[2:]:
            for stem in (
                "resid_probe",
                "resid_shift",
                "jacobian_lens",
                "cross_patch",
                "grad_dynamics",
            ):
                assert f"{stem}_{rid}.csv" in missing
            assert f"weight_diff_{rid}.parquet" in missing
            assert math.isnan(df[df["run_id"] == rid]["probe_margin_final"].iloc[0])
        assert len(missing) == 1 + 2 * 6

    def test_combined_grad_table_lacking_the_run_is_reported_missing(self, tmp_path):
        rid = summ.run_id("fmt", 1000)
        gd_rows(["evt-other"]).to_csv(tmp_path / "grad_dynamics_all.csv", index=False)
        df, missing = summ.summarize_grid(tmp_path, ["fmt"], [1000])
        assert f"grad_dynamics_{rid}.csv" in missing
        assert "gd_n_steps" not in df or math.isnan(df["gd_n_steps"].iloc[0])

    def test_print_grid_skips_all_nan_columns(self, tmp_path, capsys):
        df, _ = summ.summarize_grid(tmp_path, ["base"], [1000])
        summ.print_grid(df)
        assert capsys.readouterr().out == ""
        _write_cell(tmp_path, summ.run_id("base", 1000))
        df, _ = summ.summarize_grid(tmp_path, ["base"], [1000])
        summ.print_grid(df)
        out = capsys.readouterr().out
        assert "[evt] probe_margin_final" in out and "[evt] gd_grad_early_mass_frac" not in out


class TestSummarizePhase0:
    def _lens(self, d, m, s, top1):
        rows = [
            {
                "model": m,
                "set": s,
                "layer": i,
                "position_kind": "first_answer",
                "top1_acc": v,
                "mean_logprob_nats": -1.0,
                "mean_rank": 1.0,
                "n": 5,
            }
            for i, v in enumerate(top1)
        ]
        pd.DataFrame(rows).to_csv(d / f"logit_lens_{m}_{s}.csv", index=False)

    def test_rows_per_model_theta0_has_no_pair_metrics(self, tmp_path):
        for m in ("theta0", "thetaT"):
            pd.concat(
                [
                    probe_rows(m, steps=[-1], margins=[0.3]),
                    probe_rows(m, steps=[-1], margins=[0.5], floor=0.4, set_name="op"),
                ]
            ).to_csv(tmp_path / f"resid_probe_{m}.csv", index=False)
            self._lens(tmp_path, m, "task", [0.0, 0.0, 0.0])
            self._lens(tmp_path, m, "op", [0.0, 0.5, 0.9])
        wd_rows([(1.0, 10.0, 0.1)]).to_parquet(tmp_path / "weight_diff_thetaT.parquet")
        shift_rows(task=[0.0, 0.4], generic=[0.0, 0.2]).to_csv(
            tmp_path / "resid_shift_thetaT.csv", index=False
        )
        jac_rows(cos=[0.1], ratio=[0.3]).to_csv(tmp_path / "jacobian_lens_thetaT.csv", index=False)
        df, missing = summ.summarize_phase0(tmp_path, models=("theta0", "thetaT"))
        assert missing == []
        assert list(df["model"]) == ["theta0", "thetaT"]
        t0, tT = df.iloc[0], df.iloc[1]
        assert t0["probe_task_margin"] == pytest.approx(0.3)
        assert t0["probe_op_margin"] == pytest.approx(0.5)
        assert math.isnan(t0["lens_task_emergence_layer"]) and t0["lens_op_emergence_layer"] == 1
        assert math.isnan(t0["wd_rel_fro"]) and math.isnan(t0["shift_ratio_at_peak"])
        assert tT["wd_rel_fro"] == pytest.approx(0.05)
        assert tT["shift_ratio_at_peak"] == pytest.approx(2.0)
        assert tT["jac_best_cos"] == pytest.approx(0.1)

    def test_missing_tables_named(self, tmp_path):
        df, missing = summ.summarize_phase0(tmp_path, models=("theta0", "s7773"))
        assert len(df) == 2
        assert "resid_probe_theta0.csv" in missing
        assert "weight_diff_theta0.parquet" not in missing  # by design
        assert "weight_diff_s7773.parquet" in missing


class TestMain:
    def test_cli_writes_both_tables(self, tmp_path, monkeypatch, capsys):
        rid = summ.run_id("pp", 1000)
        _write_cell(tmp_path, rid)
        gd_rows([rid]).to_csv(tmp_path / "grad_dynamics_all.csv", index=False)
        out = tmp_path / "grid.csv"
        p0_out = tmp_path / "p0.csv"
        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--results-dir",
                str(tmp_path),
                "--arms",
                "pp",
                "--sizes",
                "1000",
                "--out",
                str(out),
                "--phase0-dir",
                str(tmp_path),
                "--phase0-out",
                str(p0_out),
            ],
        )
        summ.main()
        grid = pd.read_csv(out)
        assert len(grid) == 1 and grid["run_id"].iloc[0] == rid
        assert len(pd.read_csv(p0_out)) == len(summ.PHASE0_MODELS)
        text = capsys.readouterr().out
        assert "0 missing tables" in text and "phase0 missing: resid_probe_theta0.csv" in text

    def test_cli_requires_store_or_results_dir(self, monkeypatch):
        monkeypatch.delenv("GEODE_STORE", raising=False)
        monkeypatch.setattr("sys.argv", ["x"])
        with pytest.raises(SystemExit):
            summ.main()
