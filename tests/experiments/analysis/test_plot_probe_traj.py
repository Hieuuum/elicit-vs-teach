"""``plot_probe_traj.py`` -- probe-trajectory figure (carry-subset accuracy +
cross-format transfer vs. training step) for the ts38mt probe-trajectory
run (decisions.md 2026-08-22 "probe trajectory").

Silent failure modes guarded:

- ``trajectory_table``/``transfer_table`` picking the wrong layer as "best"
  (must be an argmax over ``acc_affected``, not e.g. the last row) or the
  wrong layer as "final" (must be the LARGEST layer present, not the last
  row in file order);
- ``checkpoint_step == -1`` (theta0) getting silently dropped or coerced
  instead of surviving into the table as-is;
- more than one row per (model, checkpoint_step) in the table (a groupby
  bug would silently duplicate or drop a step);
- a missing ``probe_traj_*`` file crashing ``main`` instead of being skipped.

CPU-only, no real models -- hand-built frames and small on-disk CSVs only.
matplotlib is a ``dev``-only dependency (pyproject.toml), so this whole
module is skipped if it is not installed.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from tests._scriptloader import load

ppt = load("plot_probe_traj")


class TestTrajectoryTable:
    def test_hand_computed(self):
        df = pd.DataFrame(
            [
                {
                    "model": "base",
                    "checkpoint_step": -1,
                    "layer": 0,
                    "acc_affected": 0.20,
                    "majority_affected_acc": 0.25,
                },
                {
                    "model": "base",
                    "checkpoint_step": -1,
                    "layer": 1,
                    "acc_affected": 0.50,
                    "majority_affected_acc": 0.25,
                },
                {
                    "model": "base",
                    "checkpoint_step": -1,
                    "layer": 2,
                    "acc_affected": 0.40,
                    "majority_affected_acc": 0.25,
                },
                {
                    "model": "base",
                    "checkpoint_step": 100,
                    "layer": 0,
                    "acc_affected": 0.30,
                    "majority_affected_acc": 0.25,
                },
                {
                    "model": "base",
                    "checkpoint_step": 100,
                    "layer": 1,
                    "acc_affected": 0.90,
                    "majority_affected_acc": 0.25,
                },
                {
                    "model": "base",
                    "checkpoint_step": 100,
                    "layer": 2,
                    "acc_affected": 0.60,
                    "majority_affected_acc": 0.25,
                },
                {
                    "model": "pp",
                    "checkpoint_step": 100,
                    "layer": 0,
                    "acc_affected": 0.10,
                    "majority_affected_acc": 0.20,
                },
                {
                    "model": "pp",
                    "checkpoint_step": 100,
                    "layer": 1,
                    "acc_affected": 0.15,
                    "majority_affected_acc": 0.20,
                },
            ]
        )
        table = ppt.trajectory_table(df)

        # one row per (model, checkpoint_step)
        assert len(table) == 3
        assert set(zip(table["model"], table["checkpoint_step"])) == {
            ("base", -1),
            ("base", 100),
            ("pp", 100),
        }
        # step -1 kept as-is (not dropped, not coerced to 0)
        assert -1 in set(table["checkpoint_step"])

        base_theta0 = table[(table["model"] == "base") & (table["checkpoint_step"] == -1)].iloc[0]
        assert base_theta0["acc_affected_best"] == 0.50
        assert base_theta0["best_layer"] == 1
        assert base_theta0["acc_affected_l8"] == 0.40  # layer 2 = max layer present
        assert base_theta0["majority_affected_acc"] == 0.25

        base_100 = table[(table["model"] == "base") & (table["checkpoint_step"] == 100)].iloc[0]
        assert base_100["acc_affected_best"] == 0.90
        assert base_100["best_layer"] == 1
        assert base_100["acc_affected_l8"] == 0.60

        pp_100 = table[table["model"] == "pp"].iloc[0]
        assert pp_100["acc_affected_best"] == 0.15
        assert pp_100["best_layer"] == 1
        assert pp_100["acc_affected_l8"] == 0.15  # layer 1 is both best and max here

    def test_best_layer_can_be_layer_zero(self):
        df = pd.DataFrame(
            [
                {
                    "model": "m",
                    "checkpoint_step": 1,
                    "layer": 0,
                    "acc_affected": 0.95,
                    "majority_affected_acc": 0.3,
                },
                {
                    "model": "m",
                    "checkpoint_step": 1,
                    "layer": 1,
                    "acc_affected": 0.40,
                    "majority_affected_acc": 0.3,
                },
            ]
        )
        row = ppt.trajectory_table(df).iloc[0]
        assert row["best_layer"] == 0
        assert row["acc_affected_best"] == 0.95
        assert row["acc_affected_l8"] == 0.40  # layer 1 is the max layer present

    def test_empty_input_gives_empty_table_with_columns(self):
        empty = pd.DataFrame(
            columns=["model", "checkpoint_step", "layer", "acc_affected", "majority_affected_acc"]
        )
        table = ppt.trajectory_table(empty)
        assert table.empty
        assert list(table.columns) == list(ppt.TRAJECTORY_COLUMNS)
        assert (table["model"] == "nonexistent").tolist() == []  # ["model"] lookup does not raise


class TestTransferTable:
    def test_hand_computed_filters_by_direction(self):
        df = pd.DataFrame(
            [
                {
                    "model": "base",
                    "checkpoint_step": -1,
                    "direction": "op_to_task",
                    "layer": 0,
                    "acc_affected": 0.30,
                    "majority_affected_acc": 0.20,
                },
                {
                    "model": "base",
                    "checkpoint_step": -1,
                    "direction": "op_to_task",
                    "layer": 1,
                    "acc_affected": 0.60,
                    "majority_affected_acc": 0.20,
                },
                {
                    "model": "base",
                    "checkpoint_step": -1,
                    "direction": "task_to_task",
                    "layer": 0,
                    "acc_affected": 0.90,
                    "majority_affected_acc": 0.20,
                },
                {
                    "model": "base",
                    "checkpoint_step": 100,
                    "direction": "op_to_task",
                    "layer": 0,
                    "acc_affected": 0.40,
                    "majority_affected_acc": 0.20,
                },
            ]
        )
        op_task = ppt.transfer_table(df, "op_to_task")
        assert len(op_task) == 2  # (base,-1) and (base,100) -- task_to_task row excluded
        assert set(op_task["checkpoint_step"]) == {-1, 100}
        row = op_task[op_task["checkpoint_step"] == -1].iloc[0]
        assert row["acc_affected_best"] == 0.60
        assert row["best_layer"] == 1

        ceiling = ppt.transfer_table(df, "task_to_task")
        assert len(ceiling) == 1
        assert ceiling.iloc[0]["acc_affected_best"] == 0.90

    def test_default_direction_is_op_to_task(self):
        df = pd.DataFrame(
            [
                {
                    "model": "m",
                    "checkpoint_step": 1,
                    "direction": "op_to_task",
                    "layer": 0,
                    "acc_affected": 0.55,
                    "majority_affected_acc": 0.1,
                },
                {
                    "model": "m",
                    "checkpoint_step": 1,
                    "direction": "task_to_op",
                    "layer": 0,
                    "acc_affected": 0.11,
                    "majority_affected_acc": 0.1,
                },
            ]
        )
        default = ppt.transfer_table(df)
        explicit = ppt.transfer_table(df, "op_to_task")
        pd.testing.assert_frame_equal(default, explicit)

    def test_missing_direction_gives_empty_table_with_columns(self):
        # a killed-mid-write transfer sweep can carry rows for SOME
        # directions (TRANSFER_DIRECTIONS order) and none yet for others --
        # the result must still be a real, correctly-columned frame (not a
        # columnless pd.DataFrame([])), or a downstream ["model"] lookup
        # (_plot_transfer, the summary merge) raises KeyError instead of
        # degrading gracefully.
        df = pd.DataFrame(
            [
                {
                    "model": "m",
                    "checkpoint_step": 1,
                    "direction": "task_to_op",
                    "layer": 0,
                    "acc_affected": 0.11,
                    "majority_affected_acc": 0.1,
                },
            ]
        )
        table = ppt.transfer_table(df, "op_to_task")
        assert table.empty
        assert list(table.columns) == list(ppt.TRANSFER_COLUMNS)
        assert (table["model"] == "nonexistent").tolist() == []  # ["model"] lookup does not raise

    def test_empty_input_gives_empty_table_with_columns(self):
        empty = pd.DataFrame(
            columns=[
                "model",
                "checkpoint_step",
                "direction",
                "layer",
                "acc_affected",
                "majority_affected_acc",
            ]
        )
        table = ppt.transfer_table(empty, "op_to_task")
        assert table.empty
        assert list(table.columns) == list(ppt.TRANSFER_COLUMNS)


def _routing_rows(model: str, steps: list[int], majority: float = 0.25) -> pd.DataFrame:
    rows = []
    for step in steps:
        for layer, acc in enumerate([0.2 + 0.1 * step_i for step_i in range(3)]):
            rows.append(
                {
                    "model": model,
                    "checkpoint_step": step,
                    "set": "task",
                    "layer": layer,
                    "acc_affected": acc + 0.05 * step,
                    "majority_affected_acc": majority,
                }
            )
    return pd.DataFrame(rows)


_ALL_DIRECTIONS = ("op_to_task", "task_to_op", "op_to_op", "task_to_task")


def _transfer_rows(
    model: str, steps: list[int], majority: float = 0.2, directions=_ALL_DIRECTIONS
) -> pd.DataFrame:
    rows = []
    for step in steps:
        for direction in directions:
            for layer in range(2):
                base = 0.9 if direction == "task_to_task" else 0.3
                rows.append(
                    {
                        "model": model,
                        "checkpoint_step": step,
                        "direction": direction,
                        "layer": layer,
                        "acc_affected": base + 0.02 * layer + 0.01 * step,
                        "majority_affected_acc": majority,
                    }
                )
    return pd.DataFrame(rows)


class TestPlotTraj:
    def test_chance_line_uses_first_available_arm_deterministically(self):
        # base and pp share the same majority_affected_acc in real data (one
        # RoutingReference per set); when they differ (as here, deliberately)
        # the chance line must come from whichever arm is FIRST in ARMS
        # order, not whichever was drawn last.
        table = pd.DataFrame(
            [
                {
                    "model": "pp",
                    "checkpoint_step": 1,
                    "acc_affected_best": 0.5,
                    "majority_affected_acc": 0.99,
                },
                {
                    "model": "base",
                    "checkpoint_step": 1,
                    "acc_affected_best": 0.5,
                    "majority_affected_acc": 0.25,
                },
            ]
        )
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            chance = ppt._plot_traj(ax, table)
        finally:
            plt.close(fig)
        assert chance == 0.25  # base sorts before pp in ARMS


class TestMainSmoke:
    def test_writes_figure_and_summary_skipping_missing_arms(self, tmp_path, monkeypatch):
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # base: full sweep + its own transfer sweep.
        _routing_rows("base", [1, 3]).to_csv(results_dir / "probe_traj_base.csv", index=False)
        _transfer_rows("base", [1, 3]).to_csv(
            results_dir / "probe_traj_base_transfer.csv", index=False
        )

        # theta0: base + pp (fmt, k7 never appear anywhere -- fully missing).
        theta0 = pd.concat([_routing_rows("base", [-1]), _routing_rows("pp", [-1], majority=0.2)])
        theta0.to_csv(results_dir / "probe_traj_theta0.csv", index=False)
        theta0_transfer = pd.concat([_transfer_rows("base", [-1]), _transfer_rows("pp", [-1])])
        theta0_transfer.to_csv(results_dir / "probe_traj_theta0_transfer.csv", index=False)

        out = tmp_path / "fig.png"
        argv = [
            "plot_probe_traj.py",
            "--results-dir",
            str(results_dir),
            "--n",
            "999",
            "--out",
            str(out),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        ppt.main()  # must not raise despite fmt/k7 being entirely absent

        assert out.is_file()
        summary_path = out.with_name(f"{out.stem}_summary.csv")
        assert summary_path.is_file()
        summary = pd.read_csv(summary_path)
        assert set(summary["model"]) == {"base", "pp"}
        # base has theta0 (-1) and the two sweep steps; pp only has theta0.
        base_steps = set(summary[summary["model"] == "base"]["checkpoint_step"])
        assert base_steps == {-1, 1, 3}
        pp_steps = set(summary[summary["model"] == "pp"]["checkpoint_step"])
        assert pp_steps == {-1}

    def test_killed_mid_write_transfer_file_does_not_crash(self, tmp_path, monkeypatch):
        # a transfer sweep file that only has op_to_task rows so far (e.g.
        # scp'd mid-write, or transfer_rows_from_feats interrupted partway
        # through TRANSFER_DIRECTIONS) must not crash main() when the
        # task_to_task ceiling line has nothing to plot.
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _routing_rows("base", [1]).to_csv(results_dir / "probe_traj_base.csv", index=False)
        _transfer_rows("base", [1], directions=("op_to_task",)).to_csv(
            results_dir / "probe_traj_base_transfer.csv", index=False
        )
        out = tmp_path / "fig.png"
        argv = [
            "plot_probe_traj.py",
            "--results-dir",
            str(results_dir),
            "--out",
            str(out),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        ppt.main()  # must not raise
        assert out.is_file()
        summary = pd.read_csv(out.with_name(f"{out.stem}_summary.csv"))
        assert set(summary["model"]) == {"base"}
        assert summary["transfer_acc_affected_best"].notna().all()

    def test_raises_when_nothing_found(self, tmp_path, monkeypatch):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        argv = [
            "plot_probe_traj.py",
            "--results-dir",
            str(empty_dir),
            "--out",
            str(tmp_path / "out.png"),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            ppt.main()
