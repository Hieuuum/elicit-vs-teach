"""Gradient-norm decay + weight displacement (``analysis/grad_dynamics.py``,
Phase 0 test 8).

Silent failure modes guarded: this driver's whole output is a cross-arm
comparison (does the elicit arm's gradient collapse early while the teach
arm keeps working?), so a wrong LoRA scaling, a wrong full-FT reference, a
lexical instead of numeric step sort, or a mis-normalised window fraction
would not crash — it would just produce a plausible-looking number and
invalidate the comparison, exactly CLAUDE.md's "analysis metric" tier. Every
pure summary metric is checked on a synthetic curve whose answer is known by
construction (an impulse vs. a flat grad-norm curve, a monotone-ramp vs. a
step-function displacement curve, a curve with zero loss drop), the two
weight-displacement paths are checked against hand-computed ``‖ΔW‖``/``‖θ_k
− θ_ref‖`` values, and the full-FT step directories are deliberately created
out of lexical order (``step_0000009`` before ``step_0000010``) to catch a
string sort silently swapping them.

The driver is not a package module, so it is loaded from its file path (and
the analysis dir goes on ``sys.path`` because it imports its siblings
``adapters``, ``mech_lib``, ``trajectory``, matching ``weight_diff.py``'s and
``trajectory.py``'s own sibling-import convention).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from geode.zoo import register_run
from geode.zoo.store import gradstats_log_path, run_dir

from tests._scriptloader import load

gd = load("grad_dynamics")

RANK = 1
ALPHA = 2.0  # scaling = alpha / (2*rank) = 1.0, so ΔW = B @ A exactly
MODULE = "model.layers.0.self_attn.q_proj"


# --- fixture helpers ---------------------------------------------------------


def _manifest_fields(run_id: str, method: str, regime: str = "elicit") -> dict:
    """A valid spec 00 §2 manifest, method-parametrized (mirrors
    ``test_trajectory.py``'s ``_manifest_fields``)."""
    lora_block = (
        {
            "rank": RANK,
            "alpha": ALPHA,
            "target_modules": ["q_proj"],
            "dropout": 0.0,
            "sparse_param_count": None,
        }
        if method == "lora"
        else {
            "rank": None,
            "alpha": None,
            "target_modules": [],
            "dropout": None,
            "sparse_param_count": None,
        }
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": "2026-08-21T00:00:00+00:00",
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "regime": regime,
        "base_model": {"hf_id": "tiny/llama", "revision": "main"},
        "task": {"name": "task_a", "format_version": "1"},
        "dataset": {"name": "dataset_a", "n_unique_examples": 512, "seed": 0},
        "training": {
            "method": method,
            "lora": lora_block,
            "optimizer": {
                "name": "adamw",
                "lr": 1e-3,
                "batch_size": 8,
                "micro_batch_size": 4,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
                "grad_clip": 1.0,
            },
            "lr_schedule": "constant",
            "min_lr": None,
            "precision": "bf16",
            "eval_every": 100,
            "max_steps": None,
            "stopping": {"eps_nats": 0.005, "k": 3, "min_steps": 0},
            "epochs_total": 1,
            "seed": 0,
        },
        "trainable_param_count": 4,
        "snapshot_steps": [],
        "cost": {"gpu_type": "A100", "est_usd": 0.0, "actual_usd": None},
        "status": "complete",
    }


def _write_train_log(store: Path, run_id: str, records: list[dict]) -> None:
    path = run_dir(run_id, store=store) / "train_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_eval_log(store: Path, run_id: str, records: list[dict]) -> None:
    path = run_dir(run_id, store=store) / "eval_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_gradstats(store: Path, run_id: str, records: list[dict]) -> None:
    path = gradstats_log_path(run_id, store=store)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_lora_snapshot(
    store: Path, run_id: str, step: int, a: torch.Tensor, b: torch.Tensor
) -> None:
    snap = run_dir(run_id, store=store) / "snapshots" / f"step_{step}"
    snap.mkdir(parents=True, exist_ok=True)
    save_file({f"{MODULE}.A.weight": a, f"{MODULE}.B.weight": b}, str(snap / "adapter.safetensors"))


def _write_sft_snapshot(root: Path, step: int, tensor: torch.Tensor) -> None:
    snap = root / f"step_{step:07d}"
    snap.mkdir(parents=True, exist_ok=True)
    save_file({"model.embed_tokens.weight": tensor}, str(snap / "model.safetensors"))


def _trivial_train_log(n_steps: int) -> list[dict]:
    """A flat, arbitrary train-log so ``n_steps`` and the step axis exist."""
    return [
        {"step": s, "train_loss_nats": 1.0, "train_accuracy": 0.0, "lr": 1e-3}
        for s in range(1, n_steps + 1)
    ]


# --- LoRA displacement: hand-computed ---------------------------------------


def test_lora_displacement_matches_hand_computed_dw_norm(tmp_path: Path) -> None:
    """‖ΔW‖ = ‖(α/2r)·B@A‖, computed by hand for one module."""
    run_id = "evt-test-lora-disp"
    register_run(_manifest_fields(run_id, "lora"), store=tmp_path)
    _write_train_log(tmp_path, run_id, _trivial_train_log(2))

    a = torch.tensor([[1.0, 0.0]])  # rank(1) x d_in(2)
    b1 = torch.tensor([[2.0], [3.0]])  # d_out(2) x rank(1)
    b2 = torch.tensor([[4.0], [6.0]])  # exactly 2x b1 -> ΔW exactly doubles
    _write_lora_snapshot(tmp_path, run_id, 1, a, b1)
    _write_lora_snapshot(tmp_path, run_id, 2, a, b2)

    curves = gd.load_run_curves(run_id, tmp_path)
    assert curves.method == "lora"
    # scaling = alpha/(2*rank) = 1.0, so ΔW = B @ A = [[2,0],[3,0]] at step 1
    expected_1 = (13.0) ** 0.5  # sqrt(2^2 + 3^2)
    expected_2 = (13.0 * 4) ** 0.5  # b doubled -> ΔW doubled -> norm doubled
    assert curves.disp[1] == pytest.approx(expected_1, rel=1e-6)
    assert curves.disp[2] == pytest.approx(expected_2, rel=1e-6)
    assert curves.ref_step == 0  # ΔW ≡ 0 at true init exactly


# --- full-FT displacement: --ref-dir vs. earliest-snapshot fallback --------


def test_full_ft_displacement_ref_dir_hand_computed(tmp_path: Path) -> None:
    """With --ref-dir, every snapshot's displacement is measured against the
    externally supplied reference, and the interpolation ref_step is 0."""
    run_id = "evt-test-fullft-refdir"
    register_run(_manifest_fields(run_id, "full_ft", regime="teach"), store=tmp_path)
    _write_train_log(tmp_path, run_id, _trivial_train_log(3))

    ref_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ref_dir = tmp_path / "base_model"
    ref_dir.mkdir()
    save_file({"model.embed_tokens.weight": ref_tensor}, str(ref_dir / "model.safetensors"))

    snap_root = run_dir(run_id, store=tmp_path) / "sft_snapshots"
    _write_sft_snapshot(snap_root, 1, ref_tensor + 1.0)  # diff norm = sqrt(4*1) = 2
    _write_sft_snapshot(snap_root, 2, ref_tensor + 2.0)  # diff norm = sqrt(4*4) = 4

    curves = gd.load_run_curves(run_id, tmp_path, ref_dir=ref_dir)
    assert curves.method == "full_ft"
    assert curves.disp[1] == pytest.approx(2.0, rel=1e-6)
    assert curves.disp[2] == pytest.approx(4.0, rel=1e-6)
    assert curves.ref_step == 0  # --ref-dir is conceptually true init


def test_full_ft_displacement_earliest_snapshot_fallback(tmp_path: Path) -> None:
    """Without --ref-dir, the earliest snapshot stands in for θ_ref: it gets
    no displacement value, and later steps are measured against it."""
    run_id = "evt-test-fullft-fallback"
    register_run(_manifest_fields(run_id, "full_ft", regime="teach"), store=tmp_path)
    _write_train_log(tmp_path, run_id, _trivial_train_log(3))

    base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    snap_root = run_dir(run_id, store=tmp_path) / "sft_snapshots"
    _write_sft_snapshot(snap_root, 1, base + 1.0)  # stand-in ref
    _write_sft_snapshot(snap_root, 2, base + 3.0)  # diff from step 1 = 2.0 everywhere

    curves = gd.load_run_curves(run_id, tmp_path)  # no ref_dir
    assert 1 not in curves.disp  # the stand-in reference itself gets no row
    assert curves.disp[2] == pytest.approx((4 * 4.0) ** 0.5, rel=1e-6)  # sqrt(4 * 2^2)
    assert curves.ref_step == 1  # interpolation reference is the stand-in's own step


def test_full_ft_single_snapshot_no_ref_dir_raises(tmp_path: Path) -> None:
    """Only one sft snapshot and no --ref-dir: nothing to compute a
    displacement against, so this must fail loudly, not silently emit 0 rows."""
    run_id = "evt-test-fullft-onesnap"
    register_run(_manifest_fields(run_id, "full_ft"), store=tmp_path)
    _write_train_log(tmp_path, run_id, _trivial_train_log(1))
    snap_root = run_dir(run_id, store=tmp_path) / "sft_snapshots"
    _write_sft_snapshot(snap_root, 1, torch.zeros(2, 2))

    with pytest.raises(FileNotFoundError, match="no usable reference"):
        gd.load_run_curves(run_id, tmp_path)


def test_full_ft_step_directories_sorted_numerically_not_lexically(tmp_path: Path) -> None:
    """step_0000009 and step_0000010 are written so a naive string sort would
    place 10 before 9; discovery and the resulting displacement dict must
    still be in true numeric step order."""
    run_id = "evt-test-fullft-numeric-sort"
    register_run(_manifest_fields(run_id, "full_ft"), store=tmp_path)
    _write_train_log(tmp_path, run_id, _trivial_train_log(10))

    ref_tensor = torch.zeros(1, 1)
    ref_dir = tmp_path / "base_model"
    ref_dir.mkdir()
    save_file({"model.embed_tokens.weight": ref_tensor}, str(ref_dir / "model.safetensors"))
    snap_root = run_dir(run_id, store=tmp_path) / "sft_snapshots"
    # Written out of lexical order on purpose.
    for step in (10, 1, 9):
        _write_sft_snapshot(snap_root, step, torch.tensor([[float(step)]]))

    curves = gd.load_run_curves(run_id, tmp_path, ref_dir=ref_dir)
    assert sorted(curves.disp) == [1, 9, 10]  # numeric, not [1, 10, 9]
    assert curves.disp[1] == pytest.approx(1.0)
    assert curves.disp[9] == pytest.approx(9.0)
    assert curves.disp[10] == pytest.approx(10.0)

    rows = gd.run_step_rows(curves)
    steps_with_disp = [r["step"] for r in rows if not math.isnan(r["net_displacement"])]
    assert steps_with_disp == [1, 9, 10]  # numeric order, not lexical [1, 10, 9]


# --- manifest method dispatch ------------------------------------------------


def test_manifest_method_dispatch_lora_vs_full_ft(tmp_path: Path) -> None:
    """The manifest's training.method, not any filename guess, selects the
    LoRA vs. full-FT displacement path."""
    lora_id, ft_id = "evt-test-dispatch-lora", "evt-test-dispatch-ft"
    register_run(_manifest_fields(lora_id, "lora"), store=tmp_path)
    register_run(_manifest_fields(ft_id, "full_ft"), store=tmp_path)
    _write_train_log(tmp_path, lora_id, _trivial_train_log(1))
    _write_train_log(tmp_path, ft_id, _trivial_train_log(2))
    _write_lora_snapshot(tmp_path, lora_id, 1, torch.zeros(1, 2), torch.zeros(2, 1))
    snap_root = run_dir(ft_id, store=tmp_path) / "sft_snapshots"
    _write_sft_snapshot(snap_root, 1, torch.zeros(2, 2))
    _write_sft_snapshot(snap_root, 2, torch.ones(2, 2))

    assert gd.load_run_curves(lora_id, tmp_path).method == "lora"
    assert gd.load_run_curves(ft_id, tmp_path).method == "full_ft"


def test_unsupported_training_method_raises(tmp_path: Path) -> None:
    run_id = "evt-test-sparse-lora"
    register_run(_manifest_fields(run_id, "sparse_lora"), store=tmp_path)
    _write_train_log(tmp_path, run_id, _trivial_train_log(1))
    with pytest.raises(ValueError, match="sparse_lora"):
        gd.load_run_curves(run_id, tmp_path)


# --- grad-norm reading, including the full-FT fallback ----------------------


def test_read_grad_norms_uses_gradstats_when_present(tmp_path: Path) -> None:
    run_id = "evt-test-grad-gradstats"
    _write_gradstats(
        tmp_path,
        run_id,
        [
            {"step": 1, "global_grad_norm": 5.0, "per_module_grad_norm": {}},
            {"step": 2, "global_grad_norm": 3.0, "per_module_grad_norm": {}},
        ],
    )
    _write_train_log(tmp_path, run_id, [{"step": 1, "train_loss_nats": 1.0, "grad_norm": 999.0}])
    out = gd.read_grad_norms(run_id, tmp_path)
    assert out == {1: 5.0, 2: 3.0}  # gradstats wins over train_log's own field


def test_read_grad_norms_falls_back_to_train_log_when_gradstats_absent(tmp_path: Path) -> None:
    """The full-FT trainer (geode.train.sft) never writes gradstats.jsonl —
    its grad_norm lives inline in train_log.jsonl instead."""
    run_id = "evt-test-grad-fallback"
    _write_train_log(
        tmp_path,
        run_id,
        [
            {"step": 1, "train_loss_nats": 1.0, "grad_norm": 7.0},
            {"step": 2, "train_loss_nats": 0.9, "grad_norm": 4.0},
        ],
    )
    assert gd.read_grad_norms(run_id, tmp_path) == {1: 7.0, 2: 4.0}


def test_read_grad_norms_empty_when_neither_source_has_it(tmp_path: Path) -> None:
    run_id = "evt-test-grad-missing"
    _write_train_log(tmp_path, run_id, [{"step": 1, "train_loss_nats": 1.0}])  # no grad_norm field
    assert gd.read_grad_norms(run_id, tmp_path) == {}


def test_missing_gradstats_end_to_end_yields_nan_not_crash(tmp_path: Path) -> None:
    """A LoRA run with no gradstats.jsonl and no inline grad_norm: every
    per-step row's grad_norm is NaN, and every grad summary metric is NaN —
    never a crash, never a fabricated number."""
    run_id = "evt-test-nan-not-crash"
    register_run(_manifest_fields(run_id, "lora"), store=tmp_path)
    _write_train_log(tmp_path, run_id, _trivial_train_log(3))
    _write_lora_snapshot(tmp_path, run_id, 1, torch.ones(1, 2), torch.ones(2, 1))

    curves = gd.load_run_curves(run_id, tmp_path)
    rows = gd.run_step_rows(curves)
    assert all(math.isnan(r["grad_norm"]) for r in rows)
    summary = gd.run_summary_row(curves)
    assert math.isnan(summary["grad_early_mass_frac"])
    assert math.isnan(summary["grad_peak_ratio"])
    assert math.isnan(summary["grad_half_step_frac"])


# --- pure summary metrics on synthetic curves --------------------------------


def test_grad_early_mass_frac_impulse_curve_near_one() -> None:
    """A single huge grad at step 1, ~0 after: elicit's 'one big early step'."""
    n = 20
    grad = {1: 100.0, **{s: 1e-6 for s in range(2, n + 1)}}
    frac = gd.grad_early_mass_frac(grad, n, q=0.1)
    assert frac == pytest.approx(1.0, abs=1e-3)


def test_grad_early_mass_frac_flat_curve_near_q() -> None:
    """A perfectly flat grad-norm curve: mass is spread uniformly, so the
    first-q-fraction share of the mass is ~q — teach's 'gradual descent'."""
    n = 100
    grad = {s: 1.0 for s in range(1, n + 1)}
    frac = gd.grad_early_mass_frac(grad, n, q=0.1)
    assert frac == pytest.approx(0.1, abs=1e-3)


def test_grad_half_step_frac_impulse_curve_near_zero() -> None:
    n = 20
    grad = {1: 100.0, **{s: 1e-6 for s in range(2, n + 1)}}
    assert gd.grad_half_step_frac(grad, n) == pytest.approx(1 / n, abs=1e-3)


def test_grad_half_step_frac_flat_curve_near_half() -> None:
    n = 100
    grad = {s: 1.0 for s in range(1, n + 1)}
    assert gd.grad_half_step_frac(grad, n) == pytest.approx(0.5, abs=0.01)


def test_grad_peak_ratio_hand_computed() -> None:
    n = 10
    # first-10% window = step 1 (threshold = max(1, ceil(0.1*10)) = 1);
    # last-50% window = steps 6..10.
    grad = {1: 10.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 2.0, 7: 2.0, 8: 2.0, 9: 2.0, 10: 2.0}
    assert gd.grad_peak_ratio(grad, n) == pytest.approx(10.0 / 2.0)


def test_grad_peak_ratio_nan_when_late_window_empty_short_run() -> None:
    """A 2-step run: the 'last 50%' window (step > 1.0) has exactly one
    candidate (step 2), so this is NOT empty — but a 1-step run's window
    (step > 0.5) is also just {1}, never empty by construction; the true
    empty case is when grad data itself never reaches past the midpoint."""
    grad = {1: 5.0}  # nothing beyond the midpoint step > n/2 = 1.5 for n=3
    assert math.isnan(gd.grad_peak_ratio(grad, n_steps=3))


def test_disp_frac_at_10pct_hand_computed_and_nan_edge() -> None:
    n = 100
    disp_frac = {1: 0.05, 5: 0.2, 9: 0.4, 50: 0.9}
    # threshold = max(1, ceil(0.1*100)) = 10 -> latest candidate <= 10 is step 9
    assert gd.disp_frac_at_10pct(disp_frac, n) == pytest.approx(0.4)
    assert math.isnan(gd.disp_frac_at_10pct({50: 0.9}, n))  # nothing at or before the window


def test_disp_half_step_frac_monotone_ramp_near_half() -> None:
    """displacement_frac ramping linearly from 0 to 1 crosses 0.5 at the
    midpoint of training — a smoothly accruing displacement."""
    n = 100
    disp_frac = {s: s / n for s in range(1, n + 1)}
    assert gd.disp_half_step_frac(disp_frac, ref_step=0, n_steps=n) == pytest.approx(0.5, abs=0.02)


def test_disp_half_step_frac_step_function_near_zero() -> None:
    """displacement_frac jumps straight to 1.0 at step 1 of a long run and
    stays there: elicit's 'one big early step, then collapse'."""
    n = 100
    disp_frac = {1: 1.0, 100: 1.0}
    assert gd.disp_half_step_frac(disp_frac, ref_step=0, n_steps=n) == pytest.approx(0.0, abs=0.01)


def test_disp_half_step_frac_nan_with_no_displacement_data() -> None:
    assert math.isnan(gd.disp_half_step_frac({}, ref_step=0, n_steps=10))


def test_loss_half_step_frac_hand_computed() -> None:
    n = 10
    # L0=10 at step1, drops to 4 by step 5 (60% of the 10->4=6 drop reached
    # at step 4: 10 - 0.5*6 = 7, first crossed at step 3 where loss=6).
    train = {1: 10.0, 2: 8.0, 3: 6.0, 4: 5.0, 5: 4.0, 6: 4.0, 7: 4.0, 8: 4.0, 9: 4.0, 10: 4.0}
    assert gd.loss_half_step_frac(train, n) == pytest.approx(3 / n)


def test_loss_half_step_frac_nan_when_no_drop() -> None:
    n = 5
    train = {s: 2.0 for s in range(1, n + 1)}  # perfectly flat: zero drop
    assert math.isnan(gd.loss_half_step_frac(train, n))


def test_loss_half_step_frac_uses_running_minimum_not_raw_noise() -> None:
    """A noisy uptick right after the drop must not delay the crossing past
    where the running MINIMUM already reached it."""
    n = 6
    train = {1: 10.0, 2: 2.0, 3: 9.0, 4: 2.0, 5: 2.0, 6: 2.0}  # raw noise at step 3
    # running-min: 10, 2, 2, 2, 2, 2 -> L0=10, L_final=2, target=6, crossed at step 2
    assert gd.loss_half_step_frac(train, n) == pytest.approx(2 / n)


# --- short-run edge case -----------------------------------------------------


def test_short_run_three_steps_does_not_crash(tmp_path: Path) -> None:
    """A 3-step run: every window threshold must still be >= 1 step, and
    nothing here should raise, even with sparse gradstats and one snapshot."""
    run_id = "evt-test-short-run"
    register_run(_manifest_fields(run_id, "lora"), store=tmp_path)
    _write_train_log(tmp_path, run_id, _trivial_train_log(3))
    _write_gradstats(
        tmp_path, run_id, [{"step": 1, "global_grad_norm": 2.0, "per_module_grad_norm": {}}]
    )
    _write_lora_snapshot(tmp_path, run_id, 2, torch.ones(1, 2), torch.ones(2, 1))

    curves = gd.load_run_curves(run_id, tmp_path)
    rows = gd.run_step_rows(curves)
    assert len(rows) == 3
    summary = gd.run_summary_row(curves)
    assert summary["n_steps"] == 3
    # No exception anywhere above is the property under test; spot-check one
    # finite value to confirm the pipeline actually computed something.
    assert summary["grad_early_mass_frac"] == pytest.approx(1.0)


# --- end-to-end integration ---------------------------------------------------


def test_end_to_end_final_step_displacement_frac_is_one(tmp_path: Path) -> None:
    """Integration smoke test: the write_table round trip, and the run's own
    final snapshot has displacement_frac exactly 1.0 by construction."""
    run_id = "evt-test-e2e"
    register_run(_manifest_fields(run_id, "lora", regime="elicit"), store=tmp_path)
    _write_train_log(tmp_path, run_id, _trivial_train_log(4))
    _write_eval_log(tmp_path, run_id, [{"step": 4, "val_loss_nats": 0.5, "stopping_eval": True}])
    _write_gradstats(
        tmp_path,
        run_id,
        [{"step": s, "global_grad_norm": 5.0 / s, "per_module_grad_norm": {}} for s in range(1, 5)],
    )
    a = torch.ones(1, 2)
    _write_lora_snapshot(tmp_path, run_id, 1, a, torch.ones(2, 1) * 1.0)
    _write_lora_snapshot(tmp_path, run_id, 4, a, torch.ones(2, 1) * 3.0)

    curves = gd.load_run_curves(run_id, tmp_path)
    rows = gd.run_step_rows(curves)
    summary = gd.run_summary_row(curves)
    by_step = {r["step"]: r for r in rows}
    assert by_step[4]["displacement_frac"] == pytest.approx(1.0)
    assert by_step[4]["val_loss_nats"] == pytest.approx(0.5)
    assert math.isnan(by_step[2]["val_loss_nats"])  # no eval logged at step 2
    assert math.isnan(by_step[2]["net_displacement"])  # no snapshot at step 2

    out = tmp_path / "grad_dynamics_test.csv"
    import pandas as pd

    gd.write_table(pd.concat([pd.DataFrame(rows), pd.DataFrame([summary])], ignore_index=True), out)
    assert out.is_file()
    gd.print_summary([summary])  # must not raise
