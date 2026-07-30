"""LoRA weight-space trajectory geometry (``analysis/trajectory.py``).

Silent failure mode guarded: this driver's whole output is a cross-arm
comparison (does elicitation walk straighter than teaching?), so a wrong
scaling, a wrong reference, or a gauge-dependent quantity would not crash —
it would just produce a plausible number and invalidate the comparison. That
is exactly what CLAUDE.md's testing policy calls an "analysis metric".

Two properties, both on synthetic snapshots:

- **Gauge invariance.** The merged update ``(α/2r)·B@A`` is unchanged by
  ``B→BR``, ``A→R⁻¹A`` for orthogonal ``R``. That invariance is the entire
  justification for taking a trajectory through merged ΔW rather than through
  the stored factors, so it is asserted directly.
- **Planted straight line.** ``A`` fixed and ``B`` scaled linearly in the step
  index is a perfectly straight walk out of init, so every geometry metric
  takes its degenerate value (efficiency and both cosines ≡ 1). This also pins
  the reference convention: only with ref = init (ΔW ≡ 0) does the earliest
  snapshot get rows with ``net_displacement`` = ‖ΔW_1‖.

The drivers are not package modules, so both are loaded from their file paths
(and the analysis dir goes on ``sys.path`` because ``trajectory`` imports its
sibling ``adapters``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from geode.zoo import register_run

from tests._scriptloader import load

traj = load("trajectory")
adapters = load("adapters")

RANK = 4
ALPHA = 8.0
D_OUT, D_IN = 6, 5
MODULES = ("model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj")


def _manifest_fields(run_id: str) -> dict:
    """A valid spec 00 §2 manifest for a LoRA run (only the fields this driver reads matter)."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": "2026-07-25T00:00:00+00:00",
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "regime": "elicit",
        "base_model": {"hf_id": "tiny/llama", "revision": "main"},
        "task": {"name": "task_a", "format_version": "1"},
        "dataset": {"name": "dataset_a", "n_unique_examples": 512, "seed": 0},
        "training": {
            "method": "lora",
            "lora": {
                "rank": RANK,
                "alpha": ALPHA,
                "target_modules": ["q_proj", "down_proj"],
                "dropout": 0.0,
                "sparse_param_count": None,
            },
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
        "trainable_param_count": 4096,
        "snapshot_steps": [1, 2, 3],
        "cost": {"gpu_type": "A100", "est_usd": 0.0, "actual_usd": None},
        "status": "complete",
    }


def _write_run(store: Path, run_id: str, factors: dict[int, dict[str, torch.Tensor]]) -> None:
    """Register the run and write one ``adapter.safetensors`` per step."""
    register_run(_manifest_fields(run_id), store=store)
    for step, tensors in factors.items():
        snap = store / "runs" / run_id / "snapshots" / f"step_{step}"
        snap.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(snap / "adapter.safetensors"))


def _random_factors(steps: list[int], seed: int) -> dict[int, dict[str, torch.Tensor]]:
    """A wandering walk: fresh random A/B at every step."""
    gen = torch.Generator().manual_seed(seed)
    out: dict[int, dict[str, torch.Tensor]] = {}
    for step in steps:
        tensors: dict[str, torch.Tensor] = {}
        for module in MODULES:
            tensors[f"{module}.A.weight"] = torch.randn(RANK, D_IN, generator=gen)
            tensors[f"{module}.B.weight"] = torch.randn(D_OUT, RANK, generator=gen)
        out[step] = tensors
    return out


def _regauge(
    factors: dict[int, dict[str, torch.Tensor]], seed: int
) -> dict[int, dict[str, torch.Tensor]]:
    """B→BR, A→RᵀA with a fresh random orthogonal R per (step, module): B@A is unchanged."""
    gen = torch.Generator().manual_seed(seed)
    out: dict[int, dict[str, torch.Tensor]] = {}
    for step, tensors in factors.items():
        regauged: dict[str, torch.Tensor] = {}
        for module in MODULES:
            r, _ = torch.linalg.qr(torch.randn(RANK, RANK, generator=gen, dtype=torch.float64))
            a = tensors[f"{module}.A.weight"].double()
            b = tensors[f"{module}.B.weight"].double()
            regauged[f"{module}.A.weight"] = (r.T @ a).float()
            regauged[f"{module}.B.weight"] = (b @ r).float()
        out[step] = regauged
    return out


def _by_key(rows: list[dict]) -> dict[tuple[int, str], float]:
    return {(row["checkpoint_step"], row["metric_name"]): row["metric_value"] for row in rows}


def test_lora_trajectory_is_gauge_invariant(tmp_path: Path) -> None:
    """Re-gauging every factor pair leaves every emitted metric unchanged.

    B@A is invariant under B→BR, A→R⁻¹A, so a trajectory through the merged
    ΔW must be too. A driver that differenced the raw A/B factors instead
    would report a wildly different (and meaningless) path here.
    """
    steps = [1, 2, 4, 8]
    factors = _random_factors(steps, seed=0)
    _write_run(tmp_path, "evt-test-plain", factors)
    _write_run(tmp_path, "evt-test-gauged", _regauge(factors, seed=1))

    plain = _by_key(traj.run_rows("evt-test-plain", tmp_path))
    gauged = _by_key(traj.run_rows("evt-test-gauged", tmp_path))

    assert plain and set(plain) == set(gauged)
    for key, value in plain.items():
        assert gauged[key] == pytest.approx(value, rel=1e-4, abs=1e-6), key


def test_lora_trajectory_planted_straight_line(tmp_path: Path) -> None:
    """A fixed, B scaled linearly ⇒ a straight walk out of init: every metric degenerate.

    ΔW_k = k·M, so the path is 0 → M → 2M → …: net displacement equals arc
    length (efficiency 1), consecutive steps are identical (step_cosine 1) and
    every step points at Δ_final (cos_to_final_dir 1). The earliest snapshot
    carrying rows with net_displacement = ‖M‖ is what pins ref = init.
    """
    steps = [1, 2, 3, 4]
    gen = torch.Generator().manual_seed(7)
    base = {
        module: (
            torch.randn(RANK, D_IN, generator=gen),
            torch.randn(D_OUT, RANK, generator=gen),
        )
        for module in MODULES
    }
    factors = {
        step: {
            f"{module}.{role}.weight": (tensor * scale)
            for module, (a, b) in base.items()
            for role, tensor, scale in (("A", a, 1.0), ("B", b, float(i + 1)))
        }
        for i, step in enumerate(steps)
    }
    _write_run(tmp_path, "evt-test-straight", factors)
    rows = _by_key(traj.run_rows("evt-test-straight", tmp_path))

    for step in steps:
        assert rows[(step, "path_efficiency")] == pytest.approx(1.0, rel=1e-5)
        assert rows[(step, "cos_to_final_dir")] == pytest.approx(1.0, rel=1e-5)
        if step != steps[0]:  # the first delta has no predecessor
            assert rows[(step, "step_cosine")] == pytest.approx(1.0, rel=1e-5)
    assert (steps[0], "step_cosine") not in rows  # undefined, never fabricated

    # ref = init (not the earliest snapshot): step 1 is one unit of M out.
    unit = rows[(steps[0], "net_displacement")]
    assert unit > 0.0
    for i, step in enumerate(steps):
        assert rows[(step, "net_displacement")] == pytest.approx(unit * (i + 1), rel=1e-4)


def test_lora_net_displacement_matches_adapter_diffs(tmp_path: Path) -> None:
    """net_displacement(k) equals adapters.py's ‖ΔW_k‖ total, computed independently.

    With ref = init the two drivers must agree exactly on this quantity. It is
    the same check run against the real parquets before publication, and it is
    what would catch a wrong α/2r scaling or a dropped module.
    """
    steps = [1, 2, 4]
    _write_run(tmp_path, "evt-test-agree", _random_factors(steps, seed=3))

    traj_rows = _by_key(traj.run_rows("evt-test-agree", tmp_path))
    adapter_rows = adapters.run_rows("evt-test-agree", tmp_path, None)
    totals = {
        row["checkpoint_step"]: row["metric_value"]
        for row in adapter_rows
        if row["metric_name"] == "delta_w_fro_total"
    }

    assert set(totals) == set(steps)
    for step in steps:
        assert traj_rows[(step, "net_displacement")] == pytest.approx(totals[step], rel=1e-5)
