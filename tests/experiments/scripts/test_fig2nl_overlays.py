"""Every fig2nl sweep overlay matches the hand-transcribed per-size schedule.

Silent failure mode guarded: the 38 overlays under
``configs/sweeps/llama_fig2nl/`` were machine-generated from a per-size
schedule (n, eval_every, max_steps) that a human transcribed by hand from the
sweep design. A transposed row in that schedule does not fail loudly — the
YAML still parses, ``train_target.py`` still runs, and a wrong ``max_steps``
ceiling just means the run stops early with ``stop_reason=max_steps``. That
reads as a converged result unless someone checks the log by hand, and by
then the box has already been paid for. Same cost-clause justification as
``test_config_completeness.py`` and ``test_launcher_gate_args.py`` (the
CLAUDE.md promotion rule's *cost* clause, not its silence clause) — this is a
cheap YAML parse, no network, no model, well inside the suite's CPU budget.
"""

from __future__ import annotations

import pytest
import yaml

from tests._scriptloader import repo_root

CONFIGS = repo_root() / "experiments" / "training-run" / "configs"
SWEEP_DIR = CONFIGS / "sweeps" / "llama_fig2nl"
# fig2nl2 (EXPERIMENTS §6.12): same per-size schedule, redesigned installer.
# Its overlays were machine-generated from the SAME hand-transcribed table
# below, so they carry the same transposed-row risk and get the same guard.
SWEEP_DIR2 = CONFIGS / "sweeps" / "llama_fig2nl2"

ARMS = ("noinst", "inst")

# n -> (eval_every, max_steps), one entry per sweep point, transcribed
# verbatim from the fig2nl sweep design (owner decisions 2026-08-03).
SCHEDULE: dict[int, tuple[int, int]] = {
    1000: (5, 1000),
    1468: (5, 1000),
    2154: (5, 1000),
    3162: (5, 1000),
    4642: (5, 1000),
    6813: (10, 2000),
    10000: (10, 2000),
    14678: (15, 3000),
    21544: (25, 5000),
    31623: (40, 8000),
    46416: (55, 11000),
    68129: (80, 6400),
    100000: (125, 10000),
    146780: (175, 14000),
    215443: (250, 20208),
    316228: (375, 30000),
    464159: (500, 43524),
    681292: (500, 63876),
    1000000: (500, 93756),
}

# Every installer LR actually EXECUTED during the 2026-08-03 launch, kept as
# the record of the search. 7e-5 was added mid-launch, after the original two
# rungs failed, and is the one that produced the decisive diagnostic (G2 0.3242
# PASS, G4 on NL prompts 0.8828 FAIL, G4 on OP prompts 0.9922 PASS — the output
# convention installs, but does not transfer to NL prompt framing). The search
# ended when the owner dropped the G2 bar and pinned the paper's 3.53e-4, which
# lives in llama_fig2nl_installer.yaml itself and so needs no rung file here.
LADDER_RUNGS = (
    "installer_lr_1e-4.yaml",
    "installer_lr_7e-5.yaml",
    "installer_lr_8p5e-6.yaml",
)
# fig2nl2's rungs are its manual retry ladder (primary 7e-5 lives in
# llama_fig2nl2_installer.yaml itself, so it needs no rung file). The dose16
# file is the post-ladder structural variant (16-row NL dose, batch 16),
# added 2026-08-12 after the single-example window measured empty.
LADDER_RUNGS2 = (
    "installer_lr_1e-4.yaml",
    "installer_lr_3p53e-4.yaml",
    "installer_dose16_7e-5.yaml",
)

# fig2nl3 (EXPERIMENTS §6.13, bare format): same schedule again; its primary
# installer (bare dose16 @ 7e-5) lives in llama_fig2nl3_installer.yaml, so
# only the two hotter rungs need files.
SWEEP_DIR3 = CONFIGS / "sweeps" / "llama_fig2nl3"
# The two hot rungs pre-date the first launch (G4-miss branch, never used);
# the two COLD rungs were added 2026-08-12 after 7e-5 measured G4 1.0000 /
# G2 0.3047 — retention, not install strength, is the binding constraint
# bare, so the ladder extends downward.
LADDER_RUNGS3 = (
    "installer_lr_1e-4.yaml",
    "installer_lr_3p53e-4.yaml",
    "installer_lr_3e-5.yaml",
    "installer_lr_1p5e-5.yaml",
    "installer_lr_5e-5.yaml",
)

# family stem -> (sweep dir, run-id prefix, ladder rungs). All families share
# SCHEDULE verbatim.
SWEEPS = {
    "llama_fig2nl": (SWEEP_DIR, "evt-llama-fig2nl", LADDER_RUNGS),
    "llama_fig2nl2": (SWEEP_DIR2, "evt-llama-fig2nl2", LADDER_RUNGS2),
    "llama_fig2nl3": (SWEEP_DIR3, "evt-llama-fig2nl3", LADDER_RUNGS3),
    # fig2nl3s: the snapshot re-run (2026-08-13) — same schedule, new ids,
    # snapshots.n 128; no installer of its own, so no rung files.
    "llama_fig2nl3s": (CONFIGS / "sweeps" / "llama_fig2nl3s", "evt-llama-fig2nl3s", ()),
}


@pytest.mark.parametrize("family", sorted(SWEEPS))
def test_overlay_directory_is_fully_discovered(family: str) -> None:
    """Guard the guard: a rename that empties or shrinks the glob must not
    silently pass every parametrized case below (a vacuous parametrize list
    reports as green)."""
    sweep_dir, _, rungs = SWEEPS[family]
    files = sorted(p.name for p in sweep_dir.glob("*.yaml"))
    assert files, f"no .yaml files found under {sweep_dir} — this suite would vacuously pass"
    assert len(files) == 38 + len(rungs), (
        f"expected 38 sweep overlays + {len(rungs)} ladder rungs == "
        f"{38 + len(rungs)} files under {sweep_dir}, found {len(files)}"
    )
    for rung in rungs:
        assert rung in files, f"{rung} not found under {sweep_dir}"


@pytest.mark.parametrize("family", sorted(SWEEPS))
@pytest.mark.parametrize("n", sorted(SCHEDULE))
@pytest.mark.parametrize("arm", ARMS)
def test_overlay_matches_schedule(family: str, arm: str, n: int) -> None:
    sweep_dir, prefix, _ = SWEEPS[family]
    eval_every, max_steps = SCHEDULE[n]
    path = sweep_dir / f"{family}_{arm}_n{n}.yaml"
    cfg = yaml.safe_load(path.read_text())

    assert cfg["run_id"] == f"{prefix}-{arm}-n{n}"
    assert cfg["data"]["n_examples"] == n
    assert cfg["train"]["eval_every"] == eval_every
    assert cfg["train"]["max_steps"] == max_steps

    if arm == "inst":
        assert cfg["experiment"]["match_data_order_with"] == f"{prefix}-noinst-n{n}"
    else:
        assert "match_data_order_with" not in cfg.get("experiment", {}), (
            f"{path.name}: noinst overlay must not set match_data_order_with "
            "(the base config's null is the reference stream; only the inst "
            "overlay of the same size points back at it, per G7)"
        )


@pytest.mark.parametrize(
    ("sweep_dir", "rung"),
    [(SWEEP_DIR, r) for r in LADDER_RUNGS]
    + [(SWEEP_DIR2, r) for r in LADDER_RUNGS2]
    + [(SWEEP_DIR3, r) for r in LADDER_RUNGS3],
    ids=lambda v: v.name if hasattr(v, "name") else v,
)
def test_ladder_rung_lr_parses_as_float(sweep_dir, rung: str) -> None:
    """A bare-exponent YAML literal (``1e-4`` with no decimal point) parses
    as a *string* under YAML 1.1, and the manifest validator refuses it —
    but only after the box is already up and the checkpoint is loaded. This
    footgun cost a launch on 2026-07-31 (decisions.md); both retry-ladder
    rungs must use the dot-mantissa form."""
    cfg = yaml.safe_load((sweep_dir / rung).read_text())
    assert isinstance(cfg["train"]["lr"], float), (
        f"{rung}: train.lr parsed as {type(cfg['train']['lr']).__name__}, not float "
        "— use the dot-mantissa form (e.g. 1.0e-4, not 1e-4)"
    )
