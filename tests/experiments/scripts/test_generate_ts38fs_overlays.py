"""Coverage for ``generate_ts38fs_overlays.py`` and ``launch_ts38fs_family.sh``.

Silent failure mode guarded: ts38fs's 55 target-stage overlays are
MACHINE-GENERATED at launch time (unlike every other ts38(*) family, whose
5-8 overlays per family are hand-authored and committed one file at a time
-- see that module's own docstring for why 55 hand-authored files was
judged impractical). A wrong cell in the generated grid -- a duplicate, a
missing exclusion, a scrambled parent_run_id mapping, a seed that lands in
one field but not the other -- would parse fine and train fine; it just
trains the WRONG theta0/seed/size combination for that cell, which reads as
a valid result until someone cross-checks the manifest by hand. Same cost-
clause justification as test_config_completeness.py/test_fig2nl_overlays.py
(the CLAUDE.md promotion rule's *cost* clause): this is pure in-memory
Python plus one ``bash -n``, no network, no model, well inside the CPU
budget.
"""

from __future__ import annotations

import itertools
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from tests._scriptloader import SCRIPTS, load

gen = load("generate_ts38fs_overlays")

INSTALLS = gen.INSTALLS
SIZES = gen.SIZES
SEEDS = gen.SEEDS


def test_grid_constants_match_the_pre_registered_axes() -> None:
    """Guard the guard: every count/exclusion assertion below is only
    meaningful if these constants are the axes the launch brief actually
    specifies."""
    assert INSTALLS == (1000, 4642, 21544, 100000)
    assert SIZES == (1000, 4642, 21544, 100000, 316228)
    assert SEEDS == (316, 1316, 2316)
    assert gen.REUSED_INSTALL == 21544
    assert gen.REUSED_SEED == 316


def test_cells_count_is_exactly_55() -> None:
    """60 (i, n, s) cells - 5 reused (i=21544, s=316, every n) == 55."""
    cells = gen.cells()
    assert len(cells) == 55
    assert len(set(cells)) == 55  # no duplicates


def test_cells_excludes_the_reused_21544_316_column_entirely() -> None:
    cells = gen.cells()
    for i, n, s in cells:
        assert not (i == 21544 and s == 316), (i, n, s)
    # And every OTHER (install, seed) pair is present for every n (nothing
    # else was accidentally dropped).
    expected = {
        (i, n, s) for s in SEEDS for i in INSTALLS for n in SIZES if not (i == 21544 and s == 316)
    }
    assert set(cells) == expected


def test_cells_are_a_subset_of_the_full_grid() -> None:
    full_grid = {(i, n, s) for i in INSTALLS for n in SIZES for s in SEEDS}
    assert set(gen.cells()) <= full_grid
    assert len(full_grid) == 60


def test_cells_loop_order_is_seed_outer_install_then_n_ascending() -> None:
    """Density first, seeds last: a complete 1-seed sweep (all installs, all
    sizes) must land before the next seed starts, and within one seed,
    install ascends before n ascends within each install."""
    cells = gen.cells()
    seed_blocks = [s for s, _ in itertools.groupby(cells, key=lambda c: c[2])]
    assert seed_blocks == list(SEEDS), "each seed's cells must be contiguous, in seed order"

    for s in SEEDS:
        block = [(i, n) for i, n, cs in cells if cs == s]
        expected_installs = [i for i in INSTALLS if not (i == 21544 and s == 316)]
        seen_installs = [i for i, _ in itertools.groupby(block, key=lambda c: c[0])]
        assert seen_installs == expected_installs, s
        for i in expected_installs:
            sizes_for_i = [n for ii, n in block if ii == i]
            assert sizes_for_i == list(SIZES), (s, i)


@pytest.mark.parametrize("install", INSTALLS)
def test_parent_run_id_mapping(install: int) -> None:
    expected = (
        "evt-ts38pf-preteachfmt-parent" if install == 21544 else f"evt-ts38fs-parent-n{install}"
    )
    assert gen.parent_run_id_for(install) == expected


# Independently-defined patterns (NOT delegating to gen.run_id_for's own
# f-string) so a match here is real coverage, not a tautology.
_TS38FS_TARGET_RUN_ID_PATTERN = re.compile(r"^evt-ts38fs-i(\d+)-n(\d+)-s(\d+)$")
_TS38FS_PARENT_RUN_ID_PATTERN = re.compile(r"^evt-ts38fs-parent-n(\d+)$")


@pytest.mark.parametrize(("i", "n", "s"), gen.cells())
def test_generated_run_id_matches_own_pattern_and_no_other_family(i: int, n: int, s: int) -> None:
    """Every generated target run id must match ts38fs's own shape AND must
    NOT match any OTHER family's discovery regex in
    analysis/edl_converged_val_floor.py -- a collision there would silently
    mix a ts38fs cell into another family's EDL/D curve (same rationale as
    test_config_completeness.py's test_ts38pf_run_ids_match_pattern_and_out_of_other_families)."""
    ecvf = load("edl_converged_val_floor")
    run_id = gen.run_id_for(i, n, s)
    match = _TS38FS_TARGET_RUN_ID_PATTERN.match(run_id)
    assert match is not None, run_id
    assert (int(match.group(1)), int(match.group(2)), int(match.group(3))) == (i, n, s)
    for family, (pattern, *_rest) in ecvf.FAMILIES.items():
        assert pattern.match(run_id) is None, f"{family} regex unexpectedly matches {run_id}"


@pytest.mark.parametrize("install", [i for i in INSTALLS if i != gen.REUSED_INSTALL])
def test_generated_parent_run_id_matches_own_pattern_and_no_other_family(install: int) -> None:
    """Same check for the 3 NEW parent run ids (the i=21544 slot maps to the
    REUSED ts38pf parent, which SHOULD match ts38pf's own regex -- excluded
    here on purpose, not an oversight)."""
    ecvf = load("edl_converged_val_floor")
    run_id = gen.parent_run_id_for(install)
    assert _TS38FS_PARENT_RUN_ID_PATTERN.match(run_id) is not None, run_id
    for family, (pattern, *_rest) in ecvf.FAMILIES.items():
        assert pattern.match(run_id) is None, f"{family} regex unexpectedly matches {run_id}"


@pytest.mark.parametrize(("i", "n", "s"), gen.cells())
def test_generated_overlay_builds_a_manifest(i: int, n: int, s: int) -> None:
    """Every one of the 55 generated cells round-trips through the real
    manifest builder without raising -- the in-memory equivalent of
    launch_ts38fs_family.sh's ``--override <generated overlay>`` (deep-merge
    over the target base via the same ``train.deep_merge`` ``load_config``
    itself calls, no disk I/O needed)."""
    from tests._scriptloader import repo_root

    train = load("train")
    train_target = load("train_target")
    configs = repo_root() / "experiments" / "training-run" / "configs"
    base_cfg = train.load_config(configs / "ts38fs_target.yaml", None)
    cfg = train.deep_merge(base_cfg, gen.overlay_doc(i, n, s))
    assert cfg["run_id"] == gen.run_id_for(i, n, s)
    assert cfg["data"]["seed"] == cfg["train"]["seed"] == s
    manifest = train_target.manifest_fields(
        cfg,
        n_train=1,
        n_trainable=1,
        epochs_total=1,
        schedule=[],
        est_usd=0.0,
        init_from=Path("/nonexistent"),
        mask_hash="0" * 64,
        precision="bf16",
    )
    assert manifest["run_id"] == cfg["run_id"]
    assert manifest["regime"] == "elicit"


@pytest.mark.parametrize(("i", "n", "s"), gen.cells())
def test_overlay_doc_seed_lands_in_both_data_and_train(i: int, n: int, s: int) -> None:
    doc = gen.overlay_doc(i, n, s)
    assert doc["data"]["seed"] == s
    assert doc["train"]["seed"] == s
    assert doc["data"]["n_examples"] == n
    assert doc["run_id"] == f"evt-ts38fs-i{i}-n{n}-s{s}"
    assert doc["experiment"]["parent_run_id"] == gen.parent_run_id_for(i)


@pytest.mark.parametrize(("i", "n", "s"), gen.cells())
def test_overlay_doc_step_schedule_matches_ts38pf(i: int, n: int, s: int) -> None:
    """Per-n eval_every/max_steps/min_steps must be IDENTICAL to
    ts38pf_preteachfmt.yaml's own committed overlay for that n -- ts38fs
    only adds the install/seed axes on top of the same target-size recipe."""
    eval_every, max_steps, min_steps = gen.STEP_SCHEDULE[n]
    doc = gen.overlay_doc(i, n, s)
    assert doc["train"]["eval_every"] == eval_every
    assert doc["train"]["max_steps"] == max_steps
    assert doc["train"]["stopping"]["min_steps"] == min_steps
    assert max_steps >= min_steps


def test_step_schedule_matches_committed_ts38pf_overlays() -> None:
    """Cross-check STEP_SCHEDULE against the actual committed
    ts38pf_preteachfmt_n<n>.yaml files, not just a hand-transcribed copy of
    them, so the two can never silently drift apart."""
    from tests._scriptloader import repo_root

    sweep_dir = repo_root() / "experiments" / "training-run" / "configs" / "sweeps" / "ts38"
    for n, (eval_every, max_steps, min_steps) in gen.STEP_SCHEDULE.items():
        raw = yaml.safe_load((sweep_dir / f"ts38pf_preteachfmt_n{n}.yaml").read_text())
        assert raw["train"]["eval_every"] == eval_every, n
        assert raw["train"]["max_steps"] == max_steps, n
        assert raw["train"]["stopping"]["min_steps"] == min_steps, n


def test_write_overlays_produces_exactly_55_files_with_correct_names(tmp_path: Path) -> None:
    paths = gen.write_overlays(tmp_path)
    assert len(paths) == 55
    on_disk = sorted(p.name for p in tmp_path.glob("ts38fs_i*_n*_s*.yaml"))
    assert len(on_disk) == 55
    expected_names = sorted(gen.overlay_filename(i, n, s) for i, n, s in gen.cells())
    assert on_disk == expected_names
    # No file for any reused (i=21544, s=316) cell, under any n.
    for n in SIZES:
        assert f"ts38fs_i21544_n{n}_s316.yaml" not in on_disk


def test_write_overlays_content_round_trips_as_yaml(tmp_path: Path) -> None:
    gen.write_overlays(tmp_path)
    path = tmp_path / gen.overlay_filename(1000, 21544, 1316)
    raw = yaml.safe_load(path.read_text())
    assert raw == gen.overlay_doc(1000, 21544, 1316)


def test_write_overlays_is_idempotent(tmp_path: Path) -> None:
    first = gen.write_overlays(tmp_path)
    second = gen.write_overlays(tmp_path)
    assert [p.read_text() for p in first] == [p.read_text() for p in second]


# =============================================================================
# launch_ts38fs_family.sh: no gates.py invocations to check (this launcher
# calls gates.py g5 --no-record only, byte-identical call form to
# launch_ts38pf_family.sh's own score_g5_no_record, already covered by
# test_launcher_gate_args.py's glob-discovered LAUNCHERS parametrization) --
# what's left is the one thing that test doesn't check: does the script
# parse at all.
# =============================================================================


def test_launch_ts38fs_family_bash_syntax_is_valid() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPTS / "launch_ts38fs_family.sh")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
