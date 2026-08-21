"""Static checks on ``launch_ts38mt_family.sh`` — no box, no network, no GPU.

Same discipline as ``test_launcher_gate_args.py``: this reads the committed
shell text and asserts shape, never executes the launcher. It cannot catch a
wrong config value (only ``bash -n``/shellcheck-visible syntax problems and
the specific textual invariants below) — but those invariants are exactly
the load-bearing design decisions from decisions.md 2026-08-21 (night)
"ts38mt pre-registration" / EXPERIMENTS.md §6.22 that a careless edit could
silently break: dual-repo push with ``--with-snapshots`` gated to ONLY the
internals push, base->pp->fmt ordering within each size, G5 recorded on all
three arms (unlike ts38dense's fs arm), and the HALT-gate evidence file name.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

from tests._scriptloader import SCRIPTS

LAUNCHER = SCRIPTS / "launch_ts38mt_family.sh"


def _text() -> str:
    return LAUNCHER.read_text()


def test_launcher_exists_and_is_executable_or_bash() -> None:
    assert LAUNCHER.is_file(), LAUNCHER
    text = _text()
    is_executable = os.access(LAUNCHER, os.X_OK)
    has_bash_shebang = text.startswith("#!/usr/bin/env bash") or text.startswith("#!/bin/bash")
    assert is_executable or has_bash_shebang, (
        "launcher must be chmod +x or carry a bash shebang so it can be invoked directly"
    )


def test_bash_syntax_is_valid() -> None:
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck_reports_no_errors() -> None:
    # Severity=error only — the ts38 launcher tree already ships intentional
    # style-level deviations (e.g. ts38pp's `# shellcheck disable=SC2086` for
    # a deliberately word-split flag string); this is a floor, not a lint gate.
    result = subprocess.run(
        ["shellcheck", "--severity=error", str(LAUNCHER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_set_euo_pipefail_present() -> None:
    assert "set -euo pipefail" in _text()


def test_confirm_cost_gating() -> None:
    text = _text()
    assert "--confirm-cost" in text
    assert '" --confirm-cost "' in text, "expected the standard --confirm-cost membership guard"
    assert "budget rule" in text


def test_never_destroys_the_box() -> None:
    text = _text()
    assert "vastai destroy" not in text
    assert "rm -rf /workspace" not in text


def test_references_all_four_config_filenames() -> None:
    text = _text()
    for filename in (
        "ts38mt_fmt_parent.yaml",
        "ts38mt_base.yaml",
        "ts38mt_pp.yaml",
        "ts38mt_fmt.yaml",
    ):
        assert filename in text, filename


def test_references_the_overlay_glob() -> None:
    text = _text()
    # Every per-size overlay filename shape must appear (the stage-4 presence
    # guard builds these explicitly, not via a directory glob).
    for stem in ("ts38mt_base_n", "ts38mt_pp_n", "ts38mt_fmt_n"):
        assert stem in text, stem
    # And the launcher's own header/stage comment names the glob shape it
    # refuses to use for counting (parity with ts38dense's "no directory
    # glob" rationale) — pins that the 30-overlay design is stated somewhere.
    assert "ts38mt_*_n*.yaml" in text


def test_pushes_to_both_repo_ids() -> None:
    text = _text()
    assert "mhieuuu/geode-store" in text
    assert "mhieuuu/geode-internals" in text
    assert "RELAY_REPO" in text
    assert "INTERNALS_REPO" in text


def test_with_snapshots_only_on_the_internals_push() -> None:
    """Every literal ``hf_checkpoint.py push`` invocation IN THIS SCRIPT'S OWN
    TEXT must carry --with-snapshots and target $INTERNALS_REPO — the
    geode-store push goes through lib/launch_common.sh::push_run, which never
    accepts --with-snapshots at all, so it cannot appear inside this file."""
    text = _text()
    push_lines = [line for line in text.splitlines() if "hf_checkpoint.py push" in line]
    assert push_lines, "expected at least one literal hf_checkpoint.py push call (push_internals)"
    for line in push_lines:
        assert "--with-snapshots" in line, line
        assert "INTERNALS_REPO" in line, line
    # push_run (geode-store, no snapshots) must still be called, repeatedly —
    # confirms the geode-store side of the dual-repo push is wired up even
    # though it never appears as a literal hf_checkpoint.py push line here.
    assert text.count('push_run "$PARENT_RID"') >= 1
    assert text.count('push_run "$BASE_ARM_RID"') >= 1
    assert text.count('push_run "$PP_ARM_RID"') >= 1
    assert text.count('push_run "$FMT_ARM_RID"') >= 1


def test_push_internals_helper_defined_not_in_launch_common() -> None:
    text = _text()
    assert "push_internals()" in text
    common = (SCRIPTS / "lib" / "launch_common.sh").read_text()
    assert "--with-snapshots" not in common, (
        "push_internals must be defined in the launcher itself, not by editing the "
        "shared lib/launch_common.sh"
    )


def test_grid_loop_orders_base_before_pp_before_fmt() -> None:
    text = _text()
    m = re.search(
        r"BASE_ARM_RID=evt-ts38mt-base-n.*?PP_ARM_RID=evt-ts38mt-pp-n.*?FMT_ARM_RID=evt-ts38mt-fmt-n",
        text,
        re.DOTALL,
    )
    assert m, "expected base -> pp -> fmt ordering within the per-size grid loop"


def test_record_g5_appears_for_all_three_arms() -> None:
    text = _text()
    for rid_var in ("BASE_ARM_RID", "PP_ARM_RID", "FMT_ARM_RID"):
        assert f'record_g5 "${rid_var}"' in text, rid_var


def test_require_converged_appears_for_all_three_arms() -> None:
    text = _text()
    for rid_var in ("BASE_ARM_RID", "PP_ARM_RID", "FMT_ARM_RID"):
        assert f'require_converged "${rid_var}"' in text, rid_var


def test_halt_gate_reads_theta0_json_filename() -> None:
    text = _text()
    assert "ts38mt_family_theta0.json" in text
    assert "THETA0_JSON" in text


def test_halt_gate_names_the_lr_fallback() -> None:
    """Pins the design's exact requirement: the NOT_LEARNED failure message
    must name the pinned LR and the (unimplemented) fallback — not just fail
    silently with no guidance for the next operator."""
    text = _text()
    assert "NOT_LEARNED" in text
    assert "1e-4" in text
    assert "3e-5" in text


def test_parent_is_full_ft_not_lora() -> None:
    text = _text()
    assert "P_METHOD == full_ft" in text
    assert "not LoRA" in text


def test_no_merge_stage_anywhere() -> None:
    """All three theta0's (base, pp, fmt) are plain full-FT weights — this
    family should never call merge_adapter.py."""
    text = _text()
    assert "merge_adapter.py" not in text


def test_run_count_is_31() -> None:
    text = _text()
    assert "RUN_COUNT=$((3 * ${#SIZES[@]} + 1))" in text


def test_default_sizes_match_the_ts38dense_ten_point_grid() -> None:
    text = _text()
    assert "DEFAULT_SIZES=(1000 2154 4642 10000 21544 46416 100000 146780 215443 316228)" in text


def test_receiver_status_capture_is_errexit_safe() -> None:
    """Under `set -e` (this launcher, unlike its siblings, uses it), a bare
    `VAR=$(cmd); STATUS=$?` is unsafe: `run_receiver_check` returning 1 is
    the ORDINARY "some runs missing, retry" case, and that failed simple
    assignment triggers errexit immediately — `STATUS=$?` is never reached,
    so the retry branch never runs and the script dies with no `fail`
    message. `VAR=$(cmd) && STATUS=0 || STATUS=$?` keeps the assignment
    inside an `&&`/`||` list, which -e exempts. Empirically verified (not
    just asserted here) against a stub `run_receiver_check` that fails once
    then succeeds: the bare form dies silently, the `&&`/`||` form reaches
    the retry, re-checks, and completes."""
    text = _text()
    assert re.search(r"^\w+_STATUS=\$\?\s*$", text, re.MULTILINE) is None, (
        "found a bare `X_STATUS=$?` on its own line — the preceding "
        "`X_OUT=$(run_receiver_check ...)` assignment must be paired with "
        "`&& X_STATUS=0 || X_STATUS=$?`, not followed by a separate `X_STATUS=$?` line"
    )
    for var in ("STORE_STATUS", "INTERNALS_STATUS"):
        assert text.count(f"&& {var}=0 || {var}=$?") == 2, (
            f"expected the errexit-safe capture idiom for {var} exactly twice "
            "(initial check + one retry check)"
        )


def test_geode_store_receiver_check_requires_target_weights_too() -> None:
    """Stage 10's design (Part B item 10) is explicit: geode-store must be
    checked for model weights, not just manifest/logs — a plain push always
    carries the adapter along (Part A only strips snapshots), so requiring
    it here is a real, checkable invariant, not a redundant restatement of
    the internals-only branch."""
    text = _text()
    func_body = text[text.index("run_receiver_check() {") :]
    else_start = func_body.index("else")
    # The literal "missing = [...]" line appears TWICE in this function (once
    # for the parent, once for targets) — search for it starting AFTER
    # "else", not from the top, or this finds the parent's earlier one and
    # produces an empty (start > end) slice.
    else_end = func_body.index("missing = [r for r in required if r not in files]", else_start)
    else_branch = func_body[else_start:else_end]
    assert "model/adapter.safetensors" in else_branch
