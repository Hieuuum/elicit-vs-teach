"""Every ``gates.py`` invocation in every launcher shell parses.

Second instance of the same failure class as
``test_config_completeness.py``, and the same justification (the CLAUDE.md
promotion rule's *cost* clause, not its silence clause): on 2026-07-27
``launch_phase3.sh`` called

    gates.py g4 --run "$PARENT_RID" --prompt-config ... --threshold 0.90

with no ``--config``, which ``gates.py`` requires unconditionally — it reads
the tokenizer path and ``cfg["train"]["stopping"]`` from it before it ever
looks at ``--prompt-config``. argparse exited 2. Both G4 call sites had it,
so the installer branch would have hit the same wall later in the chain.

The launcher's own guard behaved correctly — it parses the printed rate rather
than trusting the exit code, so it refused to read an argparse error as a
score and halted without recording anything. The cost was still a box
round-trip on a paid GPU, after the parent had trained.

This checks only that the arguments are *accepted*: values are placeholders,
no gate runs, no model loads. It cannot catch a wrong config being passed —
only a missing or misspelled flag.
"""

from __future__ import annotations

import re
import shlex

import pytest

from tests._scriptloader import SCRIPTS, load

# "$VAR" / ${VAR} / $VAR -> a placeholder. Every shell variable in these call
# sites is a run id or a checkpoint path, i.e. a plain argument value; none
# expands to a flag, so substitution cannot change the parse.
_VAR = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


def _invocations(text: str) -> list[list[str]]:
    """Every `python3 gates.py ...` command in a shell script, as argv lists."""
    joined = text.replace("\\\n", " ")
    out = []
    for line in joined.splitlines():
        line = line.split("#", 1)[0] if line.lstrip().startswith("#") else line
        for m in re.finditer(r"python3 gates\.py (.+)", line):
            tail = m.group(1)
            # Stop at the first shell token that is not an argument: a control
            # operator, a closing subshell paren, or a redirection (` 2>&1`).
            tail = re.split(r"\|\||&&|\|(?!\|)|\)|;|\s\d*>", tail)[0]
            out.append(shlex.split(_VAR.sub("PLACEHOLDER", tail)))
    return out


# The live probe100k launcher stays in scripts/; the retired launchers moved to
# scripts/archive/ (reorg commit 14). Glob both (non-recursively, so scripts/lib/
# glue is excluded) as SCRIPTS-relative paths, so the live launcher keeps
# coverage and the archived ones stay byte-identical-but-tested.
LAUNCHERS = sorted(
    str(p.relative_to(SCRIPTS))
    for directory in (SCRIPTS, SCRIPTS / "archive")
    for p in directory.glob("launch_*.sh")
)


def test_launchers_are_discovered() -> None:
    """Guard the guard: a rename that empties the glob must not silently pass."""
    assert LAUNCHERS, "no launch_*.sh found — this test would vacuously pass"
    assert "launch_llama_probe100k.sh" in LAUNCHERS, "live probe100k launcher not discovered"
    assert sum(name.startswith("archive/") for name in LAUNCHERS) == 10, (
        "expected 10 archived launchers under scripts/archive/"
    )


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_gate_invocations_parse(launcher: str) -> None:
    parser = load("gates").build_parser()
    calls = _invocations((SCRIPTS / launcher).read_text())
    for argv in calls:
        try:
            parser.parse_args(argv)
        except SystemExit as exc:  # argparse exits 2 on a bad/missing argument
            pytest.fail(f"{launcher}: `gates.py {' '.join(argv)}` does not parse ({exc})")
