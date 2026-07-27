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

import importlib.util
import re
import shlex
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "experiments" / "training-run" / "scripts"

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


def _gates_parser():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("gates", SCRIPTS / "gates.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build_parser()
    finally:
        sys.path.remove(str(SCRIPTS))


LAUNCHERS = sorted(p.name for p in SCRIPTS.glob("launch_*.sh"))


def test_launchers_are_discovered() -> None:
    """Guard the guard: a rename that empties the glob must not silently pass."""
    assert LAUNCHERS, "no launch_*.sh found — this test would vacuously pass"


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_gate_invocations_parse(launcher: str) -> None:
    parser = _gates_parser()
    calls = _invocations((SCRIPTS / launcher).read_text())
    for argv in calls:
        try:
            parser.parse_args(argv)
        except SystemExit as exc:  # argparse exits 2 on a bad/missing argument
            pytest.fail(f"{launcher}: `gates.py {' '.join(argv)}` does not parse ({exc})")
