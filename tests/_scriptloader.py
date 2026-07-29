"""Path-anchored loader for the training-run ``scripts/`` and ``analysis/`` modules.

``experiments/training-run`` has a hyphen, so its scripts and analysis modules
cannot be a dotted package (``experiments.training-run.scripts.foo`` is not a
valid import path). They are flat modules that import their siblings by bare
name (``from train import load_config``, ``from adapters import ...``), matching
the launcher contract of ``cd scripts/ && python3 foo.py``.

This helper anchors on the repo root (the nearest ancestor holding
``pyproject.toml``, so it is independent of how deep the calling test lives) and
exposes the two source dirs plus ``load(name)``, which imports a module by file
path with both dirs on ``sys.path`` so those bare intra-tree imports resolve.
The inserts are left in place and the module registered in ``sys.modules`` so a
sibling imported lazily — e.g. ``trajectory`` doing ``from adapters import ...``
inside a function, or ``steering`` doing ``from train import load_config`` —
resolves later too.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def repo_root() -> Path:
    """The geode repo root: the nearest ancestor containing ``pyproject.toml``."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("no pyproject.toml found above tests/_scriptloader.py")


_ROOT = repo_root()
SCRIPTS = _ROOT / "experiments" / "training-run" / "scripts"
ANALYSIS = _ROOT / "experiments" / "training-run" / "analysis"


def load(name: str) -> ModuleType:
    """Import a training-run ``scripts/``/``analysis/`` module by file path.

    Looks in ``SCRIPTS`` then ``ANALYSIS`` for ``<name>.py`` (the two trees share
    no module names). Both dirs are placed on ``sys.path`` and left there so the
    module's bare sibling imports resolve, including any deferred until call
    time; the module is registered in ``sys.modules`` under ``name`` before it is
    executed (a path-loaded ``@dataclass`` resolves its own module through
    ``sys.modules`` at class-creation time).
    """
    for directory in (SCRIPTS, ANALYSIS):
        path = directory / f"{name}.py"
        if path.is_file():
            for source_dir in (SCRIPTS, ANALYSIS):
                if str(source_dir) not in sys.path:
                    sys.path.insert(0, str(source_dir))
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"no {name}.py in {SCRIPTS} or {ANALYSIS}")
