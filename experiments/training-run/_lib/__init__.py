"""Shared training-run boilerplate: ``REPO_ROOT``, ``load_config``, ``phase``,
``git_commit`` (and ``deep_merge``).

``experiments/training-run`` has a hyphen, so it can be neither a dotted package
(``experiments.training-run._lib`` is not a valid import path) nor pip-installed.
Reach it either by file path anchored on the repo root (see
``analysis/steering.py``) or, from a launcher-run trainer, with
``experiments/training-run`` on ``sys.path``.

Single source of truth: these names are DEFINED in ``scripts/train.py`` and
already imported from there by the other trainers (``from train import
REPO_ROOT, git_commit, load_config, phase``). This package re-exports the very
same objects by path-loading ``train.py`` — one definition, zero drift. It is a
forward-looking facade: when the boilerplate later migrates into this package,
the trainers' imports flip to ``_lib`` with no change at any call site.

``train.py`` has no module-level side effects (only ``REPO_ROOT`` + function
defs, with everything runnable behind an ``if __name__ == '__main__'`` guard)
and no scripts-sibling imports, so path-loading it here is safe and needs no
``sys.path`` manipulation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_TRAIN_PY = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
# A private, unique module name — never "train": the test scriptloader registers
# ``sys.modules["train"]`` for the trainer tests, and sharing that name would
# double-execute the module and make imports order-dependent.
_MODULE_NAME = "_evt_train_boilerplate"


def _load_train_boilerplate() -> ModuleType:
    """Path-load ``scripts/train.py`` once under a private module name."""
    cached = sys.modules.get(_MODULE_NAME)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_train = _load_train_boilerplate()

REPO_ROOT = _train.REPO_ROOT
deep_merge = _train.deep_merge
load_config = _train.load_config
phase = _train.phase
git_commit = _train.git_commit

__all__ = ["REPO_ROOT", "deep_merge", "load_config", "phase", "git_commit"]
