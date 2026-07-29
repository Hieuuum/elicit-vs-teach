"""Import smoke + equivalence belt for the training-run ``_lib`` package and
``analysis/steering.py``'s repoint onto it (reorg commit 10).

``_lib`` re-exports the trainer boilerplate that is defined once in
``scripts/train.py`` (single source). steering imports ``load_config`` from it
by file path, with NO ``sys.path`` manipulation. These smoke tests pin:
  * ``_lib`` imports and exposes the boilerplate names;
  * ``_lib.load_config`` produces byte-for-byte the same merged config as the
    single-source ``train.load_config`` (the belt against drift under X);
  * steering still imports and its ``load_config`` resolves through ``_lib``.
"""

from __future__ import annotations

import importlib.util

from tests._scriptloader import load, repo_root

REPO = repo_root()
CONFIGS = REPO / "experiments" / "training-run" / "configs"
_SAMPLE_CONFIG = CONFIGS / "run2_algo.yaml"  # walks up to configs/common.yaml


def _load_lib():
    """Path-load the ``_lib`` package the same way steering does."""
    init = REPO / "experiments" / "training-run" / "_lib" / "__init__.py"
    spec = importlib.util.spec_from_file_location("evt_training_lib_test", init)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lib_reexports_boilerplate() -> None:
    lib = _load_lib()
    for name in ("REPO_ROOT", "deep_merge", "load_config", "phase", "git_commit"):
        assert hasattr(lib, name), name


def test_lib_load_config_equals_train() -> None:
    lib = _load_lib()
    train = load("train")
    assert lib.load_config(_SAMPLE_CONFIG, None) == train.load_config(_SAMPLE_CONFIG, None)


def test_steering_load_config_resolves_through_lib() -> None:
    steer = load("steering")
    train = load("train")
    assert steer.load_config(_SAMPLE_CONFIG, None) == train.load_config(_SAMPLE_CONFIG, None)
