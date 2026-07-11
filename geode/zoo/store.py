"""Artifact store resolution and path helpers (specs/00 §1).

The store root is ``$GEODE_STORE`` — an environment variable, never a
hardcoded path. An explicit ``store=`` argument always wins over the
environment; when neither is available, resolution fails loudly.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_store(store: Path | None = None) -> Path:
    """Resolve the store root: explicit ``store`` arg, else ``$GEODE_STORE``."""
    if store is not None:
        return Path(store)
    env = os.environ.get("GEODE_STORE")
    if env:
        return Path(env)
    raise RuntimeError(
        "no artifact store configured: pass store= explicitly or set the "
        "GEODE_STORE environment variable (specs/00 §1)"
    )


def run_dir(run_id: str, *, store: Path | None = None) -> Path:
    """Directory for one run: ``$GEODE_STORE/runs/{run_id}``."""
    return resolve_store(store) / "runs" / run_id


def manifest_path(run_id: str, *, store: Path | None = None) -> Path:
    """Path to a run's manifest: ``$GEODE_STORE/runs/{run_id}/manifest.json``."""
    return run_dir(run_id, store=store) / "manifest.json"
