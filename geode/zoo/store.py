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


def checkpoint_dir(run_id: str, *, store: Path | None = None) -> Path:
    """The run's final ``save_pretrained`` checkpoint directory (specs/00 §1, V0.8).

    Flat layout: ``runs/{run_id}/model``. Legacy (pre-2026-07-21 migration)
    layout: ``runs/{run_id}/{phase}/model`` where the phase dir is named for
    how the run was trained (``pretrain/`` for run 1, ``sft/`` for runs 2-4).
    Exactly one checkpoint must exist across both patterns: zero means a
    wrong store or run_id, more than one means a half-migrated or corrupt
    run dir. Both are refusals naming the run dir — silently loading the
    wrong checkpoint is the failure this resolver exists to prevent.
    Snapshots (``snapshots/step_{k}/model.safetensors``, no ``model/`` path
    component) never match either pattern.
    """
    rd = run_dir(run_id, store=store)
    flat = rd / "model" / "model.safetensors"
    candidates = ([flat] if flat.is_file() else []) + sorted(rd.glob("*/model/model.safetensors"))
    if not candidates:
        raise FileNotFoundError(
            f"no final checkpoint under {rd}: neither model/model.safetensors (flat, "
            "specs/00 §1) nor */model/model.safetensors (legacy phase dir) — "
            "wrong store or run_id?"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"ambiguous final checkpoint under {rd}: found "
            f"{[str(c) for c in candidates]} — half-migrated or corrupt run dir; "
            "finish or redo the layout migration (migrate_store_layout.py), "
            "never guess between checkpoints"
        )
    return candidates[0].parent


def manifest_path(run_id: str, *, store: Path | None = None) -> Path:
    """Path to a run's manifest: ``$GEODE_STORE/runs/{run_id}/manifest.json``."""
    return run_dir(run_id, store=store) / "manifest.json"


def prequential_log_path(run_id: str, *, store: Path | None = None) -> Path:
    """Path to a run's prequential log: ``.../logs/prequential.jsonl`` (specs/00 §3)."""
    return run_dir(run_id, store=store) / "logs" / "prequential.jsonl"


def gradstats_log_path(run_id: str, *, store: Path | None = None) -> Path:
    """Path to a run's gradient statistics log: ``.../logs/gradstats.jsonl`` (specs/00 §4)."""
    return run_dir(run_id, store=store) / "logs" / "gradstats.jsonl"


def test_loss_path(run_id: str, *, store: Path | None = None) -> Path:
    """Path to a run's held-out loss record: ``.../eval/test_loss.json`` (specs/00 §5)."""
    return run_dir(run_id, store=store) / "eval" / "test_loss.json"
