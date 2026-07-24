"""Migrate run dirs from the legacy phase-dir layout to the flat layout.

Legacy runs nest everything under a phase dir named for how they were
trained — ``runs/<id>/pretrain/`` (run 1) or ``runs/<id>/sft/`` (runs
2-4). The flat layout (spec 00 §1, 2026-07-21) puts the final checkpoint
at ``runs/<id>/model/`` and every other phase-dir file (train_log.jsonl,
eval_log.jsonl, training_meta.json, ...) at the run root. This script is
the ONLY thing that changes a store's layout: all readers
(``geode.zoo.checkpoint_dir``, monitor.py, hf_checkpoint.py) accept both
layouts, so migration is never implicit and never urgent. Run it only
after training ends — never against a run that is still writing.

DRY-RUN by default: prints the exact planned moves and touches nothing.
``--apply`` executes. Already-flat runs are no-ops (idempotent); a run
with an ambiguity (two phase dirs) or a collision (a destination already
exists) is refused with a loud per-run error while the other runs
continue. ``--run-id`` restricts to one run; without it, every dir under
``runs/`` is visited — the script never invents run ids.

Usage:
    python3 migrate_store_layout.py [--store <root>]        # dry-run, all runs
    python3 migrate_store_layout.py --apply                 # migrate all runs
    python3 migrate_store_layout.py --run-id <id> --apply   # one run
    python3 migrate_store_layout.py --run-id <id> --relay --apply

``--relay`` mirrors one already-migrated run to the HF relay repo
(default mhieuuu/geode-store, which mirrors ``runs/<id>/...`` paths
exactly). It verifies the local run is flat, then calls ``upload_folder``
with ``delete_patterns`` for ``sft/*`` and ``pretrain/*`` — additions and
deletions land in ONE commit, so the remote never lacks a checkpoint.
``--relay`` requires an explicit ``--run-id``: the relay is never touched
wholesale. Without ``--apply`` it only prints the plan.

RELAY WARNING: four relay runs must NEVER be pushed or modified from a
rental box — evt-run1-base-v1, -v2, -v2-ext, -v3. Their relay manifests
were backfilled from the laptop, so any box-side copy is stale and a push
from a box would clobber the authoritative record. If those runs ever
need this migration, run it (including the --relay step) on the laptop,
where the authoritative copies live.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from geode.zoo import checkpoint_dir

REPO_ROOT = Path(__file__).resolve().parents[3]

# Legacy phase dirs by name — the two layouts that ever existed (spec 00 §1
# history). Named explicitly so run-root dirs like snapshots/, logs/, eval/
# can never be mistaken for a phase dir.
PHASE_DIRS = ("pretrain", "sft")


class MigrationError(RuntimeError):
    """A run dir that cannot be migrated safely (ambiguity or collision)."""


def migrate_run(run_dir: Path, *, apply: bool) -> list[tuple[Path, Path]]:
    """Plan (and with ``apply``) execute the legacy->flat moves for one run.

    Returns the (src, dst) pairs; ``[]`` means no phase dir — already flat
    (or nothing trained yet), a no-op. Raises :class:`MigrationError` when
    two phase dirs exist or any destination already exists; every
    destination is checked before the first rename, so a refused run is
    left untouched. Renames stay inside ``run_dir`` (same filesystem).
    """
    phases = [run_dir / name for name in PHASE_DIRS if (run_dir / name).is_dir()]
    if not phases:
        return []
    if len(phases) > 1:
        raise MigrationError(
            f"{run_dir}: both phase dirs present ({[p.name for p in phases]}) — "
            "ambiguous; resolve by hand before migrating"
        )
    phase = phases[0]
    moves = [(src, run_dir / src.name) for src in sorted(phase.iterdir())]
    collisions = [str(dst) for _, dst in moves if dst.exists()]
    if collisions:
        raise MigrationError(
            f"{run_dir}: destination already exists for {collisions} — "
            "refusing to overwrite; resolve by hand before migrating"
        )
    if apply:
        for src, dst in moves:
            src.rename(dst)
        phase.rmdir()  # raises if anything appeared concurrently — good
    return moves


def relay_run(store: Path, run_id: str, repo_id: str, *, apply: bool) -> None:
    """Mirror one migrated run to the HF relay: upload flat + delete legacy.

    Verifies the local run is flat first (the relay must only ever receive
    a layout the local store actually has), then uses ``upload_folder``
    with ``delete_patterns`` so the flat additions and the legacy
    deletions form a single commit — the remote never lacks a checkpoint.
    Raises :class:`MigrationError` when the local run is not cleanly flat.
    """
    run_dir = store / "runs" / run_id
    try:
        ckpt = checkpoint_dir(run_id, store=store)
    except (FileNotFoundError, RuntimeError) as e:
        raise MigrationError(f"{run_dir}: not relayable — {e}") from e
    if ckpt != run_dir / "model":
        raise MigrationError(
            f"{run_dir}: checkpoint still at {ckpt} — migrate locally (--apply) before --relay"
        )
    patterns = [f"{name}/*" for name in PHASE_DIRS]
    if not apply:
        print(
            f"[migrate] DRY-RUN relay {run_id}: would upload {run_dir} -> "
            f"{repo_id}:runs/{run_id} and delete remote {patterns} in the same commit"
        )
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        folder_path=run_dir,
        path_in_repo=f"runs/{run_id}",
        delete_patterns=patterns,
        commit_message=f"{run_id}: migrate to flat run layout (spec 00 §1)",
    )
    print(f"[migrate] relayed {run_id} -> https://huggingface.co/{repo_id}/tree/main/runs/{run_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=None, help="restrict to one run (required for --relay)")
    parser.add_argument("--apply", action="store_true", help="execute; default is dry-run")
    parser.add_argument("--relay", action="store_true", help="mirror the migrated run to HF")
    parser.add_argument("--repo-id", default="mhieuuu/geode-store", help="relay repo")
    parser.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")),
        help="artifact store root (default: $GEODE_STORE, else <repo>/geode-store)",
    )
    args = parser.parse_args()

    runs_root = args.store / "runs"
    if args.run_id:
        if not (runs_root / args.run_id).is_dir():
            print(f"[migrate] no such run dir: {runs_root / args.run_id}")
            return 1
        run_dirs = [runs_root / args.run_id]
    else:
        if args.relay:
            print("[migrate] --relay requires an explicit --run-id; the relay is never")
            print("[migrate] touched wholesale (see the module docstring's relay warning)")
            return 1
        run_dirs = (
            sorted(d for d in runs_root.iterdir() if d.is_dir()) if runs_root.is_dir() else []
        )
        if not run_dirs:
            print(f"[migrate] no run dirs under {runs_root}")
            return 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    failed = 0
    for run_dir in run_dirs:
        try:
            moves = migrate_run(run_dir, apply=args.apply)
        except MigrationError as e:
            failed += 1
            print(f"[migrate] ERROR {e}")
            continue
        if not moves:
            print(f"[migrate] {run_dir.name}: already flat, nothing to do")
            continue
        for src, dst in moves:
            print(
                f"[migrate] {mode} {run_dir.name}: "
                f"{src.relative_to(run_dir)} -> {dst.relative_to(run_dir)}"
            )
    if args.relay and failed == 0:
        try:
            relay_run(args.store, args.run_id, args.repo_id, apply=args.apply)
        except MigrationError as e:
            failed += 1
            print(f"[migrate] ERROR {e}")
    if not args.apply:
        print("[migrate] dry-run only — nothing was changed; re-run with --apply to execute")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
