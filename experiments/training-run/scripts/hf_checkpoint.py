"""Move run artifacts through HF Hub when a direct laptop->box rsync is slow.

The 2026-07-20 v2-ext transfer crawled at ~40 kB/s on a single lossy TCP
stream to the rental box. huggingface_hub >= 1.0 transfers via the Xet
backend — parallel chunked up/downloads to a well-peered CDN — so
laptop -> HF -> box beats one bad route whenever the laptop's raw uplink
isn't the limit. Side benefit: an off-site, hash-verified checkpoint
archive that future boxes can pull without the laptop.

Usage:
    # laptop (must be logged in — check with `hf auth whoami`):
    python3 hf_checkpoint.py push --run-id <run-id>
    # box (repo is private by default => `hf auth login --force` with a read token):
    python3 hf_checkpoint.py pull --run-id <run-id>
    # laptop, logs/manifest only — no *.safetensors (analysis without weights):
    python3 hf_checkpoint.py pull --run-id <run-id> --no-weights
    # push manifest/logs/gates + any adapter.safetensors sidecar — no
    # model.safetensors (bulk-sweep runs whose full checkpoint is
    # intentionally never archived to the relay, but a LoRA run's compact
    # adapter still rides along so its weights stay cheaply recoverable):
    python3 hf_checkpoint.py push --run-id <run-id> --no-weights
    # push manifest/logs/gates/eval ONLY — not one byte of *.safetensors,
    # adapter sidecars included (a sweep whose runs are deliberately not
    # recoverable: the relay keeps the measurement, the weights die with
    # the box). Strictly stronger than --no-weights; the two are exclusive:
    python3 hf_checkpoint.py push --run-id <run-id> --metadata-only
    # either direction: snapshots/ and sft_snapshots/ are SKIPPED by default
    # (multi-GB, needed only for extraction) — opt in explicitly:
    python3 hf_checkpoint.py pull --run-id <run-id> --with-snapshots

The hub repo (default mhieuuu/geode-store, private) mirrors the local
store layout — runs/<run-id>/... — so every run lands as its own folder
in the one repo; --run-id selects which run moves. Idempotent both ways:
Xet dedups unchanged chunks, so re-runs are cheap. Pull verifies
model.safetensors against the hub's stored sha256.

model_merged/ (scripts/merge_adapter.py's plain-checkpoint fold of a LoRA
install run's adapter, for a later stage's --init-from) is EXCLUDED from
every push, weights or no-weights (2026-07-31): it is a derivable artifact,
regenerable from model/ in seconds on any box, so archiving it to the
relay is pure waste.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from geode.zoo import checkpoint_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPO_ID = "mhieuuu/geode-store"


def find_checkpoint(store: Path, run_id: str) -> Path:
    """The run's single final ``model.safetensors`` (flat OR legacy layout).

    Thin wrapper over ``geode.zoo.checkpoint_dir`` — the one resolver for
    the flat ``runs/<id>/model/`` layout and the legacy phase-dir layout
    (``pretrain/`` run 1, ``sft/`` runs 2-4), spec 00 §1 — converting its
    refusals into clean CLI errors. Exactly one checkpoint must exist
    across both patterns; zero or several is a wrong-dir/half-migrated
    signal, not a guess to make.
    """
    try:
        return checkpoint_dir(run_id, store=store) / "model.safetensors"
    except (FileNotFoundError, RuntimeError) as e:
        raise SystemExit(f"[hf] {e} — wrong --store/--run-id?") from e


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def hub_lfs_sha256(repo_id: str, repo_path: str) -> str | None:
    """The hub's stored LFS sha256 for one file, or ``None`` if unrecorded.

    A single wrapper over ``get_paths_info`` so every push/pull verifier reads
    the remote digest the same way. ``None`` means the hub has no LFS hash for
    that path (a non-LFS file, or a path the repo does not hold); each caller
    decides whether that is a warning (``pull``) or a hard failure
    (``verify_hub_checkpoint``)."""
    info = HfApi().get_paths_info(repo_id, [repo_path])
    return info[0].lfs.sha256 if info and info[0].lfs else None


def verify_hub_checkpoint(store: Path, run_id: str, repo_id: str = DEFAULT_REPO_ID) -> str:
    """Assert the hub's ``model.safetensors`` sha256 equals the local one.

    The single implementation of the "push, then confirm the relay actually
    holds the bytes I have" check that launchers used to inline as a Python
    heredoc after ``hf_checkpoint.py push``. Strict by contract: a missing hub
    hash (``None``) counts as a mismatch and raises, because a push verifier
    that is silent when the hub reports nothing is worthless. Returns the
    verified local sha256 on success so the caller can log it.

    ``pull`` keeps its own leniency (``None`` -> warning) — that is a
    post-download sanity note, not a gate on deleting local weights."""
    ckpt = find_checkpoint(store, run_id)
    repo_path = f"runs/{run_id}/{ckpt.relative_to(store / 'runs' / run_id).as_posix()}"
    local = sha256_of(ckpt)
    remote = hub_lfs_sha256(repo_id, repo_path)
    if remote != local:
        raise SystemExit(f"[hf] {run_id}: hub sha256 {remote!r} != local {local}")
    return local


def push(
    store: Path,
    run_id: str,
    repo_id: str,
    public: bool,
    with_snapshots: bool = False,
    no_weights: bool = False,
    metadata_only: bool = False,
) -> int:
    src = store / "runs" / run_id
    # Mirrors pull's ignore-list construction below: same two knobs, same
    # patterns, so push and pull can never silently drift apart.
    # sft_snapshots/ (train_sft.py's mid-run checkpoint dir, geode/train/sft.py
    # — distinct from train_target.py's snapshots/, used by LoRA target runs)
    # skips by the same default: a full-FT parent with train.snapshot_steps
    # set would otherwise ship multi-GB intermediate checkpoints on every
    # plain push.
    ignore = [] if with_snapshots else ["snapshots/*", "sft_snapshots/*"]
    # model_merged/ (scripts/merge_adapter.py) is a derivable artifact — a
    # LoRA install run's base + adapter folded into plain weights, purely for
    # a later stage's --init-from — never worth shipping to the relay
    # (≈2.5GB, regenerable from model/ in seconds on any box). Excluded from
    # EVERY push, weights or no-weights: --no-weights already caught it by
    # coincidence (its "*model.safetensors" pattern matches any path ending
    # that way, nested or not), but a plain weights-included push had no
    # exclusion for it at all.
    ignore.append("model_merged/*")
    if metadata_only:
        # Strictly stronger than --no-weights: that flag's "*model.safetensors"
        # deliberately lets adapter.safetensors ride along, which at LoRA r512
        # is ~0.72 GB per run — the dominant relay cost of a 39-run sweep, and
        # larger in total than the full checkpoints it excludes. This branch
        # ships the measurement (manifest/logs/gates/eval) and nothing else.
        ignore.append("*.safetensors")
        print(
            "[hf] --metadata-only: ALL *.safetensors excluded from push "
            "(adapter sidecars too) — these run weights are not recoverable "
            "from the relay"
        )
    elif no_weights:
        ignore.append("*model.safetensors")
        print(
            "[hf] --no-weights: model.safetensors excluded from push "
            "(adapter.safetensors sidecars still included, if present)"
        )
    else:
        ckpt = find_checkpoint(store, run_id)
        print(f"[hf] local  model.safetensors sha256 {sha256_of(ckpt)}")
    if not with_snapshots:
        print(
            "[hf] snapshots/ and sft_snapshots/ skipped (default; pass --with-snapshots to include)"
        )
    api = HfApi()
    api.create_repo(repo_id, private=not public, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=src,
        path_in_repo=f"runs/{run_id}",
        commit_message=f"{run_id} run artifacts",
        ignore_patterns=ignore or None,
    )
    print(f"[hf] pushed {src} -> https://huggingface.co/{repo_id}/tree/main/runs/{run_id}")
    if metadata_only:
        print("[hf] metadata-only push complete: no weights on the relay for this run")
    elif no_weights:
        print("[hf] next: push again without --no-weights when this run's weights are needed on the hub")
    else:
        print("[hf] next, on the box: python3 hf_checkpoint.py pull")
    return 0


def pull(
    store: Path, run_id: str, repo_id: str, no_weights: bool = False, with_snapshots: bool = False
) -> int:
    ignore = (
        [] if with_snapshots else [f"runs/{run_id}/snapshots/*", f"runs/{run_id}/sft_snapshots/*"]
    )
    if no_weights:
        ignore.append("*.safetensors")
    if not with_snapshots:
        print(
            "[hf] snapshots/ and sft_snapshots/ skipped (default; pass --with-snapshots to include)"
        )
    # The repo mirrors the store layout, so unpacking at the store root
    # puts runs/<run-id>/... exactly where train.py expects it.
    snapshot_download(
        repo_id=repo_id,
        local_dir=store,
        allow_patterns=[f"runs/{run_id}/*"],
        ignore_patterns=ignore or None,
    )
    if no_weights:
        # No checkpoint landed, so there is nothing to sha-verify; a later
        # plain pull fills in the weights (snapshot_download resumes). But
        # snapshot_download itself is fail-open (e.g. a bad/expired token can
        # silently fetch nothing), so check that SOMETHING landed before
        # printing success — a run dir with no manifest.json means the pull
        # got nothing at all.
        manifest_path = store / "runs" / run_id / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(
                f"[hf] pull landed nothing at {manifest_path} — check HF auth "
                f"(`hf auth whoami`) and that '{run_id}' exists on {repo_id}"
            )
        print(f"[hf] pulled runs/{run_id} manifest/logs only (*.safetensors skipped)")
        return 0
    ckpt = find_checkpoint(store, run_id)
    repo_path = f"runs/{run_id}/{ckpt.relative_to(store / 'runs' / run_id).as_posix()}"
    local = sha256_of(ckpt)
    remote = hub_lfs_sha256(repo_id, repo_path)
    if remote is None:
        print(
            f"[hf] WARNING: hub reports no hash for {repo_path}; local sha256 {local} — "
            "compare by eye against the push-side printout"
        )
    elif local != remote:
        raise SystemExit(f"[hf] sha256 MISMATCH after pull: local {local} != hub {remote}")
    else:
        print(f"[hf] verified model.safetensors sha256 {local}")
    print(f"[hf] checkpoint ready: --init-from {ckpt.parent}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("push", "pull"))
    # Required, no default: a forgotten --run-id must not silently push a run
    # whose relay manifest was backfilled elsewhere (run-1 v1/v2/v2-ext/v3).
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")),
        help="artifact store root (default: $GEODE_STORE, else <repo>/geode-store)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="push only: create the repo public, so pull needs no token",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="push: exclude model.safetensors (the full state dict) but still "
        "include any adapter.safetensors sidecar, so a LoRA run's weights stay "
        "cheaply recoverable even from a metadata-scale push; pull: manifest/"
        "logs/gates land for laptop-side analysis without any weights",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="push only: exclude EVERY *.safetensors, adapter sidecars included "
        "— manifest/logs/gates/eval reach the relay and the run's weights do "
        "not survive box teardown. Stronger than --no-weights; exclusive with it",
    )
    parser.add_argument(
        "--with-snapshots",
        action="store_true",
        help="include runs/<run-id>/snapshots/ AND runs/<run-id>/sft_snapshots/ "
        "(skipped by default both ways — multi-GB, needed only for extraction)",
    )
    args = parser.parse_args()
    if args.no_weights and args.with_snapshots:
        parser.error("--no-weights and --with-snapshots contradict (snapshots are weights)")
    if args.metadata_only and args.with_snapshots:
        parser.error("--metadata-only and --with-snapshots contradict (snapshots are weights)")
    if args.metadata_only and args.no_weights:
        parser.error("--metadata-only and --no-weights are exclusive (pick one exclusion policy)")
    if args.metadata_only and args.cmd == "pull":
        parser.error(
            "--metadata-only is push-only; pull --no-weights already excludes *.safetensors"
        )
    if args.cmd == "push":
        return push(
            args.store,
            args.run_id,
            args.repo_id,
            args.public,
            args.with_snapshots,
            args.no_weights,
            args.metadata_only,
        )
    return pull(args.store, args.run_id, args.repo_id, args.no_weights, args.with_snapshots)


if __name__ == "__main__":
    sys.exit(main())
