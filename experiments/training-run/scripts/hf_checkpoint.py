"""Move run artifacts through HF Hub when a direct laptop->box rsync is slow.

The 2026-07-20 v2-ext transfer crawled at ~40 kB/s on a single lossy TCP
stream to the rental box. huggingface_hub >= 1.0 transfers via the Xet
backend — parallel chunked up/downloads to a well-peered CDN — so
laptop -> HF -> box beats one bad route whenever the laptop's raw uplink
isn't the limit. Side benefit: an off-site, hash-verified checkpoint
archive that future boxes can pull without the laptop.

Usage:
    # laptop (must be logged in — check with `hf auth whoami`):
    python hf_checkpoint.py push
    # box (repo is private by default => `hf auth login` with a read token):
    python hf_checkpoint.py pull

The hub repo (default mhieuuu/geode-store, private) mirrors the local
store layout — runs/<run-id>/... — so every run lands as its own folder
in the one repo; --run-id selects which run moves. Idempotent both ways:
Xet dedups unchanged chunks, so re-runs are cheap. Pull verifies
model.safetensors against the hub's stored sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[3]
CKPT_REL = Path("pretrain") / "model" / "model.safetensors"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def push(store: Path, run_id: str, repo_id: str, public: bool) -> int:
    src = store / "runs" / run_id
    ckpt = src / CKPT_REL
    if not ckpt.is_file():
        raise SystemExit(f"[hf] no checkpoint at {ckpt} — wrong --store/--run-id?")
    print(f"[hf] local  model.safetensors sha256 {sha256_of(ckpt)}")
    api = HfApi()
    api.create_repo(repo_id, private=not public, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=src,
        path_in_repo=f"runs/{run_id}",
        commit_message=f"{run_id} run artifacts",
    )
    print(f"[hf] pushed {src} -> https://huggingface.co/{repo_id}/tree/main/runs/{run_id}")
    print("[hf] next, on the box: python hf_checkpoint.py pull")
    return 0


def pull(store: Path, run_id: str, repo_id: str) -> int:
    repo_path = f"runs/{run_id}/{CKPT_REL.as_posix()}"
    # The repo mirrors the store layout, so unpacking at the store root
    # puts runs/<run-id>/... exactly where train.py expects it.
    snapshot_download(repo_id=repo_id, local_dir=store, allow_patterns=[f"runs/{run_id}/*"])
    ckpt = store / "runs" / run_id / CKPT_REL
    if not ckpt.is_file():
        raise SystemExit(f"[hf] pull finished but {ckpt} is missing — wrong --repo-id/--run-id?")
    local = sha256_of(ckpt)
    info = HfApi().get_paths_info(repo_id, [repo_path])
    remote = info[0].lfs.sha256 if info and info[0].lfs else None
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
    parser.add_argument("--run-id", default="evt-run1-base-v2")
    parser.add_argument("--repo-id", default="mhieuuu/geode-store")
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
    args = parser.parse_args()
    if args.cmd == "push":
        return push(args.store, args.run_id, args.repo_id, args.public)
    return pull(args.store, args.run_id, args.repo_id)


if __name__ == "__main__":
    sys.exit(main())
