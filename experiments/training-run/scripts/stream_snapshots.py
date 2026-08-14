"""Stream a training run's snapshots to HF as they are written; delete after verify.

Sidecar for launch_fig2nl3s_llama.sh (owner 2026-08-13: "delete locally after
every snap, not after the whole run"). Runs alongside train_target.py and
keeps local disk flat: each ``snapshots/step_N/adapter.safetensors`` is
uploaded to the run's per-run HF repo, sha256-verified against the hub's LFS
record, and only then deleted locally.

Write-race safety without trainer changes: ``_save_snapshot`` writes
non-atomically but strictly serially, so a ``step_N`` file is guaranteed
complete once any HIGHER-step snapshot exists. The streamer therefore only
touches snapshots older than the newest one — until the launcher touches the
DONE marker (training finished), after which everything remaining, including
``snapshots/base/``, is drained.

Crash note: if training dies mid-run, already-streamed snapshots live only on
the hub. The launcher's train_or_skip refuses a 'running' manifest on resume;
if the operator deletes the run dir and retrains, re-streamed files overwrite
the same hub paths — stale steps from the aborted attempt may linger in the
hub repo and are harmless (the manifest names the real schedule).

Usage (launcher-managed):
    python3 stream_snapshots.py --run-id <rid> --repo-id <ns>/<rid> \
        --done-marker <path> [--interval 30]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from pathlib import Path

from huggingface_hub import HfApi

from hf_checkpoint import hub_lfs_sha256

STEP_RE = re.compile(r"^step_(\d+)$")


def local_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def eligible_files(snap_root: Path, done: bool) -> list[Path]:
    """Snapshot files safe to upload now (module docstring's serial-write rule)."""
    step_dirs = []
    for d in snap_root.iterdir() if snap_root.is_dir() else []:
        m = STEP_RE.match(d.name)
        if m and d.is_dir():
            step_dirs.append((int(m.group(1)), d))
    step_dirs.sort()
    if not done and step_dirs:
        step_dirs = step_dirs[:-1]  # newest may still be mid-write
    files = [d / "adapter.safetensors" for _, d in step_dirs]
    if done:
        base = snap_root / "base" / "model.safetensors"
        if base.is_file():
            files.append(base)
    return [f for f in files if f.is_file()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--done-marker", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", Path(__file__).resolve().parents[3] / "geode-store")),
    )
    args = ap.parse_args()

    run_dir = args.store / "runs" / args.run_id
    snap_root = run_dir / "snapshots"
    api = HfApi()
    api.create_repo(args.repo_id, private=True, exist_ok=True)
    n_uploaded = 0

    while True:
        done = args.done_marker.exists()
        for f in eligible_files(snap_root, done):
            rel = f.relative_to(run_dir)  # snapshots/step_N/adapter.safetensors
            repo_path = f"runs/{args.run_id}/{rel.as_posix()}"
            want = local_sha256(f)
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=repo_path,
                repo_id=args.repo_id,
            )
            got = hub_lfs_sha256(args.repo_id, repo_path)
            if got != want:
                raise SystemExit(
                    f"[stream] {repo_path}: hub sha {got} != local {want} — NOT deleting; "
                    "upload corrupt or racing, investigate"
                )
            f.unlink()
            try:  # remove the emptied step dir; base/ dir likewise
                f.parent.rmdir()
            except OSError:
                pass
            n_uploaded += 1
            print(f"[stream] uploaded+verified+deleted {rel.as_posix()} "
                  f"({n_uploaded} total)", flush=True)
        if done and not eligible_files(snap_root, True):
            try:
                if snap_root.is_dir():
                    snap_root.rmdir()
            except OSError:
                pass  # non-empty means something unexpected; leave for inspection
            print(f"[stream] drained: {n_uploaded} snapshot file(s) on {args.repo_id}")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
