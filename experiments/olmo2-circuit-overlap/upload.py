"""Upload only completed/partial evaluation artifacts to an explicit private HF repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

from huggingface_hub import CommitOperationAdd, HfApi

from geode.circuits.artifacts import artifact_upload_files, sha256_file, write_json


def upload(root: Path, repo_id: str, prefix: str) -> str:
    if not prefix or PurePosixPath(prefix).is_absolute() or ".." in PurePosixPath(prefix).parts:
        raise ValueError("prefix must be a relative repository path without parent traversal")
    if not (root / "run_metadata.json").is_file():
        raise ValueError("not an evaluation artifact directory")
    # A manifest cannot hash its own final contents. Exclude an existing root
    # manifest on repeat uploads, while still uploading the freshly written
    # manifest alongside every content artifact below.
    files = [p for p in artifact_upload_files(root) if p != root / "upload_manifest.json"]
    manifest = {
        str(p.relative_to(root)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in files
    }
    write_json(root / "upload_manifest.json", manifest)
    api = HfApi()
    info = api.repo_info(repo_id, repo_type="dataset")
    if not info.private:
        raise ValueError("artifact destination must be private")
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{prefix}/{p.relative_to(root).as_posix()}", path_or_fileobj=str(p)
        )
        for p in artifact_upload_files(root)
    ]
    result = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Add {prefix} evaluation artifacts",
    )
    return result.commit_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    print(json.dumps({"commit_url": upload(args.directory, args.repo, args.prefix)}))
