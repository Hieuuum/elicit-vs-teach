"""Atomic, finite, provenance-rich artifacts for offline evaluation (spec 00 C9)."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=lambda x: asdict(x) if is_dataclass(x) else _unsupported(x),
    )


def _unsupported(value: Any) -> None:
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    text = canonical_json(value) + "\n"  # validate BEFORE opening destination
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(canonical_json(row) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def environment_provenance(repo_root: Path) -> dict:
    packages = {}
    for name in ("torch", "transformers", "huggingface-hub", "numpy", "scipy", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    tracked = list((repo_root / "geode/circuits").glob("*.py"))
    tracked += list((repo_root / "experiments/olmo2-circuit-overlap").glob("*.py"))
    hashes = {str(p.relative_to(repo_root)): sha256_file(p) for p in sorted(tracked)}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False
    )
    deployment_path = repo_root / "deployment.json"
    deployment = json.loads(deployment_path.read_text()) if deployment_path.exists() else {}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit.stdout.strip() or deployment.get("git_commit"),
        "deployment": deployment,
        "source_files": hashes,
        "source_fingerprint": fingerprint(hashes),
    }


def start_run(path: Path, config: dict, provenance: dict) -> dict:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing evaluation: {path}")
    path.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": 1,
        "kind": "offline_circuit_eval",
        "status": "running",
        "started_utc": utc_now(),
        "config": config,
        "config_fingerprint": fingerprint(config),
        "provenance": provenance,
    }
    write_json(path / "run_metadata.json", meta)
    return meta


def validate_matched_rows(a: dict, b: dict) -> None:
    for key in ("item_ids", "group_ids", "node_names", "protocol_hash", "tokenizer_hash"):
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f"unmatched comparison field: {key}")


def artifact_upload_files(root: Path) -> list[Path]:
    """Allowlist evaluation outputs; never recursively upload a model/cache or secrets."""
    allowed = {".json", ".jsonl", ".npz", ".parquet", ".png", ".md", ".csv"}
    paths = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink not allowed in artifacts: {path}")
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if any(
            p.startswith(".") or p.lower() in {"model", "models", "cache", "credentials"}
            for p in parts
        ):
            continue
        if path.suffix in allowed:
            paths.append(path)
    return paths
