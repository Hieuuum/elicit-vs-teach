"""Round-trip smoke for the consolidated hub push+verify helpers.

``verify_hub_checkpoint`` is the single implementation the next launcher calls
after ``hf_checkpoint.py push`` to confirm the relay holds the exact bytes it
pushed — the logic launchers used to inline as a Python heredoc. This is a
smoke test, not a property test: it pins the contract that verification passes
on a matching hub sha256 and fails LOUDLY on a mismatched or missing one,
using a fake ``HfApi`` (CPU-only, no network) and a temp store.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._scriptloader import load

hfc = load("hf_checkpoint")


class _FakeLfs:
    def __init__(self, sha: str) -> None:
        self.sha256 = sha


class _FakePathInfo:
    def __init__(self, sha: str | None) -> None:
        self.lfs = _FakeLfs(sha) if sha is not None else None


class _FakeApi:
    """Stand-in for ``huggingface_hub.HfApi``: no network, one canned hash."""

    def __init__(self, remote_sha: str | None) -> None:
        self._remote_sha = remote_sha

    def get_paths_info(self, repo_id: str, paths: list[str]) -> list[_FakePathInfo]:
        return [_FakePathInfo(self._remote_sha)]


def _fake_hub(monkeypatch: pytest.MonkeyPatch, remote_sha: str | None) -> None:
    monkeypatch.setattr(hfc, "HfApi", lambda: _FakeApi(remote_sha))


def _flat_checkpoint(store: Path, run_id: str) -> str:
    """Write the flat-layout checkpoint and return its local sha256."""
    ckpt = store / "runs" / run_id / "model" / "model.safetensors"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"fake safetensors payload")
    return hfc.sha256_of(ckpt)


def test_verify_passes_on_matching_hub_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rid = "evt-smoke-run"
    local = _flat_checkpoint(tmp_path, rid)
    _fake_hub(monkeypatch, local)
    assert hfc.verify_hub_checkpoint(tmp_path, rid) == local


def test_verify_raises_on_mismatched_hub_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid = "evt-smoke-run"
    _flat_checkpoint(tmp_path, rid)
    _fake_hub(monkeypatch, "0" * 64)
    with pytest.raises(SystemExit):
        hfc.verify_hub_checkpoint(tmp_path, rid)


def test_verify_raises_when_hub_reports_no_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid = "evt-smoke-run"
    _flat_checkpoint(tmp_path, rid)
    _fake_hub(monkeypatch, None)  # strict: a missing hub hash is a failure, not a shrug
    with pytest.raises(SystemExit):
        hfc.verify_hub_checkpoint(tmp_path, rid)


def test_hub_lfs_sha256_reads_the_stored_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_hub(monkeypatch, "abc123")
    assert hfc.hub_lfs_sha256("repo/id", "runs/x/model/model.safetensors") == "abc123"
    _fake_hub(monkeypatch, None)
    assert hfc.hub_lfs_sha256("repo/id", "runs/x/model/model.safetensors") is None
