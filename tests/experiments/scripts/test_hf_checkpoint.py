"""Round-trip smoke for the consolidated hub push+verify helpers.

``verify_hub_checkpoint`` is the single implementation the next launcher calls
after ``hf_checkpoint.py push`` to confirm the relay holds the exact bytes it
pushed — the logic launchers used to inline as a Python heredoc. This is a
smoke test, not a property test: it pins the contract that verification passes
on a matching hub sha256 and fails LOUDLY on a mismatched or missing one,
using a fake ``HfApi`` (CPU-only, no network) and a temp store.

``push``'s ``--no-weights`` (added for launch_fig2_llama.sh's push stage,
which archives weights for only 2 of 39 runs) gets its own smoke tests below:
it must exclude ``model.safetensors`` (but ride any ``adapter.safetensors``
LoRA sidecar along, 2026-07-31 — so every run's weights stay cheaply
recoverable) from the upload and — the actual point of the flag — must NOT
require a local checkpoint to exist, since the runs it is for may already
have had their local ``model/`` pruned by the time this push runs. The
default (weights-included) path keeps requiring one.

``pull``'s ``--no-weights`` gets one loud-failure smoke test too: it must
refuse (nonzero exit) rather than print success when the download landed
nothing at all — the fail-open a bad/expired HF token produces silently.
"""

from __future__ import annotations

import fnmatch
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


class _FakeUploadApi:
    """Stand-in for ``HfApi`` during ``push()``: records ``upload_folder``'s
    kwargs instead of hitting the network."""

    def __init__(self) -> None:
        self.upload_calls: list[dict] = []

    def create_repo(self, repo_id: str, private: bool, exist_ok: bool) -> None:
        pass

    def upload_folder(self, **kwargs) -> None:
        self.upload_calls.append(kwargs)


def test_push_no_weights_excludes_model_but_keeps_adapter_without_a_local_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid = "evt-smoke-run"
    run_dir = tmp_path / "runs" / rid
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}")  # no model/ — already pruned locally
    fake = _FakeUploadApi()
    monkeypatch.setattr(hfc, "HfApi", lambda: fake)

    assert hfc.push(tmp_path, rid, "repo/id", public=False, no_weights=True) == 0

    assert len(fake.upload_calls) == 1
    ignore = fake.upload_calls[0]["ignore_patterns"]
    assert ignore is not None
    # Excludes the full state dict but rides any adapter.safetensors sidecar
    # along (2026-07-31): a metadata-scale push still leaves a LoRA run's
    # weights cheaply recoverable.
    assert any(fnmatch.fnmatch("model/model.safetensors", p) for p in ignore)
    assert not any(fnmatch.fnmatch("model/adapter.safetensors", p) for p in ignore)


def test_push_with_weights_uploads_without_excluding_safetensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid = "evt-smoke-run"
    _flat_checkpoint(tmp_path, rid)
    fake = _FakeUploadApi()
    monkeypatch.setattr(hfc, "HfApi", lambda: fake)

    assert hfc.push(tmp_path, rid, "repo/id", public=False) == 0

    ignore = fake.upload_calls[0]["ignore_patterns"]
    assert ignore is not None and "*.safetensors" not in ignore


def test_push_with_weights_fails_loudly_without_a_local_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid = "evt-smoke-run"
    (tmp_path / "runs" / rid).mkdir(parents=True)
    monkeypatch.setattr(hfc, "HfApi", lambda: _FakeUploadApi())

    with pytest.raises(SystemExit):
        hfc.push(tmp_path, rid, "repo/id", public=False)


def test_pull_no_weights_fails_loudly_when_nothing_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad/expired token can make ``snapshot_download`` fetch nothing without
    raising (fail-open); ``pull --no-weights`` must refuse loudly rather than
    print its success line over an empty run dir."""
    rid = "evt-smoke-run"
    monkeypatch.setattr(hfc, "snapshot_download", lambda **kwargs: tmp_path)

    with pytest.raises(SystemExit, match="auth"):
        hfc.pull(tmp_path, rid, "repo/id", no_weights=True)
