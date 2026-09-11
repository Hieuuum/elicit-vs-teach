"""Offline uploader smoke checks; HfApi never accesses a network."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def upload_module():
    script = Path(__file__).resolve().parents[3] / "experiments/olmo2-circuit-overlap/upload.py"
    spec = importlib.util.spec_from_file_location("circuit_upload_test_module", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def artifacts(tmp_path):
    (tmp_path / "run_metadata.json").write_text('{"status":"complete"}\n')
    (tmp_path / "report.md").write_text("First report.\n")
    return tmp_path


def test_repeat_upload_manifest_hashes_current_content_and_excludes_itself(
    upload_module, artifacts, monkeypatch
):
    commits = []

    class FakeHfApi:
        def repo_info(self, repo_id, *, repo_type):
            assert (repo_id, repo_type) == ("owner/results", "dataset")
            return SimpleNamespace(private=True)

        def create_commit(self, *, repo_id, repo_type, operations, commit_message):
            assert (repo_id, repo_type) == ("owner/results", "dataset")
            assert commit_message == "Add pilot evaluation artifacts"
            uploaded = {
                operation.path_in_repo: Path(operation.path_or_fileobj).read_bytes()
                for operation in operations
            }
            manifest = json.loads(uploaded["pilot/upload_manifest.json"])
            assert "upload_manifest.json" not in manifest
            assert set(uploaded) == {f"pilot/{name}" for name in manifest} | {
                "pilot/upload_manifest.json"
            }
            for name, metadata in manifest.items():
                content = uploaded[f"pilot/{name}"]
                assert metadata == {
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            commits.append(manifest)
            return SimpleNamespace(commit_url=f"https://example.test/commit/{len(commits)}")

    monkeypatch.setattr(upload_module, "HfApi", FakeHfApi)
    assert upload_module.upload(artifacts, "owner/results", "pilot").endswith("/1")
    (artifacts / "report.md").write_text("Updated report with additional pilot checks.\n")
    (artifacts / "extra.json").write_text('{"new":true}\n')
    assert upload_module.upload(artifacts, "owner/results", "pilot").endswith("/2")
    assert commits[0]["report.md"] != commits[1]["report.md"]
    assert "extra.json" not in commits[0] and "extra.json" in commits[1]
    assert json.loads((artifacts / "upload_manifest.json").read_text()) == commits[1]


def test_public_destination_is_rejected_before_creating_commit(
    upload_module, artifacts, monkeypatch
):
    class PublicHfApi:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(private=False)

        def create_commit(self, **_kwargs):
            raise AssertionError("A public destination must never receive artifact content")

    monkeypatch.setattr(upload_module, "HfApi", PublicHfApi)
    with pytest.raises(ValueError, match="destination must be private"):
        upload_module.upload(artifacts, "owner/public-results", "pilot")
