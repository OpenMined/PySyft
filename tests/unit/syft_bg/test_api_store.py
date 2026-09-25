"""Tests for storing auto-approval objects as apis in SyftBox."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from syft_bg.approve.api_store import ApiStore
from syft_permissions import PERMISSION_FILE_NAME, RuleSet

DO_EMAIL = "do@test.com"


@pytest.fixture
def store(temp_dir):
    with patch("syft_bg.approve.api_store.get_syftbg_dir", return_value=temp_dir):
        yield ApiStore(temp_dir / "syftbox", DO_EMAIL)


@pytest.fixture
def script(temp_dir) -> Path:
    path = temp_dir / "main.py"
    path.write_text("print('hi')\n")
    return path


def _readers(store: ApiStore, name: str) -> list[str]:
    ruleset = RuleSet.load(store.api_dir(name) / PERMISSION_FILE_NAME)
    assert [r.pattern for r in ruleset.rules] == ["**"]
    return ruleset.rules[0].access.read


def test_create_writes_api_folder_with_peer_permissions(store, script, temp_dir):
    name, obj = store.create(
        "analysis", [("run.sh", script)], ["config.yaml"], ["bob@a.com", "al@b.com"]
    )

    api_dir = temp_dir / "syftbox" / DO_EMAIL / "app_data" / "apis" / "analysis"
    assert name == "analysis"
    assert (api_dir / "files" / "run.sh").read_text() == "print('hi')\n"
    stored = yaml.safe_load((api_dir / "api.yaml").read_text())
    assert set(stored["file_contents"][0]) == {"relative_path", "hash"}
    assert obj.file_contents[0].path == str(api_dir / "files" / "run.sh")
    assert obj.file_paths == ["config.yaml"]
    assert _readers(store, "analysis") == ["al@b.com", "bob@a.com"]


def test_no_peers_grants_read_to_everyone(store, script):
    store.create("open", [("run.sh", script)], [], [])
    assert _readers(store, "open") == ["*"]


def test_save_revokes_removed_peers(store, script):
    store.create("analysis", [("run.sh", script)], [], ["a@x.com", "b@x.com"])

    obj = store.get("analysis")
    obj.peers.remove("a@x.com")
    store.save("analysis", obj)

    assert store.get("analysis").peers == ["b@x.com"]
    assert _readers(store, "analysis") == ["b@x.com"]


def test_name_collision_gets_suffix(store, script):
    store.create("analysis", [("run.sh", script)], [], ["a@x.com"])
    name, _ = store.create("analysis", [("run.sh", script)], [], ["a@x.com"])
    assert name == "analysis_1"
    assert store.names() == ["analysis", "analysis_1"]


def test_delete_removes_folder(store, script):
    store.create("analysis", [("run.sh", script)], [], ["a@x.com"])
    assert store.delete("analysis") is True
    assert not store.api_dir("analysis").exists()
    assert store.delete("analysis") is False
