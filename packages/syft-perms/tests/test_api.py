from pathlib import Path

import pytest

import syft_perms
from syft_perms import SyftFile, SyftFolder, SyftPermContext
from syft_perms.api import _default_context
from syft_perms.datasite_utils import (
    _candidate_syftbox_folders,
    _find_datasite,
)


OWNER = "alice@example.com"


@pytest.fixture
def syftbox(tmp_path, monkeypatch):
    """Create a SyftBox folder with a single datasite."""
    datasite = tmp_path / OWNER
    datasite.mkdir()
    monkeypatch.setenv("SYFTBOX_FOLDER", str(tmp_path))
    return tmp_path


# --- _candidate_syftbox_folders ---


def test_env_var_takes_priority(monkeypatch):
    monkeypatch.setenv("SYFTBOX_FOLDER", "/custom/path")
    assert _candidate_syftbox_folders() == [Path("/custom/path")]


def test_fallback_to_home(monkeypatch):
    monkeypatch.delenv("SYFTBOX_FOLDER", raising=False)
    candidates = _candidate_syftbox_folders()
    assert Path.home() / "SyftBox" in candidates


# --- _find_datasite ---


def test_find_single_datasite(tmp_path):
    ds = tmp_path / "user@example.com"
    ds.mkdir()
    assert _find_datasite([tmp_path]) == ds


def test_find_ignores_non_email_dirs(tmp_path):
    (tmp_path / "not-a-datasite").mkdir()
    ds = tmp_path / "user@example.com"
    ds.mkdir()
    assert _find_datasite([tmp_path]) == ds


def test_find_multiple_datasites_raises(tmp_path):
    (tmp_path / "a@example.com").mkdir()
    (tmp_path / "b@example.com").mkdir()
    with pytest.raises(ValueError, match="Multiple datasites"):
        _find_datasite([tmp_path])


def test_find_no_datasite_raises(tmp_path):
    with pytest.raises(ValueError, match="Could not auto-detect"):
        _find_datasite([tmp_path])


def test_find_nonexistent_folder_raises():
    with pytest.raises(ValueError, match="Could not auto-detect"):
        _find_datasite([Path("/nonexistent/path")])


# --- top-level convenience functions ---


def test_open_file(syftbox):
    f = syft_perms.open("data.csv")
    assert isinstance(f, SyftFile)


def test_open_folder(syftbox):
    (syftbox / OWNER / "project").mkdir()
    f = syft_perms.open("project/")
    assert isinstance(f, SyftFolder)


def test_files_returns_browser(syftbox):
    (syftbox / OWNER / "a.txt").touch()
    items = syft_perms.files().all()
    assert len(items) == 1


def test_folders_returns_browser(syftbox):
    (syftbox / OWNER / "subdir").mkdir()
    items = syft_perms.folders().all()
    assert len(items) == 1


def test_files_and_folders_returns_browser(syftbox):
    (syftbox / OWNER / "a.txt").touch()
    (syftbox / OWNER / "subdir").mkdir()
    items = syft_perms.files_and_folders().all()
    assert len(items) == 2


def test_default_context_returns_syft_perms_context(syftbox):
    ctx = _default_context()
    assert isinstance(ctx, SyftPermContext)
    assert ctx.owner == OWNER
