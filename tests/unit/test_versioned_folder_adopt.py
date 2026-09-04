"""A client adopts a private Drive folder from an earlier client version.

A private folder name holds the client version. After a minor upgrade the name of
the current version does not exist yet. Without adoption the client creates a new
folder, and the datasite of the user stays on Drive out of reach.

These tests cover the private folders only. The name of a P2P folder is a
rendezvous string that both peers compute, so a client must never rename one.
"""

from unittest.mock import Mock

import pytest

from syft.sync.connections.drive.gdrive_transport import (
    GDriveConnection,
    _partition_by_version,
)

EMAIL = "alice@example.com"


def _conn():
    conn = GDriveConnection(email=EMAIL, verbose=False)
    conn.drive_service = Mock()
    return conn


def _renames(conn):
    """Return the (fileId, new name) pairs the connection sent to Drive."""
    return [
        (kwargs["fileId"], kwargs["body"]["name"])
        for _, kwargs in conn.drive_service.files().update.call_args_list
        if "body" in kwargs and "name" in kwargs.get("body", {})
    ]


# ---------- _partition_by_version -------------------------------------------


def test_partition_splits_compatible_older_and_newer():
    folders = [
        ("old", f"0.1.9#{EMAIL}"),
        ("same", f"0.2.5#{EMAIL}"),
        ("new", f"0.3.0#{EMAIL}"),
    ]
    compatible, older, newer = _partition_by_version(folders, current_version="0.2.7")
    assert compatible == [("same", f"0.2.5#{EMAIL}")]
    assert older == [("old", f"0.1.9#{EMAIL}")]
    assert newer == [("new", f"0.3.0#{EMAIL}")]


def test_partition_sorts_by_number_not_by_string():
    folders = [("a", f"0.1.9#{EMAIL}"), ("b", f"0.1.10#{EMAIL}")]
    _, older, _ = _partition_by_version(folders, current_version="0.2.0")
    assert [fid for fid, _ in older] == ["a", "b"]


def test_partition_drops_names_without_a_version():
    folders = [("a", f"0.1.9#{EMAIL}"), ("b", "no_version_here")]
    _, older, _ = _partition_by_version(folders, current_version="0.2.0")
    assert older == [("a", f"0.1.9#{EMAIL}")]


def test_partition_returns_empty_for_a_bad_current_version():
    folders = [("a", f"0.1.9#{EMAIL}")]
    assert _partition_by_version(folders, current_version="garbage") == ([], [], [])


# ---------- adoption --------------------------------------------------------


def test_a_compatible_folder_wins_and_nothing_is_renamed():
    conn = _conn()
    folders = [("same", f"0.2.5#{EMAIL}"), ("old", f"0.1.9#{EMAIL}")]
    got = conn._find_or_adopt_versioned_folder(
        folders, current_name=f"0.2.7#{EMAIL}", current_version="0.2.7"
    )
    assert got == "same"
    assert _renames(conn) == []


def test_an_older_folder_is_adopted_by_rename():
    conn = _conn()
    folders = [("old", f"0.1.9#{EMAIL}")]
    got = conn._find_or_adopt_versioned_folder(
        folders, current_name=f"0.2.7#{EMAIL}", current_version="0.2.7"
    )
    assert got == "old", "the client must keep the folder that holds the data"
    assert _renames(conn) == [("old", f"0.2.7#{EMAIL}")]


def test_the_highest_older_folder_is_adopted():
    conn = _conn()
    folders = [
        ("v1", f"0.1.9#{EMAIL}"),
        ("v2", f"0.1.20#{EMAIL}"),
        ("v0", f"0.0.4#{EMAIL}"),
    ]
    got = conn._find_or_adopt_versioned_folder(
        folders, current_name=f"0.2.7#{EMAIL}", current_version="0.2.7"
    )
    assert got == "v2"
    assert _renames(conn) == [("v2", f"0.2.7#{EMAIL}")]


def test_a_newer_folder_stops_the_client():
    # A new folder here would hide data that this client cannot read. Report the
    # version to install instead.
    conn = _conn()
    folders = [("new", f"0.3.0#{EMAIL}")]
    with pytest.raises(RuntimeError, match="0.3.0"):
        conn._find_or_adopt_versioned_folder(
            folders, current_name=f"0.2.7#{EMAIL}", current_version="0.2.7"
        )
    assert _renames(conn) == []


def test_no_folder_returns_none_so_the_caller_creates_one():
    conn = _conn()
    got = conn._find_or_adopt_versioned_folder(
        [], current_name=f"0.2.7#{EMAIL}", current_version="0.2.7"
    )
    assert got is None
    assert _renames(conn) == []


def test_two_compatible_folders_still_raise():
    conn = _conn()
    folders = [("a", f"0.2.1#{EMAIL}"), ("b", f"0.2.2#{EMAIL}")]
    with pytest.raises(RuntimeError):
        conn._find_or_adopt_versioned_folder(
            folders, current_name=f"0.2.7#{EMAIL}", current_version="0.2.7"
        )
