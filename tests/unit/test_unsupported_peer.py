"""A peer or a SyftBox of a release whose formats this one does not read.

Such a version file cannot be read, so the peer is refused with that reason
instead of the "not approved yet" one, peering still completes, and a login on
an old SyftBox stops with a way out instead of crashing later.
"""

import json
from pathlib import Path

import pytest
from syft.sync import login_utils
from syft.sync.connections.drive.drive_peering import DrivePeeringConnection
from syft.sync.connections.drive.gdrive_client import GDriveClient
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.version.exceptions import UnreadablePeerInfoError, VersionUnknownError
from syft.sync.version.peer_manager import CompatAction
from syft.sync.version.version_info import CompatibilityStatus
from syft.version import VERSION_FILE_NAME

# The version file of a release whose PeerInfo is a version this one lacks.
EARLIER_RELEASE_FILE = json.dumps(
    {
        "canonical_name": "PeerInfo",
        "version": "1",
        "version_info": {"canonical_name": "VersionInfo", "version": "2"},
        "encryption_policy": "optional",
    }
).encode()


def _returns(data: bytes):
    def download_file(self, file_id: str) -> bytes:
        return data

    return download_file


def test_a_peer_of_an_earlier_release_is_refused_with_the_reason(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        check_versions=True
    )
    peer_manager = ds_manager.peer_manager
    monkeypatch.setattr(GDriveClient, "download_file", _returns(EARLIER_RELEASE_FILE))

    with pytest.warns(UserWarning, match="does not support"):
        infos = peer_manager.load_peer_infos_parallel([do_manager.email])
    assert infos == {do_manager.email: None}

    result = peer_manager.get_peer_compatibility_status(
        do_manager.email, CompatAction.SUBMIT
    )
    assert result.status == CompatibilityStatus.UNKNOWN
    assert result.should_skip
    assert "does not support" in result.unknown_reason
    with pytest.raises(VersionUnknownError, match="does not support"):
        result.raise_on_skip("submit job")


def test_a_readable_file_clears_the_reason(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    peer_manager = ds_manager.peer_manager
    with monkeypatch.context() as patch:
        patch.setattr(GDriveClient, "download_file", _returns(EARLIER_RELEASE_FILE))
        with pytest.warns(UserWarning):
            peer_manager.load_peer_infos_parallel([do_manager.email])

    peer_manager.load_peer_infos_parallel([do_manager.email])

    result = peer_manager.get_peer_compatibility_status(
        do_manager.email, CompatAction.SUBMIT
    )
    assert result.status == CompatibilityStatus.SAME
    assert result.unknown_reason is None


def test_peering_with_an_unsupported_peer_completes(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        add_peers=False, check_versions=True
    )

    def unreadable(self, peer: str):
        raise UnreadablePeerInfoError("Could not read the peer info file")

    monkeypatch.setattr(DrivePeeringConnection, "read_peer_info", unreadable)

    with pytest.warns(UserWarning, match="does not support"):
        ds_manager.add_peer(do_manager.email, sync=False)

    peer = ds_manager.peer_manager.get_cached_peer(do_manager.email)
    assert peer is not None and peer.info is None
    result = ds_manager.peer_manager.get_peer_compatibility_status(
        do_manager.email, CompatAction.SYNC
    )
    assert result.should_skip and "does not support" in result.unknown_reason


@pytest.fixture
def login_without_drive(monkeypatch, tmp_path):
    """Login helpers pointed at a local folder, with no Drive and no terminal."""
    syftbox = tmp_path / "SyftBox_do@test.org"
    monkeypatch.setattr(login_utils, "_get_default_syftbox_path", lambda email: syftbox)
    monkeypatch.setattr(login_utils, "_resolve_token_path", lambda token_path: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    return syftbox


def test_login_on_a_syftbox_of_an_unsupported_release_stops(
    monkeypatch, login_without_drive
):
    def unreadable(email: str, token_path: Path | None):
        raise UnreadablePeerInfoError("Could not read the own peer info file")

    monkeypatch.setattr(login_utils, "_read_remote_version", unreadable)

    with pytest.raises(UnreadablePeerInfoError, match="start fresh"):
        login_utils.handle_potential_version_mismatches_on_login("do@test.org")


def test_login_on_a_local_syftbox_of_an_unsupported_release_stops(
    monkeypatch, login_without_drive
):
    monkeypatch.setattr(login_utils, "_read_remote_version", lambda e, t: None)
    login_without_drive.mkdir(parents=True)
    (login_without_drive / VERSION_FILE_NAME).write_text(EARLIER_RELEASE_FILE.decode())

    with pytest.raises(UnreadablePeerInfoError, match="start fresh"):
        login_utils.handle_potential_version_mismatches_on_login("do@test.org")
