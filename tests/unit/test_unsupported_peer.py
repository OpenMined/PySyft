"""A peer of a release whose formats this one does not read.

Such a version file cannot be read, so the peer is refused with that reason
instead of the "not approved yet" one, and peering still completes.
"""

import pytest
from syft.sync.connections.drive.drive_peering import DrivePeeringConnection
from syft.sync.connections.drive.gdrive_client import GDriveClient
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.version.exceptions import UnreadablePeerInfoError, VersionUnknownError
from syft.sync.version.peer_manager import CompatAction
from syft.sync.version.version_info import CompatibilityStatus
from tests.unit.utils import EARLIER_RELEASE_FILE


def _download_file_returning(data: bytes):
    def download_file(self, file_id: str) -> bytes:
        return data

    return download_file


def test_earlier_release_peer_refused_with_reason(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        check_versions=True
    )
    peer_manager = ds_manager.peer_manager
    monkeypatch.setattr(
        GDriveClient, "download_file", _download_file_returning(EARLIER_RELEASE_FILE)
    )

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


def test_readable_file_clears_reason(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    peer_manager = ds_manager.peer_manager
    with monkeypatch.context() as patch:
        patch.setattr(
            GDriveClient,
            "download_file",
            _download_file_returning(EARLIER_RELEASE_FILE),
        )
        with pytest.warns(UserWarning):
            peer_manager.load_peer_infos_parallel([do_manager.email])

    peer_manager.load_peer_infos_parallel([do_manager.email])

    result = peer_manager.get_peer_compatibility_status(
        do_manager.email, CompatAction.SUBMIT
    )
    assert result.status == CompatibilityStatus.SAME
    assert result.unknown_reason is None


def test_peering_with_unsupported_peer_completes(monkeypatch):
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
