"""SYFT_peers.json carries a version, and an unreadable peer state is logged.

The file is a flat map of peer email to entry, so a version cannot go at the top
level: every existing client reads a top-level key as an email. The version lives
under a reserved key instead. An older client parses the state of that entry,
fails, and skips it, so the reserved key is invisible to a client that predates
it.

The record itself is safe either way. The only writer is `update_peer_registry`,
which changes one entry of the raw map and writes the rest back, so a peer this
client cannot read is not erased for the other side.
"""

import logging
from unittest.mock import Mock, patch

from syft.sync.connections.drive.gdrive_client import GDriveClient
from syft.sync.connections.drive.gdrive_layout import DriveLayout
from syft.sync.connections.drive.drive_peering import DrivePeering
from syft.sync.peers.peer import PeerState
from syft.sync.platforms.gdrive_files_platform import GdriveFilesPlatform
from syft.sync.transport.naming import PEERS_META_KEY, SYFT_PEERS_VERSION

PEER = "bob@example.com"


def _peering(peers_data):
    layout = DriveLayout(GDriveClient(Mock(), Mock), "alice@example.com")
    peering = DrivePeering(layout)
    peering._peers_json_cache = dict(peers_data)
    return peering


def _router(peering):
    router = Mock()
    router.transport.peering = peering
    router.transport.platform = GdriveFilesPlatform()
    from syft.sync.connections.connection_router import ConnectionRouter

    return ConnectionRouter.get_all_peers_from_json.__get__(router, ConnectionRouter)


def test_a_write_stamps_the_reserved_entry():
    peering = _peering({PEER: {"state": "accepted"}})
    with (
        patch.object(DrivePeering, "_get_peers_file_id", return_value="file-id"),
        patch.object(DriveLayout, "get_syftbox_folder_id", return_value="folder-id"),
        patch.object(GDriveClient, "create_file_payload", return_value=(Mock(), None)),
    ):
        peering._write_peers_json({PEER: {"state": "accepted"}})

    assert peering._peers_json_cache[PEERS_META_KEY] == {"version": SYFT_PEERS_VERSION}
    assert peering._peers_json_cache[PEER] == {"state": "accepted"}


def test_the_reserved_entry_is_not_a_peer():
    peering = _peering(
        {
            PEERS_META_KEY: {"version": SYFT_PEERS_VERSION},
            PEER: {"state": "accepted"},
        }
    )
    peers = _router(peering)()
    assert [p.email for p in peers] == [PEER]


def test_a_known_state_loads():
    peering = _peering({PEER: {"state": "rejected"}})
    peers = _router(peering)()
    assert peers[0].state == PeerState.REJECTED


def test_an_unknown_state_is_skipped_and_logged(caplog):
    peering = _peering({PEER: {"state": "quarantined"}})
    with caplog.at_level(logging.WARNING, logger="syft"):
        peers = _router(peering)()
    assert peers == []
    assert any(PEER in r.getMessage() for r in caplog.records)
    assert any("quarantined" in r.getMessage() for r in caplog.records)


def test_a_file_without_the_reserved_entry_still_loads():
    # Written before the reserved key existed.
    peering = _peering({PEER: {"state": "accepted"}})
    peers = _router(peering)()
    assert [p.email for p in peers] == [PEER]


def test_a_reserved_entry_from_a_newer_client_does_not_stop_the_read(caplog):
    peering = _peering(
        {
            PEERS_META_KEY: {"version": SYFT_PEERS_VERSION + 1},
            PEER: {"state": "accepted"},
        }
    )
    with caplog.at_level(logging.WARNING, logger="syft"):
        peers = _router(peering)()
    assert [p.email for p in peers] == [PEER]
