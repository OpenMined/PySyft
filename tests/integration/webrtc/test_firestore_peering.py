"""FirestorePeeringConnection on the Firebase emulators."""

import socket
from dataclasses import replace
from pathlib import Path

import pytest

from syft.sync.connections.exceptions import ConnectionUnavailableError
from syft.sync.connections.webrtc.firebase.firestore import FirestoreClient, SetFields
from syft.sync.connections.webrtc.firebase.firestore_peering import (
    PEER_INFO_FIELD,
    FirestorePeeringConfig,
    FirestorePeeringConnection,
    shared_path,
)
from syft.sync.encryption_policy import EncryptionPolicy
from syft.sync.peers.peer import PeerState
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.version.exceptions import UnreadablePeerInfoError
from syft.sync.version.peer_info import PeerInfo
from syft.sync.version.version_info import VersionInfo
from tests.integration.webrtc.firebase_emulators import (
    EMULATOR_HOST,
    EMULATOR_PROJECT,
    firestore_of,
    sign_in,
)

DO = "do@example.com"
DS = "ds@example.com"

pytestmark = pytest.mark.usefixtures("firebase_emulators")


def firestore_peering(email: str, syftbox_folder: Path) -> FirestorePeeringConnection:
    config = FirestorePeeringConfig(email, syftbox_folder)
    return FirestorePeeringConnection(config, firestore_of(email, syftbox_folder))


def test_peering_flow_exchanges_key_bundles():
    do_config, ds_config = SyftboxManager._mock_pair_configs(encryption=True)
    do_connections, ds_connections = SyftboxManager._mock_drive_connections(
        do_config, ds_config
    )
    do_peering = firestore_peering(do_config.email, do_config.syftbox_folder)
    ds_peering = firestore_peering(ds_config.email, ds_config.syftbox_folder)

    ds_manager, do_manager = SyftboxManager._test_pair(
        do_config,
        do_connections.model_copy(update={"peering_con": do_peering}),
        ds_config,
        ds_connections.model_copy(update={"peering_con": ds_peering}),
        add_peers=True,
    )

    do_peer = do_manager.peer_manager.get_cached_peer(ds_manager.email)
    ds_peer = ds_manager.peer_manager.get_cached_peer(do_manager.email)
    assert do_peer.state == ds_peer.state == PeerState.ACCEPTED
    assert do_manager._peer_store.has_peer_bundle(ds_manager.email)
    assert ds_manager._peer_store.has_peer_bundle(do_manager.email)


def test_new_peer_info_reaches_granted_peer(tmp_path):
    do = firestore_peering(DO, tmp_path / "do")
    ds = firestore_peering(DS, tmp_path / "ds")
    old_info = PeerInfo(
        version_info=VersionInfo.current(), encryption_policy=EncryptionPolicy.OPTIONAL
    )
    do.grant_peer_info_read(DS, old_info)
    new_info = old_info.model_copy(
        update={"encryption_policy": EncryptionPolicy.REQUIRED}
    )

    do.publish_peer_info(new_info)

    assert ds.read_peer_info(DO) == new_info


def test_incoming_requests_skip_decided_peers(tmp_path):
    do = firestore_peering(DO, tmp_path / "do")
    firestore_peering(DS, tmp_path / "ds").open_channels(DO)
    assert do.incoming_requests() == [DS]

    do.update_peer_registry(DS, PeerState.REJECTED.value)
    assert do.incoming_requests() == []


def test_unreachable_firestore_raises_connection_unavailable(tmp_path):
    with socket.socket() as unused:
        unused.bind((EMULATOR_HOST, 0))
        port = unused.getsockname()[1]
    project = replace(
        EMULATOR_PROJECT, firestore_url=f"http://{EMULATOR_HOST}:{port}/v1"
    )
    firestore = FirestoreClient(project, sign_in(DO, tmp_path))
    peering = FirestorePeeringConnection(
        FirestorePeeringConfig(DO, tmp_path), firestore
    )

    with pytest.raises(ConnectionUnavailableError):
        peering.incoming_requests()


def test_incoming_request_keeps_peer_list_spelling(tmp_path):
    ds = firestore_peering(DS, tmp_path / "ds")
    ds.update_peer_registry(DO.upper(), PeerState.REQUESTED_BY_ME.value)

    firestore_peering(DO, tmp_path / "do").open_channels(DS)

    assert ds.incoming_requests() == [DO.upper()]


def test_unreadable_peer_info_raises_unreadable_error(tmp_path):
    unreadable = SetFields(shared_path(DO, DS), {PEER_INFO_FIELD: "{not json"})
    firestore_of(DO, tmp_path / "do").write_documents([unreadable])

    with pytest.raises(UnreadablePeerInfoError):
        firestore_peering(DS, tmp_path / "ds").read_peer_info(DO)
