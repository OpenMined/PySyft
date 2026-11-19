"""ServerPeeringConnection against a signaling server on its own thread."""

import socket
from pathlib import Path

import pytest

from syft.sync.connections.exceptions import ConnectionUnavailableError
from syft.sync.connections.webrtc.server_peering import (
    ServerPeeringConfig,
    ServerPeeringConnection,
)
from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig
from syft.sync.peers.peer import PeerState
from syft.sync.syftbox_manager import SyftboxManager
from tests.unit.webrtc.signaling_server_thread import (
    ACCESS_KEY,
    signaling_server_in_thread,
)

DO = "do@example.com"
DS = "ds@example.com"


@pytest.fixture
def server_url():
    with signaling_server_in_thread() as url:
        yield url


def server_peering(
    url: str, email: str, syftbox_folder: Path
) -> ServerPeeringConnection:
    server = WebRTCConfig(signaling_url=url, access_key=ACCESS_KEY)
    return ServerPeeringConnection(ServerPeeringConfig(server, email, syftbox_folder))


def test_peering_flow_exchanges_key_bundles(server_url):
    do_config, ds_config = SyftboxManager._mock_pair_configs(encryption=True)
    do_connections, ds_connections = SyftboxManager._mock_drive_connections(
        do_config, ds_config
    )
    do_peering = server_peering(server_url, do_config.email, do_config.syftbox_folder)
    ds_peering = server_peering(server_url, ds_config.email, ds_config.syftbox_folder)

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


def test_incoming_requests_skip_decided_peers(server_url, tmp_path):
    do = server_peering(server_url, DO, tmp_path / "do")
    server_peering(server_url, DS, tmp_path / "ds").open_channels(DO)
    assert do.incoming_requests() == [DS]

    do.update_peer_registry(DS, PeerState.REJECTED.value)
    assert do.incoming_requests() == []


def test_unreachable_server_raises_connection_unavailable(tmp_path):
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
    peering = server_peering(f"ws://127.0.0.1:{port}", DO, tmp_path)

    with pytest.raises(ConnectionUnavailableError):
        peering.incoming_requests()
