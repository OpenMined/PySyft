"""Two AiortcPeerConnections that find each other through a signaling server
in this process."""

import queue
import threading

import pytest

from syft.sync.connections.webrtc.real.aiortc_peer_connections import (
    AiortcPeerConnections,
)
from syft.sync.connections.webrtc.signaling.server_signaling import (
    ServerSignaling,
    ServerSignalingConfig,
)
from tests.unit.webrtc.signaling_server_thread import (
    ACCESS_KEY,
    signaling_server_in_thread,
)

DO = "do@example.com"
DS = "ds@example.com"
TIMEOUT_SECONDS = 10


@pytest.fixture
def server_url():
    with signaling_server_in_thread() as url:
        yield url


def test_watched_peers_connect_when_online(server_url):
    config = ServerSignalingConfig(server_url, ACCESS_KEY)
    do = AiortcPeerConnections(ServerSignaling(config))
    ds = AiortcPeerConnections(ServerSignaling(config))
    do_opened, ds_opened = threading.Event(), threading.Event()
    frames_on_do: queue.Queue[tuple[str, bytes]] = queue.Queue()
    do.open(DO, lambda peer: do_opened.set(), lambda *frame: frames_on_do.put(frame))
    ds.open(DS, lambda peer: ds_opened.set(), lambda peer, frame: None)
    try:
        do.watch([DS])
        ds.watch([DO])

        assert do_opened.wait(TIMEOUT_SECONDS) and ds_opened.wait(TIMEOUT_SECONDS)
        ds.send(DO, b"hello")
        assert frames_on_do.get(timeout=TIMEOUT_SECONDS) == (DS, b"hello")
    finally:
        do.close()
        ds.close()
