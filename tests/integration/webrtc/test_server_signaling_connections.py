"""Two AiortcPeerConnections that find each other through a signaling server
in this process."""

import queue
import threading
import time

import pytest

from syft.sync.connections.webrtc.real.aiortc_peer_connections import (
    AiortcPeerConnections,
)
from syft.sync.connections.webrtc.signaling.server_signaling import ServerSignaling
from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig
from tests.unit.webrtc.signaling_server_thread import (
    ACCESS_KEY,
    signaling_server_in_thread,
)

DO = "do@example.com"
DS = "ds@example.com"
TIMEOUT_SECONDS = 10
# A login opens the connections, then loads its peers and sets them.
LOGIN_GAP_SECONDS = 1


@pytest.fixture
def server_url():
    with signaling_server_in_thread() as url:
        yield url


def test_ds_connects_to_do_that_accepts_it(server_url):
    config = WebRTCConfig(signaling_url=server_url, access_key=ACCESS_KEY)
    do = AiortcPeerConnections(ServerSignaling(config), sends_offers=False)
    ds = AiortcPeerConnections(ServerSignaling(config), sends_offers=True)
    do_opened, ds_opened = threading.Event(), threading.Event()
    frames_on_do: queue.Queue[tuple[str, bytes]] = queue.Queue()
    do.open(DO, lambda peer: do_opened.set(), lambda *frame: frames_on_do.put(frame))
    ds.open(DS, lambda peer: ds_opened.set(), lambda peer, frame: None)
    try:
        do.set_peers([DS])
        ds.set_peers([DO])

        assert do_opened.wait(TIMEOUT_SECONDS) and ds_opened.wait(TIMEOUT_SECONDS)
        ds.send(DO, b"hello")
        assert frames_on_do.get(timeout=TIMEOUT_SECONDS) == (DS, b"hello")
    finally:
        do.close()
        ds.close()


def test_ds_connects_to_do_that_signs_in_after_it(server_url):
    config = WebRTCConfig(signaling_url=server_url, access_key=ACCESS_KEY)
    do = AiortcPeerConnections(ServerSignaling(config), sends_offers=False)
    ds = AiortcPeerConnections(ServerSignaling(config), sends_offers=True)
    ds_opened = threading.Event()
    ds.open(DS, lambda peer: ds_opened.set(), lambda peer, frame: None)
    ds.set_peers([DO])
    do.open(DO, lambda peer: None, lambda peer, frame: None)
    try:
        time.sleep(LOGIN_GAP_SECONDS)
        do.set_peers([DS])

        assert ds_opened.wait(TIMEOUT_SECONDS)
    finally:
        do.close()
        ds.close()
