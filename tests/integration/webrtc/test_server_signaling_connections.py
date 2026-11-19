"""Two AiortcPeerConnections that find each other through a signaling server
in this process."""

import asyncio
import queue
import threading

import pytest

from syft.sync.connections.webrtc.real.aiortc_peer_connections import (
    AiortcPeerConnections,
)
from syft.sync.connections.webrtc.signaling.server import (
    SignalingServer,
    SignalingServerConfig,
)
from syft.sync.connections.webrtc.signaling.server_signaling import (
    ServerSignaling,
    ServerSignalingConfig,
)
from syft.sync.connections.webrtc.signaling.signaling import IceServers

DO = "do@example.com"
DS = "ds@example.com"
ACCESS_KEY = "test-key"
TIMEOUT_SECONDS = 10


async def no_ice_servers() -> IceServers:
    # Peers on one machine reach each other with no STUN or TURN server.
    return []


@pytest.fixture
def server_url():
    """The URL of a signaling server that runs on its own loop thread."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    config = SignalingServerConfig(host="127.0.0.1", port=0, access_key=ACCESS_KEY)
    running = SignalingServer(config, no_ice_servers).running()
    port = asyncio.run_coroutine_threadsafe(running.__aenter__(), loop).result()
    yield f"ws://127.0.0.1:{port}"
    asyncio.run_coroutine_threadsafe(running.__aexit__(None, None, None), loop).result()
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    loop.close()


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
