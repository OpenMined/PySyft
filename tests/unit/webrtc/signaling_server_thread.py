"""A signaling server on its own loop thread, for clients that block."""

import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from syft.sync.connections.webrtc.signaling.server import (
    SignalingServer,
    SignalingServerConfig,
)
from syft.sync.connections.webrtc.signaling.signaling import IceServers
from websockets.asyncio.client import connect

ACCESS_KEY = "test-key"
# Time for a close or a refusal to reach a client.
DELIVERY_SECONDS = 1


async def no_ice_servers() -> IceServers:
    # Peers on one machine reach each other with no STUN or TURN server.
    return []


@contextmanager
def signaling_server_in_thread() -> Iterator[str]:
    """Run a signaling server on its own loop thread, and yield its URL."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    config = SignalingServerConfig(host="127.0.0.1", port=0, access_key=ACCESS_KEY)
    running = SignalingServer(config, no_ice_servers).running()
    port = asyncio.run_coroutine_threadsafe(running.__aenter__(), loop).result()
    try:
        yield f"ws://127.0.0.1:{port}"
    finally:
        stop = running.__aexit__(None, None, None)
        asyncio.run_coroutine_threadsafe(stop, loop).result()
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


async def sign_in_and_leave(url: str, email: str) -> None:
    """Sign in as ``email`` and leave. The server ends the earlier session of
    ``email``."""
    async with connect(url) as client:
        hello = {"type": "hello", "email": email, "access_key": ACCESS_KEY}
        await client.send(json.dumps(hello))
        await client.recv()
