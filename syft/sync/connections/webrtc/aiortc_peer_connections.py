"""The aiortc layer under the WebRTC core: a loop thread and one data channel
per peer. It only moves frames; the core does everything else."""

import asyncio
import logging
import threading
from collections.abc import Coroutine
from typing import Any

from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCPeerConnection,
    RTCSessionDescription,
)

from syft.sync.connections.webrtc.channel import Channel
from syft.sync.connections.webrtc.signaling import Signaling
from syft.sync.connections.webrtc.webrtc_p2p import WebRTCP2PConnection

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 10.0
CHANNEL_LABEL = "syft"
# No STUN or TURN server: peers on one machine or LAN reach each other directly.
LOCAL_ONLY = RTCConfiguration(iceServers=[])


class AiortcChannel(Channel):
    """An aiortc data channel. ``send`` works from any thread; the frame goes
    out on the loop thread."""

    def __init__(self, channel: RTCDataChannel, loop: asyncio.AbstractEventLoop):
        self._channel = channel
        self._loop = loop

    def send(self, frame: bytes) -> None:
        self._loop.call_soon_threadsafe(self._channel.send, frame)


class AiortcPeerConnections:
    """One core's aiortc peer connections, one data channel per peer, and the
    asyncio loop thread that runs them. Frames reach the core's ``on_frame``
    on that thread."""

    def __init__(self, core: WebRTCP2PConnection, signaling: Signaling):
        self.core = core
        self.signaling = signaling
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name=f"webrtc-{core.email}", daemon=True
        )
        # peer email -> its peer connection
        self._connections: dict[str, RTCPeerConnection] = {}

    def open(self) -> None:
        """Start the loop thread and answer offers from peers."""
        self._thread.start()
        self._run(self.signaling.listen(self.core.email, self._answer))

    def connect(self, peer: str, timeout: float = CONNECT_TIMEOUT_SECONDS) -> None:
        """Connect to ``peer``; return once the channel is open."""
        self._run(self._connect(peer), timeout)

    def close(self) -> None:
        """Close every peer connection and stop the loop thread."""
        self._run(self._close_connections())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    def _run(self, coroutine: Coroutine[Any, Any, Any], timeout: float | None = None):
        """Run ``coroutine`` on the loop thread and wait for its result."""
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout)
        finally:
            future.cancel()

    async def _connect(self, peer: str) -> None:
        connection = await self._new_connection(peer)
        channel = connection.createDataChannel(CHANNEL_LABEL)
        self._attach(peer, channel)
        opened = asyncio.Event()

        @channel.on("open")
        def on_open() -> None:
            self.core.on_open(peer)
            opened.set()

        await connection.setLocalDescription(await connection.createOffer())
        offer = connection.localDescription.sdp
        answer = await self.signaling.offer(self.core.email, peer, offer)
        await connection.setRemoteDescription(RTCSessionDescription(answer, "answer"))
        await opened.wait()

    async def _answer(self, sender: str, offer: str) -> str:
        connection = await self._new_connection(sender)

        # aiortc hands over the channel already open.
        @connection.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> None:
            self._attach(sender, channel)
            self.core.on_open(sender)

        await connection.setRemoteDescription(RTCSessionDescription(offer, "offer"))
        await connection.setLocalDescription(await connection.createAnswer())
        return connection.localDescription.sdp

    async def _new_connection(self, peer: str) -> RTCPeerConnection:
        """A peer connection to ``peer``, in place of any older one."""
        old = self._connections.pop(peer, None)
        if old is not None:
            await old.close()
        self._connections[peer] = RTCPeerConnection(LOCAL_ONLY)
        return self._connections[peer]

    def _attach(self, peer: str, channel: RTCDataChannel) -> None:
        self.core.attach(peer, AiortcChannel(channel, self._loop))
        channel.on("message", lambda frame: self._on_frame(peer, frame))

    def _on_frame(self, peer: str, frame: bytes) -> None:
        # Logged, not raised: raised here it would end up inside aiortc.
        try:
            self.core.on_frame(peer, frame)
        except Exception:
            logger.exception("Dropped a WebRTC frame from %s", peer)

    async def _close_connections(self) -> None:
        await asyncio.gather(*(c.close() for c in self._connections.values()))
        self._connections.clear()
