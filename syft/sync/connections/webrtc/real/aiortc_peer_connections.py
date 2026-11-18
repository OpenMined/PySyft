"""``PeerConnections`` over aiortc: a loop thread and one data channel per
peer. It only moves frames."""

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

from syft.sync.connections.webrtc.peer_connections import (
    OnFrame,
    OnOpen,
    PeerConnections,
)
from syft.sync.connections.webrtc.signaling.signaling import Signaling

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 10.0
CHANNEL_LABEL = "syft"
# No STUN or TURN server: peers on one machine or LAN reach each other directly.
LOCAL_ONLY = RTCConfiguration(iceServers=[])


class AiortcPeerConnections(PeerConnections):
    """One aiortc peer connection and data channel per peer, and the asyncio
    loop thread that runs them. ``on_open`` and ``on_frame`` run on that
    thread."""

    def __init__(self, signaling: Signaling):
        self.signaling = signaling
        self.email: str | None = None
        self._on_open: OnOpen | None = None
        self._on_frame: OnFrame | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        # peer email -> its aiortc peer connection
        self._rtc_connections: dict[str, RTCPeerConnection] = {}
        # peer email -> its data channel
        self._channels: dict[str, RTCDataChannel] = {}

    def open(self, email: str, on_open: OnOpen, on_frame: OnFrame) -> None:
        """Start the loop thread and answer offers from peers."""
        self.email = email
        self._on_open = on_open
        self._on_frame = on_frame
        self._thread.name = f"webrtc-{email}"
        self._thread.start()
        self._run(self.signaling.listen(email, self._answer))

    def connect(self, peer: str, timeout: float = CONNECT_TIMEOUT_SECONDS) -> None:
        self._run(self._connect(peer), timeout)

    def send(self, peer: str, frame: bytes) -> None:
        """Works from any thread; the frame goes out on the loop thread."""
        self._loop.call_soon_threadsafe(self._send_on_loop, peer, frame)

    def close(self) -> None:
        """Close every peer connection and stop the loop thread."""
        self._run(self._close_rtc_connections())
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

    def _send_on_loop(self, peer: str, frame: bytes) -> None:
        """Dropped when ``peer`` has no open channel; an unconfirmed message
        stays queued and goes again on the next ``on_open``."""
        channel = self._channels.get(peer)
        if channel is not None and channel.readyState == "open":
            channel.send(frame)

    async def _connect(self, peer: str) -> None:
        rtc = await self._new_rtc_connection(peer)
        channel = rtc.createDataChannel(CHANNEL_LABEL)
        self._use_channel(peer, channel)
        opened = asyncio.Event()

        @channel.on("open")
        def on_open() -> None:
            self._on_open(peer)
            opened.set()

        await rtc.setLocalDescription(await rtc.createOffer())
        offer = rtc.localDescription.sdp
        answer = await self.signaling.offer(self.email, peer, offer)
        await rtc.setRemoteDescription(RTCSessionDescription(answer, "answer"))
        await opened.wait()

    async def _answer(self, sender: str, offer: str) -> str:
        rtc = await self._new_rtc_connection(sender)

        # aiortc hands over the channel already open.
        @rtc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> None:
            self._use_channel(sender, channel)
            self._on_open(sender)

        await rtc.setRemoteDescription(RTCSessionDescription(offer, "offer"))
        await rtc.setLocalDescription(await rtc.createAnswer())
        return rtc.localDescription.sdp

    async def _new_rtc_connection(self, peer: str) -> RTCPeerConnection:
        """An aiortc peer connection to ``peer``, in place of any older one."""
        old = self._rtc_connections.pop(peer, None)
        if old is not None:
            await old.close()
        self._rtc_connections[peer] = RTCPeerConnection(LOCAL_ONLY)
        return self._rtc_connections[peer]

    def _use_channel(self, peer: str, channel: RTCDataChannel) -> None:
        self._channels[peer] = channel
        channel.on("message", lambda frame: self._handle_frame(peer, frame))

    def _handle_frame(self, peer: str, frame: bytes) -> None:
        # Logged, not raised: raised here it would end up inside aiortc.
        try:
            self._on_frame(peer, frame)
        except Exception:
            logger.exception("Dropped a WebRTC frame from %s", peer)

    async def _close_rtc_connections(self) -> None:
        await asyncio.gather(*(c.close() for c in self._rtc_connections.values()))
        self._rtc_connections.clear()
        self._channels.clear()
