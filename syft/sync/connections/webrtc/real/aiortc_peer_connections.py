"""``PeerConnections`` over aiortc: a loop thread and one data channel per
peer. It only moves frames."""

import asyncio
import logging
import math
import threading
from collections.abc import Coroutine
from typing import Any

from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCPeerConnection,
    RTCSctpTransport,
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
# The maximum message size that aiortc advertises. For each 1200-byte SCTP fragment
# that arrives, aiortc's InboundStream.pop_messages scans all earlier fragments of
# the message. Therefore the receive time grows with the square of the message size.
MAX_MESSAGE_BYTES = RTCSctpTransport.getCapabilities().maxMessageSize
# The first byte of each chunk tells whether more chunks of the frame follow.
MORE_CHUNKS = b"\x00"
LAST_CHUNK = b"\x01"
CHUNK_PAYLOAD_BYTES = MAX_MESSAGE_BYTES - len(LAST_CHUNK)


def split_frame_into_chunks(frame: bytes) -> list[bytes]:
    """Split ``frame`` into chunks of at most ``MAX_MESSAGE_BYTES`` each.

    Returns:
        The chunks in order. The last chunk starts with ``LAST_CHUNK``, and
        each other chunk starts with ``MORE_CHUNKS``. An empty frame gives one
        chunk that holds only ``LAST_CHUNK``.
    """
    count = max(1, math.ceil(len(frame) / CHUNK_PAYLOAD_BYTES))
    return [
        (LAST_CHUNK if i == count - 1 else MORE_CHUNKS)
        + frame[i * CHUNK_PAYLOAD_BYTES : (i + 1) * CHUNK_PAYLOAD_BYTES]
        for i in range(count)
    ]


class AiortcPeerConnections(PeerConnections):
    """One aiortc peer connection and data channel per peer, and the asyncio
    loop thread that runs them. ``on_open`` and ``on_frame`` run on that
    thread.

    A frame can have any size. ``split_frame_into_chunks`` splits it into
    chunks of at most ``MAX_MESSAGE_BYTES``, and the peer joins the chunks
    again."""

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
        # peer email -> the received chunks of its incomplete frame
        self._partial_frames: dict[str, bytearray] = {}

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
            # The channel is ordered, and this call sends all chunks of the
            # frame. Therefore the chunks of two frames do not mix.
            for chunk in split_frame_into_chunks(frame):
                channel.send(chunk)

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
        # The last chunk of an incomplete frame from the old channel never arrives.
        self._partial_frames[peer] = bytearray()
        channel.on("message", lambda chunk: self._handle_chunk(peer, chunk))

    def _handle_chunk(self, peer: str, chunk: bytes) -> None:
        """Add ``chunk`` to the incomplete frame of ``peer``.

        After the last chunk, give the complete frame to ``on_frame``.

        Raises:
            ValueError: ``chunk`` starts with neither ``MORE_CHUNKS`` nor
                ``LAST_CHUNK``.
        """
        flag, payload = chunk[:1], chunk[1:]
        if flag not in (MORE_CHUNKS, LAST_CHUNK):
            raise ValueError(
                f"A chunk from {peer} must start with {MORE_CHUNKS!r} or "
                f"{LAST_CHUNK!r}, but it starts with {flag!r}"
            )
        self._partial_frames[peer] += payload
        if flag == LAST_CHUNK:
            frame = bytes(self._partial_frames[peer])
            self._partial_frames[peer] = bytearray()
            self._handle_frame(peer, frame)

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
        self._partial_frames.clear()
