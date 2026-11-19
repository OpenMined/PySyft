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
    RTCIceServer,
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

    A DS sends the offers. A DO only answers, and only the offers of its peers.

    A frame can have any size. ``split_frame_into_chunks`` splits it into
    chunks of at most ``MAX_MESSAGE_BYTES``, and the peer joins the chunks
    again."""

    # open() sets the email. The sign-in sets the RTC configuration, with the
    # ICE servers that the signaling gives.
    email: str
    _rtc_configuration: RTCConfiguration

    def __init__(self, signaling: Signaling, sends_offers: bool):
        self.signaling = signaling
        self._sends_offers = sends_offers
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
        # The emails, in lower case, of the peers of this client.
        self._peers: set[str] = set()
        # peer email -> the task that connects to it after its online event,
        # until the task ends. This dict keeps a strong reference to each task,
        # because asyncio keeps only a weak one.
        self._connect_tasks: dict[str, asyncio.Task] = {}
        # The sign-in to the signaling. None until the first set_peers or connect.
        self._sign_in_task: asyncio.Task | None = None

    def open(self, email: str, on_open: OnOpen, on_frame: OnFrame) -> None:
        """Start the loop thread.

        The client signs in to the signaling at the first ``set_peers`` or
        ``connect``. Thus no peer sends it an offer before it knows its peers.
        """
        self.email = email
        self._on_open = on_open
        self._on_frame = on_frame
        self._thread.name = f"webrtc-{email}"
        self._thread.start()

    def connect(self, peer: str, timeout: float = CONNECT_TIMEOUT_SECONDS) -> None:
        self._run(self._connect(peer), timeout)

    def set_peers(self, peers: list[str]) -> None:
        """Set the peers of this client.

        A client that sends offers connects to each peer when the peer is
        online, now and each time the peer comes online again. A client that
        only answers refuses an offer from an email that is not in ``peers``.
        The channel to each email that is not in ``peers`` closes.
        """
        self._run(self._set_peers(peers))

    def send(self, peer: str, frame: bytes) -> None:
        """Works from any thread; the frame goes out on the loop thread."""
        self._loop.call_soon_threadsafe(self._send_on_loop, peer, frame)

    def close(self) -> None:
        """Close the signaling and each peer connection, then stop the loop."""
        self._run(self._close_signaling_and_rtc_connections())
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
        if self._has_open_channel(peer):
            # The channel is ordered, and this call sends all chunks of the
            # frame. Therefore the chunks of two frames do not mix.
            for chunk in split_frame_into_chunks(frame):
                self._channels[peer].send(chunk)

    def _has_open_channel(self, peer: str) -> bool:
        channel = self._channels.get(peer)
        return channel is not None and channel.readyState == "open"

    async def _set_peers(self, peers: list[str]) -> None:
        self._peers = {peer.lower() for peer in peers}
        await self._close_rtc_connections_of_other_emails()
        await self._maybe_sign_in()
        if self._sends_offers:
            await self.signaling.watch(self.email, peers)

    def _on_peer_online(self, peer: str) -> None:
        # The server reports a watched peer again at each set_peers. A peer
        # with a channel, or with a connection on the way, gets no new one.
        if (
            peer.lower() in self._peers
            and not self._has_open_channel(peer)
            and peer not in self._connect_tasks
        ):
            task = asyncio.create_task(self._connect_in_background(peer))
            self._connect_tasks[peer] = task
            task.add_done_callback(lambda _: self._connect_tasks.pop(peer))

    async def _connect_in_background(self, peer: str) -> None:
        try:
            await asyncio.wait_for(self._connect(peer), CONNECT_TIMEOUT_SECONDS)
        except (ConnectionError, TimeoutError) as err:
            # The next online event of the peer starts a new connection.
            logger.warning(f"Could not connect to {peer}: {err!r}")

    async def _close_rtc_connections_of_other_emails(self) -> None:
        """Close the peer connection to each email that is not a peer."""
        others = [
            peer for peer in self._rtc_connections if peer.lower() not in self._peers
        ]
        for peer in others:
            if peer in self._connect_tasks:
                self._connect_tasks[peer].cancel()
            self._channels.pop(peer, None)
            self._partial_frames.pop(peer, None)
            await self._rtc_connections.pop(peer).close()

    async def _maybe_sign_in(self) -> None:
        """Sign in to the signaling, if this client did not sign in yet."""
        # A task, so that two calls at the same time share one sign-in.
        if self._sign_in_task is None:
            self._sign_in_task = asyncio.ensure_future(self._listen())
        await self._sign_in_task

    async def _listen(self) -> None:
        ice_servers = await self.signaling.listen(
            self.email, self._answer, self._on_peer_online
        )
        self._rtc_configuration = RTCConfiguration(
            iceServers=[RTCIceServer(**server) for server in ice_servers]
        )

    async def _connect(self, peer: str) -> None:
        await self._maybe_sign_in()
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
        """Answer the offer of ``sender``, and return the answer.

        Raises:
            ConnectionRefusedError: ``sender`` is not a peer of this client.
        """
        if sender.lower() not in self._peers:
            raise ConnectionRefusedError(f"{sender} is not a peer of {self.email}")
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
        self._rtc_connections[peer] = RTCPeerConnection(self._rtc_configuration)
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

    async def _close_signaling_and_rtc_connections(self) -> None:
        for task in self._connect_tasks.values():
            task.cancel()
        if self._sign_in_task is not None:
            await self.signaling.close(self.email)
        await asyncio.gather(*(c.close() for c in self._rtc_connections.values()))
        self._rtc_connections.clear()
        self._channels.clear()
        self._partial_frames.clear()
