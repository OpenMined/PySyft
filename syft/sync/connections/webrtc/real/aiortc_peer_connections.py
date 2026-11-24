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
from syft.sync.connections.webrtc.signaling.signaling import IceServers, Signaling

logger = logging.getLogger(__name__)

# The round trips of the signaling and the TURN request take seconds each.
CONNECT_TIMEOUT_SECONDS = 20.0
CHANNEL_LABEL = "syft"
# The offering side opens a new peer connection after this part of the
# lifetime of its TURN credentials.
CREDENTIALS_REFRESH_FRACTION = 0.9
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


def sign_in_failed(sign_in: asyncio.Task) -> bool:
    """Return whether ``sign_in`` ended without a session."""
    return sign_in.done() and (sign_in.cancelled() or sign_in.exception() is not None)


class AiortcPeerConnections(PeerConnections):
    """One aiortc peer connection and data channel per peer, and the asyncio
    loop thread that runs them. ``on_open`` and ``on_frame`` run on that
    thread.

    Of two peers, the one with the lower email sends the offer. Each side says
    hello to its peers when it signs in, and answers only the offers of its
    peers. The offering side opens a new peer connection before its TURN
    credentials expire. A failure in the background, such as a connect that
    fails or a signaling session that ends, waits until
    ``maybe_raise_background_errors`` raises it.

    A frame can have any size. ``split_frame_into_chunks`` splits it into
    chunks of at most ``MAX_MESSAGE_BYTES``, and the peer joins the chunks
    again."""

    # open() sets it.
    email: str

    def __init__(self, signaling: Signaling):
        self.signaling = signaling
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
        # peer email in lower case -> the peer email as set_peers spells it.
        # The signaling reports each email as its owner spells it.
        self._peers: dict[str, str] = {}
        # peer email -> the task that connects to it after its hello, until the
        # task ends. This dict keeps a strong reference to each task, because
        # asyncio keeps only a weak one.
        self._connect_tasks: dict[str, asyncio.Task] = {}
        # peer email -> the task that opens a new connection to it before the
        # TURN credentials expire
        self._refresh_tasks: dict[str, asyncio.Task] = {}
        # peer email in lower case -> the session of its last hello
        self._peer_sessions: dict[str, str] = {}
        # The tasks that say hello, until they end.
        self._hello_tasks: set[asyncio.Task] = set()
        # The sign-in to the signaling. None until the first set_peers or
        # connect, and again after the session ends.
        self._sign_in_task: asyncio.Task | None = None
        # The failures in the background that no caller saw yet.
        self._background_errors: list[ConnectionError] = []

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
        """Set the peers of this client, and say hello to each peer with no open
        channel.

        A peer that is online answers the hello, and the two connect. Thus each
        call tries again to connect to the peers that are not connected. This
        client refuses an offer from an email that is not in ``peers``. The
        channel to each email that is not in ``peers`` closes.
        """
        self._run(self._set_peers(peers))

    def maybe_raise_background_errors(self) -> None:
        """Raise the failures in the background since the last call, if there
        are any.

        Raises:
            ConnectionError: A connect or a hello failed, a peer refused an
                offer, an answer failed, or the signaling session ended.
        """
        self._run(self._maybe_raise_background_errors())

    def send(self, peer: str, frame: bytes) -> None:
        """Works from any thread; the frame goes out on the loop thread."""
        self._loop.call_soon_threadsafe(self._send_on_loop, peer, frame)

    def close(self) -> None:
        """Close the signaling and each peer connection, then stop the loop.

        Raises:
            ConnectionError: The signaling fails to close. The peer connections
                and the loop close all the same.
        """
        try:
            self._run(self._close_signaling_and_rtc_connections())
        finally:
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
        self._peers = {peer.lower(): peer for peer in peers}
        await self._close_rtc_connections_of_other_emails()
        await self._maybe_sign_in()
        unconnected = [peer for peer in peers if not self._has_open_channel(peer)]
        if unconnected:
            self._start_hello(unconnected)

    async def _maybe_raise_background_errors(self) -> None:
        if self._background_errors:
            errors, self._background_errors = self._background_errors, []
            reasons = "; ".join(str(error) for error in errors)
            raise ConnectionError(
                f"WebRTC failed in the background: {reasons}"
            ) from errors[0]

    def _on_hello(self, reported_email: str, session: str) -> None:
        """Connect to a peer that said hello: offer, or say hello back so that
        the peer offers."""
        if reported_email.lower() not in self._peers:
            return
        peer = self._peers[reported_email.lower()]
        last_session = self._peer_sessions.get(peer.lower())
        self._peer_sessions[peer.lower()] = session
        # The peer says hello again at each set_peers. A hello in the same
        # session to an open channel needs nothing.
        if self._has_open_channel(peer) and session == last_session:
            return
        if self._sends_offer_to(peer):
            self._maybe_start_connect(peer)
        else:
            self._start_hello([peer])

    def _sends_offer_to(self, peer: str) -> bool:
        return self.email.lower() < peer.lower()

    def _maybe_start_connect(self, peer: str) -> None:
        """Connect to ``peer`` in the background, unless a connect is on the
        way."""
        if peer not in self._connect_tasks:
            task = asyncio.create_task(self._connect_in_background(peer))
            self._connect_tasks[peer] = task
            task.add_done_callback(lambda _: self._connect_tasks.pop(peer))

    def _start_hello(self, peers: list[str]) -> None:
        task = asyncio.create_task(self._say_hello(peers))
        self._hello_tasks.add(task)
        task.add_done_callback(self._hello_tasks.discard)

    async def _say_hello(self, peers: list[str]) -> None:
        try:
            await self.signaling.hello(self.email, peers)
        except ConnectionError as err:
            error = ConnectionError(f"Could not say hello to {peers}: {err!r}")
            self._background_errors.append(error)

    async def _connect_in_background(self, peer: str) -> None:
        try:
            await asyncio.wait_for(self._connect(peer), CONNECT_TIMEOUT_SECONDS)
        except (ConnectionError, TimeoutError) as err:
            # The next hello of the peer starts a new connection.
            error = ConnectionError(f"Could not connect to {peer}: {err!r}")
            self._background_errors.append(error)

    def _on_signaling_closed(self, error: ConnectionError) -> None:
        self._background_errors.append(error)
        # The next set_peers or connect signs in again.
        self._sign_in_task = None

    def _on_answer_failed(self, error: ConnectionError) -> None:
        self._background_errors.append(error)

    async def _close_rtc_connections_of_other_emails(self) -> None:
        """Close the peer connection to each email that is not a peer."""
        others = [
            peer for peer in self._rtc_connections if peer.lower() not in self._peers
        ]
        for peer in others:
            if peer in self._connect_tasks:
                self._connect_tasks[peer].cancel()
            if peer in self._refresh_tasks:
                self._refresh_tasks.pop(peer).cancel()
            self._channels.pop(peer, None)
            self._partial_frames.pop(peer, None)
            await self._rtc_connections.pop(peer).close()

    async def _maybe_sign_in(self) -> None:
        """Sign in to the signaling, if this client is not signed in."""
        # A task, so that two calls at the same time share one sign-in. With
        # the shield, a cancelled caller does not cancel the shared task.
        if self._sign_in_task is None or sign_in_failed(self._sign_in_task):
            self._sign_in_task = asyncio.ensure_future(
                self.signaling.listen(
                    self.email,
                    self._answer,
                    self._on_hello,
                    self._on_signaling_closed,
                    self._on_answer_failed,
                )
            )
        await asyncio.shield(self._sign_in_task)

    async def _connect(self, peer: str) -> None:
        await self._maybe_sign_in()
        ice_servers = await self.signaling.ice_servers(self.email)
        rtc = await self._new_rtc_connection(peer, ice_servers.servers)
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
        if ice_servers.lifetime_seconds is not None:
            self._schedule_refresh(peer, ice_servers.lifetime_seconds)

    def _schedule_refresh(self, peer: str, lifetime_seconds: float) -> None:
        """Open a new connection to ``peer`` before the TURN credentials of
        this one expire."""
        if peer in self._refresh_tasks:
            self._refresh_tasks.pop(peer).cancel()
        delay = lifetime_seconds * CREDENTIALS_REFRESH_FRACTION
        self._refresh_tasks[peer] = asyncio.create_task(self._refresh(peer, delay))

    async def _refresh(self, peer: str, delay: float) -> None:
        await asyncio.sleep(delay)
        # The connect schedules the next refresh; this task is no longer it.
        self._refresh_tasks.pop(peer)
        self._maybe_start_connect(peer)

    async def _answer(self, sender: str, offer: str) -> str:
        """Answer the offer of ``sender``, and return the answer.

        Raises:
            ConnectionRefusedError: ``sender`` is not a peer of this client.
        """
        if sender.lower() not in self._peers:
            raise ConnectionRefusedError(f"{sender} is not a peer of {self.email}")
        peer = self._peers[sender.lower()]
        ice_servers = await self.signaling.ice_servers(self.email)
        rtc = await self._new_rtc_connection(peer, ice_servers.servers)

        # aiortc hands over the channel already open.
        @rtc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> None:
            self._use_channel(peer, channel)
            self._on_open(peer)

        await rtc.setRemoteDescription(RTCSessionDescription(offer, "offer"))
        await rtc.setLocalDescription(await rtc.createAnswer())
        return rtc.localDescription.sdp

    async def _new_rtc_connection(
        self, peer: str, ice_servers: IceServers
    ) -> RTCPeerConnection:
        """An aiortc peer connection to ``peer``, in place of any older one.

        Each one gets new ICE servers, because TURN credentials expire.
        """
        old = self._rtc_connections.pop(peer, None)
        if old is not None:
            await old.close()
        configuration = RTCConfiguration(
            iceServers=[RTCIceServer(**server) for server in ice_servers]
        )
        self._rtc_connections[peer] = RTCPeerConnection(configuration)
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
        tasks = [
            *self._connect_tasks.values(),
            *self._refresh_tasks.values(),
            *self._hello_tasks,
        ]
        for task in tasks:
            task.cancel()
        try:
            await self._close_signaling()
        finally:
            await self._close_rtc_connections()

    async def _close_signaling(self) -> None:
        if self._sign_in_task is not None and not self._sign_in_task.done():
            self._sign_in_task.cancel()
        elif self._sign_in_task is not None and not sign_in_failed(self._sign_in_task):
            await self.signaling.close(self.email)

    async def _close_rtc_connections(self) -> None:
        await asyncio.gather(*(c.close() for c in self._rtc_connections.values()))
        self._rtc_connections.clear()
        self._channels.clear()
        self._partial_frames.clear()
