"""Messages between peers over WebRTC.

``WebRTCP2PConnection`` keeps the queues on disk, sends receipts and resends
what the peer has not confirmed. The ``PeerConnections`` under it only carry
frames: aiortc in production, ``FakePeerConnections`` in tests.
"""

import threading
from collections import defaultdict
from collections.abc import Container
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from syft.sync.connections.protocols import P2PConnection
from syft.sync.connections.types import MessageKind, RawMessage, SendStatus
from syft.sync.connections.webrtc.message_queue import (
    BYTES_AS_BASE64,
    MessageQueue,
    QueuedMessage,
)
from syft.sync.connections.webrtc.peer_connections import PeerConnections
from syft.sync.sync.constants import CACHE_DIR

WEBRTC_QUEUES_DIR = "webrtc"
OUTGOING_DIR = "outgoing"
INCOMING_DIR = "incoming"
DELIVERY_TIMEOUT_SECONDS = 30.0


class Frame(BaseModel):
    """What travels over a channel: a message, or the receipt for one."""

    model_config = ConfigDict(frozen=True, **BYTES_AS_BASE64)

    message: QueuedMessage | None = None
    # The msg_id of a message the peer stored.
    receipt: str | None = None


class WebRTCP2PConnection(P2PConnection):
    """Messages over one channel per peer, with a queue per peer and direction.

    ``send`` stores the message in the peer's outgoing queue before sending it,
    so a restart loses nothing, and returns ``QUEUED``. The peer stores it in
    its incoming queue and sends a receipt back. The message then leaves the
    outgoing queue and ``delivered()`` reports it.

    ``AiortcPeerConnections`` calls ``on_open`` and ``on_frame`` from aiortc's
    loop thread, so they and ``send`` hold ``_lock``.
    """

    def __init__(self, email: str, syftbox_folder: Path):
        self.email = email
        self._folder = Path(syftbox_folder) / CACHE_DIR / WEBRTC_QUEUES_DIR
        # Peers whose channel opened and who got every older message again.
        # New messages go straight out only to them, so none overtakes an
        # older one.
        self._caught_up: set[str] = set()
        self._delivered: list[str] = []
        # Reentrant: FakePeerConnections brings the receipt back inside ``send``.
        self._lock = threading.RLock()
        self._receipt_arrived = threading.Condition(self._lock)
        self._peer_connections: PeerConnections | None = None

    def open(self, peer_connections: PeerConnections) -> None:
        """Reach peers through ``peer_connections`` from now on: aiortc, or
        ``FakePeerConnections`` in tests."""
        self._peer_connections = peer_connections
        peer_connections.open(self.email, self.on_open, self.on_frame)

    def connect(self, peer: str) -> None:
        """Open a channel to ``peer``; return once it is open."""
        self._peer_connections.connect(peer)

    def close(self) -> None:
        """Close the peer connections. Unconfirmed messages stay in the outgoing
        queues."""
        with self._lock:
            self._caught_up.clear()
        if self._peer_connections is not None:
            self._peer_connections.close()
            self._peer_connections = None

    def on_open(self, peer: str) -> None:
        """``peer``'s channel opened: send it every message it has not
        confirmed, oldest first. New messages follow them, so none overtakes
        an older one."""
        with self._lock:
            for message in self._outgoing(peer).messages():
                self._send_frame(peer, Frame(message=message))
            self._caught_up.add(peer)

    def send(
        self, peer: str, msg_id: str, data: bytes, kind: MessageKind
    ) -> SendStatus:
        message = QueuedMessage(msg_id=msg_id, kind=kind, data=data)
        # Stored first, and not under the folder lock while it travels:
        # FakePeerConnections brings the receipt back inside this call.
        with self._lock:
            self._outgoing(peer).add(message)
            if peer in self._caught_up:
                self._send_frame(peer, Frame(message=message))
        return SendStatus.QUEUED

    def on_frame(self, peer: str, data: bytes) -> None:
        """Called by the peer connections with each frame from ``peer``."""
        frame = Frame.model_validate_json(data)
        with self._lock:
            if frame.message is not None:
                self._incoming(peer).add(frame.message)
                self._send_frame(peer, Frame(receipt=frame.message.msg_id))
            elif frame.receipt is not None:
                self._delivered += self._outgoing(peer).remove({frame.receipt})
                self._receipt_arrived.notify_all()

    def delivered(self) -> list[str]:
        with self._lock:
            delivered, self._delivered = self._delivered, []
        return delivered

    def wait_until_delivered(self, timeout: float = DELIVERY_TIMEOUT_SECONDS) -> None:
        """Wait until every queued message has a receipt from its peer. Raises
        ``TimeoutError`` if some still have none after ``timeout`` seconds."""
        with self._lock:
            if not self._receipt_arrived.wait_for(self._all_delivered, timeout):
                raise TimeoutError(
                    f"{self.email}: messages without a receipt after {timeout}s"
                )

    def receive(
        self,
        peer: str,
        kind: MessageKind,
        after: str | None = None,
        max_messages: int | None = None,
        already_received: Container[str] = frozenset(),
    ) -> list[RawMessage]:
        """The messages of ``kind`` from ``peer`` in the incoming queue, in the
        order they arrived. ``after`` and ``already_received`` are not needed:
        the queue holds only messages not processed yet."""
        return [
            RawMessage(sender=peer, msg_id=m.msg_id, kind=m.kind, data=m.data)
            for m in self._incoming(peer).messages(kind, max_messages)
        ]

    def mark_processed(self, message: RawMessage) -> None:
        self.mark_all_processed([message])

    def mark_all_processed(self, messages: list[RawMessage]) -> None:
        """Remove ``messages`` from the incoming queues, one lock per sender."""
        # sender email -> msg_ids processed
        by_sender: dict[str, set[str]] = defaultdict(set)
        for message in messages:
            by_sender[message.sender].add(message.msg_id)
        for sender, msg_ids in by_sender.items():
            self._incoming(sender).remove(msg_ids)

    def _send_frame(self, peer: str, frame: Frame) -> None:
        """Send ``frame`` once ``open`` was called. A message that does not
        arrive stays in the outgoing queue and goes again on ``on_open``."""
        if self._peer_connections is not None:
            self._peer_connections.send(peer, frame.model_dump_json().encode())

    def _all_delivered(self) -> bool:
        outgoing = self._folder / OUTGOING_DIR
        if not outgoing.exists():
            return True
        queues = [MessageQueue(p) for p in outgoing.iterdir() if p.is_dir()]
        return all(queue.is_empty() for queue in queues)

    def _outgoing(self, peer: str) -> MessageQueue:
        return MessageQueue(self._folder / OUTGOING_DIR / peer)

    def _incoming(self, peer: str) -> MessageQueue:
        return MessageQueue(self._folder / INCOMING_DIR / peer)
