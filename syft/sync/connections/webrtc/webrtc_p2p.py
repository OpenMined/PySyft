"""Messages between peers over WebRTC: the synchronous core.

The core keeps the queues on disk, sends receipts and resends what the peer has
not confirmed. The ``Channel`` under it only carries frames: an aiortc data
channel in production, a ``FakeChannel`` in tests.
"""

from collections import defaultdict
from collections.abc import Container
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from syft.sync.connections.protocols import P2PConnection
from syft.sync.connections.types import MessageKind, RawMessage, SendStatus
from syft.sync.connections.webrtc.channel import Channel
from syft.sync.connections.webrtc.message_queue import (
    BYTES_AS_BASE64,
    MessageQueue,
    QueuedMessage,
)
from syft.sync.sync.constants import CACHE_DIR

WEBRTC_QUEUES_DIR = "webrtc"


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
    """

    def __init__(self, email: str, syftbox_folder: Path):
        self.email = email
        self._folder = Path(syftbox_folder) / CACHE_DIR / WEBRTC_QUEUES_DIR
        # peer email -> its channel
        self._channels: dict[str, Channel] = {}
        # Peers whose channel is open and got every older message again.
        self._open: set[str] = set()
        self._delivered: list[str] = []

    def attach(self, peer: str, channel: Channel) -> None:
        """Use ``channel`` for ``peer`` from now on. Receipts go over it right
        away; messages once it opens."""
        self._channels[peer] = channel

    def on_open(self, peer: str) -> None:
        """``peer``'s channel opened: send it every message it has not
        confirmed, oldest first. New messages follow them, so none overtakes
        an older one."""
        for message in self._outgoing(peer).messages():
            self._send_frame(peer, Frame(message=message))
        self._open.add(peer)

    def send(
        self, peer: str, msg_id: str, data: bytes, kind: MessageKind
    ) -> SendStatus:
        message = QueuedMessage(msg_id=msg_id, kind=kind, data=data)
        # Stored first, and not under the folder lock while it travels: a
        # FakeChannel brings the receipt back inside this call.
        self._outgoing(peer).add(message)
        if peer in self._open:
            self._send_frame(peer, Frame(message=message))
        return SendStatus.QUEUED

    def on_frame(self, peer: str, data: bytes) -> None:
        """Called by ``peer``'s channel with each frame that arrives."""
        frame = Frame.model_validate_json(data)
        if frame.message is not None:
            self._incoming(peer).add(frame.message)
            self._send_frame(peer, Frame(receipt=frame.message.msg_id))
        elif frame.receipt is not None:
            self._delivered += self._outgoing(peer).remove({frame.receipt})

    def delivered(self) -> list[str]:
        delivered, self._delivered = self._delivered, []
        return delivered

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
        """Send ``frame`` if ``peer`` has a channel. A message without one
        waits in the outgoing queue until ``on_open``."""
        channel = self._channels.get(peer)
        if channel is not None:
            channel.send(frame.model_dump_json().encode())

    def _outgoing(self, peer: str) -> MessageQueue:
        return MessageQueue(self._folder / "outgoing" / peer)

    def _incoming(self, peer: str) -> MessageQueue:
        return MessageQueue(self._folder / "incoming" / peer)
