"""Types shared by every transport backend."""

from enum import Enum

from pydantic import BaseModel, ConfigDict


class MessageKind(str, Enum):
    """What a message carries. Both kinds can travel between the same two peers."""

    # Data scientist -> data owner: proposed file changes (job submissions).
    PROPOSED_CHANGES = "proposed_changes"
    # Data owner -> data scientist: accepted file change events (results).
    FILE_EVENTS = "file_events"


class SendStatus(str, Enum):
    """What ``P2PConnection.send`` achieved."""

    # Stored where the peer will find it, even while it is offline.
    DELIVERED = "delivered"
    # Not stored yet; confirmed later through ``P2PConnection.delivered()``.
    QUEUED = "queued"


class RawMessage(BaseModel):
    """One received message, as bytes, before decoding or decryption.

    ``sender`` is set by the connection from where the message came from,
    never from its contents. ``handle`` is opaque: only the connection that
    returned the message reads it, to acknowledge it.
    """

    model_config = ConfigDict(frozen=True)

    sender: str
    msg_id: str
    kind: MessageKind
    data: bytes
    handle: str | None = None
