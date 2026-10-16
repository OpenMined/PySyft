"""What each peer holds, so the data owner can resend, delete and catch up.

STUB: the names and contracts are agreed; the bodies are to be written.

The tracker knows nothing about the transport or where its record is kept. It
works on paths and content hashes, takes the outcome of each send as a
``SendStatus``, and keeps its record through a ``DeliveryStateStore``.

Views
-----
For every peer the tracker keeps a view, ``{path: hash}``: the version of each
file that peer is confirmed to hold. ``plan`` compares a view with what the
peer may read now:

    readable, and the hash differs from the view  -> send
    in the view, but no longer readable           -> delete
    in the view, but the file is gone             -> delete

A peer with no view, because it is new or its record was lost, gets a
``PeerReset``: its view starts empty, so it is sent every file it may read.

Recording a send
----------------
``sent`` takes the ``SendStatus`` from ``P2PConnection.send``:

- ``DELIVERED``: the view changes at once.
- ``QUEUED``: the changes are held under their ``msg_id``, and the view changes
  only when ``P2PConnection.delivered()`` reports that ``msg_id`` and
  ``confirm`` is called. A receipt is not enough; only the peer's ack counts.

A view therefore never changes before a delivery is confirmed.

Pending deliveries
------------------
Changes queued for a peer but not yet sent, for example because the upload
failed or the peer was skipped by the version gate, are kept per peer by the
owner cache's pending deliveries (PR #9563). The tracker does not duplicate
them.

Storage
-------
``DeliveryState`` is the whole record, versioned. A ``DeliveryStateStore``
loads and saves it: a local file next to the owner's file hashes, and with
``persist_owner_state`` also Drive, inside the rolling state and checkpoints,
so an enclave that wipes its local state at every boot does not reset every
peer.

Sends not routed by permission, such as the echo of a submitter's own change
or a private dataset shared with an enclave, are recorded only where the
recipient may read the file, so a later permission change never deletes them.

Open decisions:
- A peer rejected, then approved again: reset its view and resend (proposed).
- Accepted job submissions: echo only to the submitter, as today (proposed), or
  also send to every peer who may read the file.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from syft.sync.transport.types import SendStatus

DELIVERY_VIEWS_FILENAME = "owner_delivery_views.json"
# Format of DeliveryState. Raise it when the layout changes, and keep a read
# path for every earlier version.
DELIVERY_STATE_VERSION = 1


class QueuedSend(BaseModel):
    """Changes sent to ``peer`` that wait for confirmation.

    ``changes`` maps each path to its new hash, or None for a delete.
    """

    peer: str
    changes: dict[str, str | None]


class DeliveryState(BaseModel):
    """Everything the tracker keeps: views, and sends awaiting confirmation."""

    version: int = DELIVERY_STATE_VERSION
    # peer -> {path: hash}
    views: dict[str, dict[str, str]] = Field(default_factory=dict)
    # msg_id -> the send it confirms
    queued: dict[str, QueuedSend] = Field(default_factory=dict)


class DeliveryPlan(BaseModel):
    """What ``plan`` decided for one peer."""

    model_config = ConfigDict(frozen=True)

    send: list[str] = Field(default_factory=list)
    delete: list[str] = Field(default_factory=list)


class DeliveryStateStore(ABC):
    """Where a ``DeliveryState`` is kept: a local file, Drive, or memory."""

    @abstractmethod
    def load(self) -> DeliveryState:
        """The stored state, or an empty one."""

    @abstractmethod
    def save(self, state: DeliveryState) -> None:
        """Store ``state``, replacing what was there."""


class DeliveryTracker:
    """Per-peer views of confirmed file versions, for the data owner."""

    def __init__(self, store: DeliveryStateStore):
        self.store = store

    def has_view(self, peer: str) -> bool:
        """Whether ``peer`` has a view. False means a ``PeerReset`` is due."""
        raise NotImplementedError

    def reset(self, peer: str) -> None:
        """Start ``peer``'s view empty (a ``PeerReset``), so the next plan sends
        it every file it may read."""
        raise NotImplementedError

    def plan(
        self, peer: str, readable: Mapping[str, str], paths: Iterable[str]
    ) -> DeliveryPlan:
        """What to send and delete for ``peer`` among ``paths``.

        ``readable`` maps every path ``peer`` may read now to its current hash.
        ``paths`` are the dirty paths; for a peer with no view, every readable
        path.
        """
        raise NotImplementedError

    def sent(
        self,
        peer: str,
        msg_id: str,
        status: SendStatus,
        changes: Mapping[str, str | None],
    ) -> None:
        """Record a send of ``changes`` (path -> new hash, None for a delete).

        ``DELIVERED`` updates the view now; ``QUEUED`` waits for ``confirm``.
        """
        raise NotImplementedError

    def confirm(self, msg_ids: Iterable[str]) -> None:
        """Apply the queued sends that ``P2PConnection.delivered()`` reported."""
        raise NotImplementedError
