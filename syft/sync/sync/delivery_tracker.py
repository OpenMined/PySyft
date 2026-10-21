"""What each peer holds, so the data owner can resend, delete and catch up.

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

A peer with no view, because it is new or its record was lost, starts with an
empty view (a ``PeerReset``), so it is sent every file it may read.

Recording a send
----------------
``sent`` takes the ``SendStatus`` from ``P2PConnection.send``:

- ``DELIVERED``: the view changes at once.
- ``QUEUED``: the changes are held under their ``msg_id``, and the view changes
  only when ``P2PConnection.delivered()`` reports that ``msg_id`` and
  ``confirm`` is called. A receipt is not enough; only the peer's ack counts.

A view therefore never changes before a delivery is confirmed. A change that is
not sent, because the upload failed, the peer was skipped, or a process
stopped before the send, stays in the plan of the next sync.
"""

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import ContextManager, Iterable, Mapping
from uuid import uuid4

import portalocker
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
        """The stored state, or an empty one.

        Raises:
            ValueError: when a later client wrote the state.
        """

    @abstractmethod
    def save(self, state: DeliveryState) -> None:
        """Store ``state``, replacing what was there."""

    def locked(self) -> ContextManager[None]:
        """Keep other writers out from a ``load`` to the ``save`` that follows."""
        return nullcontext()


class MemoryDeliveryStateStore(DeliveryStateStore):
    """A state kept in this process only."""

    def __init__(self):
        self._state = DeliveryState()

    def load(self) -> DeliveryState:
        return self._state.model_copy(deep=True)

    def save(self, state: DeliveryState) -> None:
        self._state = state.model_copy(deep=True)


class LocalDeliveryStateStore(DeliveryStateStore):
    """A state kept in a JSON file, shared by the processes of one datasite."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> DeliveryState:
        if not self.path.exists():
            return DeliveryState()
        state = DeliveryState.model_validate_json(self.path.read_text())
        if state.version > DELIVERY_STATE_VERSION:
            raise ValueError(
                f"{self.path} has version {state.version}, and this client reads "
                f"up to version {DELIVERY_STATE_VERSION}."
            )
        return state

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(self.path.suffix + ".lock"), "a") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            try:
                yield
            finally:
                portalocker.unlock(f)

    def save(self, state: DeliveryState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".tmp.{os.getpid()}.{uuid4().hex}")
        try:
            tmp.write_text(state.model_dump_json())
            tmp.replace(self.path)
        finally:
            tmp.unlink(missing_ok=True)


def _apply(view: dict[str, str], changes: Mapping[str, str | None]) -> None:
    for path, new_hash in changes.items():
        if new_hash is None:
            view.pop(path, None)
        else:
            view[path] = new_hash


class DeliveryTracker:
    """Per-peer views of confirmed file versions, for the data owner.

    Each call reads the state from the store. Each change reads and writes it
    back under the store lock, so the processes of one datasite see the same
    views and do not lose each other's changes.
    """

    def __init__(self, store: DeliveryStateStore):
        self.store = store

    def has_view(self, peer: str) -> bool:
        """Whether ``peer`` has a view. False means a ``PeerReset`` is due."""
        return peer in self.store.load().views

    def reset(self, peer: str) -> None:
        """Start ``peer``'s view empty (a ``PeerReset``), so the next plan sends
        it every file it may read."""
        with self.store.locked():
            state = self.store.load()
            state.views[peer] = {}
            self.store.save(state)

    def clear(self) -> None:
        """Forget every view and every queued send."""
        with self.store.locked():
            self.store.save(DeliveryState())

    def plan(self, peer: str, readable: Mapping[str, str]) -> DeliveryPlan:
        """What to send and delete for ``peer``.

        ``readable`` maps every path ``peer`` may read now to its current hash.
        """
        view = self.store.load().views.get(peer, {})
        return DeliveryPlan(
            send=sorted(p for p, h in readable.items() if view.get(p) != h),
            delete=sorted(p for p in view if p not in readable),
        )

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
        with self.store.locked():
            state = self.store.load()
            if status is SendStatus.DELIVERED:
                _apply(state.views.setdefault(peer, {}), changes)
            else:
                assert status is SendStatus.QUEUED, status
                state.queued[msg_id] = QueuedSend(peer=peer, changes=dict(changes))
            self.store.save(state)

    def confirm(self, msg_ids: Iterable[str]) -> None:
        """Apply the queued sends that ``P2PConnection.delivered()`` reported."""
        msg_ids = list(msg_ids)
        if not msg_ids:
            return
        with self.store.locked():
            state = self.store.load()
            for msg_id in msg_ids:
                queued = state.queued.pop(msg_id)
                _apply(state.views.setdefault(queued.peer, {}), queued.changes)
            self.store.save(state)
