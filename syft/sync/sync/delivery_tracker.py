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

A change that the owner accepts from a peer goes to that peer at once. Every
other peer that may read the file gets it in its next plan, as with any change.

A new peer starts with an empty view, so it is sent every file it may read.
When the owner rejects a peer, ``reset`` sets each path in its view to
``UNSENT_HASH``. Thus a peer
that the owner rejects and then approves again is sent every file it may read,
and a delete for each file that it may no longer have.

An owner that lost its views file, for example an enclave that wipes its local
state at every boot, has file hashes but no views. ``seed`` then starts the
view of each approved peer with every file it may read now, so the owner does
not send every file again. A view that a send already recorded is kept.

Recording a send
----------------
``record_send`` takes the ``SendStatus`` from ``P2PConnection.send``:

- ``DELIVERED``: the view changes at once.
- ``QUEUED``: the changes are held under their ``msg_id``, and the view changes
  only when ``P2PConnection.delivered()`` reports that ``msg_id`` and
  ``confirm_deliveries`` is called. A receipt is not enough; only the peer's ack counts.

A view therefore never changes before a delivery is confirmed. A change that is
not sent, because the upload failed, the peer was skipped, or a process
stopped before the send, stays in the plan of the next sync.

Storage
-------
The views are kept on the owner's local disk only, also with
``persist_owner_state``. An owner that loses its local state, file hashes
included, sends every peer every file it may read once again. With
``persist_owner_state``, the file hashes come back from Drive, so ``seed``
starts the views from them.
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import ContextManager, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from syft.sync.sync.caches.persisted_dict import locked_file, write_text_atomic
from syft.sync.transport.types import SendStatus

DELIVERY_VIEWS_FILENAME = "owner_delivery_views.json"
# Format of DeliveryState. Raise it when the layout changes, and keep a read
# path for every earlier version.
DELIVERY_STATE_VERSION = 1
# A view hash that matches no file, for a path that the peer may hold in any
# version. The next plan sends the path, or deletes it.
UNSENT_HASH = ""


class QueuedSend(BaseModel):
    """Changes sent to ``peer`` that wait for confirmation.

    ``changes`` maps each path to its new hash, or None for a delete.
    """

    peer: str
    changes: dict[str, str | None]


class DeliveryState(BaseModel):
    """Everything the tracker keeps."""

    version: int = DELIVERY_STATE_VERSION
    # peer email -> {path: hash}
    views: dict[str, dict[str, str]] = Field(default_factory=dict)
    # msg_id -> the send it confirms
    queued: dict[str, QueuedSend] = Field(default_factory=dict)
    # Whether ``DeliveryTracker.seed`` ran on this state.
    seeded: bool = False


class DeliveryPlan(BaseModel):
    """What ``plan`` decided for one peer."""

    model_config = ConfigDict(frozen=True)

    send: list[str] = Field(default_factory=list)
    delete: list[str] = Field(default_factory=list)


class DeliveryStateStore(ABC):
    """Where a ``DeliveryState`` is kept: a local file or memory."""

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

    def locked(self) -> ContextManager[None]:
        return locked_file(self.path)

    def save(self, state: DeliveryState) -> None:
        write_text_atomic(self.path, state.model_dump_json())


def _apply_changes_to_view(
    view: dict[str, str], changes: Mapping[str, str | None]
) -> None:
    for path, new_hash in changes.items():
        if new_hash is None:
            view.pop(path, None)
        else:
            view[path] = new_hash


def _record_delivery(
    state: DeliveryState, peer: str, changes: Mapping[str, str | None]
) -> None:
    """Apply ``changes`` to ``peer``'s view."""
    _apply_changes_to_view(state.views.setdefault(peer, {}), changes)


class DeliveryTracker:
    """Per-peer views of confirmed file versions, for the data owner.

    Each call reads the state from the store. Each change reads and writes it
    back under the store lock, so the processes of one datasite see the same
    views and do not lose each other's changes.
    """

    def __init__(self, store: DeliveryStateStore):
        self.store = store

    def clear(self) -> None:
        """Forget the whole state: views, queued sends and the seeded flag."""
        with self.store.locked():
            self.store.save(DeliveryState())

    def reset(self, peer: str) -> None:
        """Set each path in ``peer``'s view to ``UNSENT_HASH``.

        The next plan sends ``peer`` every file it may read, and deletes the
        paths of its view that it may no longer have.
        """
        with self.store.locked():
            state = self.store.load()
            view = state.views.get(peer, {})
            state.views[peer] = dict.fromkeys(view, UNSENT_HASH)
            self.store.save(state)

    def is_seeded(self) -> bool:
        """Whether ``seed`` ran on the stored state."""
        return self.store.load().seeded

    def seed(self, held_by_peer: Mapping[str, Mapping[str, str]]) -> None:
        """Add the files in ``held_by_peer`` to each peer's view, one time.

        ``held_by_peer`` maps each peer email to ``{path: hash}``. A path that the view already holds keeps its hash. After the first
        call, later calls do nothing.
        """
        with self.store.locked():
            state = self.store.load()
            if state.seeded:
                return
            # held_by_peer: peer email -> {path: hash}
            for peer, files in held_by_peer.items():
                state.views[peer] = {**files, **state.views.get(peer, {})}
            state.seeded = True
            self.store.save(state)

    def plan(
        self, readable_by_peer: Mapping[str, Mapping[str, str]]
    ) -> dict[str, DeliveryPlan]:
        """What to send and delete for each peer in ``readable_by_peer``.

        ``readable_by_peer`` maps each peer email to ``{path: hash}``: every
        path it may read now, with the path's current hash.
        """
        views = self.store.load().views
        plans = {}
        # readable_by_peer: peer email -> {path: hash}
        for peer, readable in readable_by_peer.items():
            view = views.get(peer, {})
            plans[peer] = DeliveryPlan(
                send=sorted(p for p, h in readable.items() if view.get(p) != h),
                delete=sorted(p for p in view if p not in readable),
            )
        return plans

    def record_send(
        self,
        peer: str,
        msg_id: str,
        status: SendStatus,
        changes: Mapping[str, str | None],
    ) -> None:
        """Record a send of ``changes`` (path -> new hash, None for a delete).

        ``DELIVERED`` updates the view now; ``QUEUED`` waits for ``confirm_deliveries``.
        """
        with self.store.locked():
            state = self.store.load()
            if status is SendStatus.DELIVERED:
                _record_delivery(state, peer, changes)
            else:
                assert status is SendStatus.QUEUED, status
                state.queued[msg_id] = QueuedSend(peer=peer, changes=dict(changes))
            self.store.save(state)

    def confirm_deliveries(self, msg_ids: Iterable[str]) -> None:
        """Apply the queued sends that ``P2PConnection.delivered()`` reported."""
        msg_ids = list(msg_ids)
        if not msg_ids:
            return
        with self.store.locked():
            state = self.store.load()
            for msg_id in msg_ids:
                queued = state.queued.pop(msg_id)
                _record_delivery(state, queued.peer, queued.changes)
            self.store.save(state)
