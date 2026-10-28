"""The jobs a transport backend does, one abstract class each.

``P2PConnection`` carries messages between peers, ``OwnerStore`` and
``WatcherStore`` hold each role's own storage, ``Collections`` shares datasets,
and ``Peering`` finds and trusts peers. A backend implements the classes it
can. The sync layer above never sees backend ids: it passes back the messages
and handles a backend returns.

Each class keeps its abstract methods to the few a backend must write. Methods
that only combine those are ordinary methods on the base class, which a
backend may override when it can do them faster.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from syft.sync.platforms.base_platform import BasePlatform
from syft.sync.transport.types import MessageKind, RawMessage, SendStatus

if TYPE_CHECKING:
    from syft.sync.connections.base_connection import FileCollection
    from syft.sync.version.peer_info import PeerInfo


class P2PConnection(ABC):
    """Messages between peers, kept until the recipient acknowledges them.

    One rule for every message, whatever its kind: ``receive`` returns the
    messages not acknowledged yet, and ``ack`` finishes one. On Drive a written
    message is ``DELIVERED`` and ``ack`` removes it. On a live link such as
    WebRTC a message is ``QUEUED`` until the peer acknowledges it, which
    ``delivered()`` then reports.
    """

    @abstractmethod
    def send(
        self, peer: str, msg_id: str, data: bytes, kind: MessageKind
    ) -> SendStatus:
        """Send ``data`` to ``peer`` under ``msg_id``.

        ``msg_id`` is chosen by the sender and stays the same if the message is
        sent again. ``kind`` lets a connection keep the two kinds apart, as
        Drive does with an inbox and an outbox.
        """

    @abstractmethod
    def delivered(self) -> list[str]:
        """The ``msg_id`` of each queued message the peer acknowledged since
        the last call. A connection that only returns ``DELIVERED`` returns
        nothing."""

    @abstractmethod
    def receive(self, max_messages: int | None = None) -> list[RawMessage]:
        """Messages from every peer that are not acknowledged yet, oldest first
        per sender, at most ``max_messages`` of them.

        A peer whose channel does not exist yet has no messages. The connection
        sets each ``RawMessage.sender`` from where the message came from.
        """

    @abstractmethod
    def ack(self, message: RawMessage) -> None:
        """Record that ``message`` was processed; ``receive`` never returns it
        again."""


class OwnerStore(ABC):
    """The data owner's own storage: event log, checkpoints, rolling state.

    Only the owner writes and reads it, and it is never shared.
    """

    # -- event log -------------------------------------------------------------
    @abstractmethod
    def append_event(self, name: str, data: bytes) -> str:
        """Append one events message to the event log; return its handle."""

    @abstractmethod
    def list_event_handles(self, since_timestamp: float | None = None) -> list[str]:
        """Handles of the events messages sent after ``since_timestamp``, oldest
        first; every message when it is None."""

    @abstractmethod
    def fetch(self, handle: str) -> bytes:
        """The bytes stored under ``handle``."""

    def fetch_events_since(self, since_timestamp: float | None) -> list[bytes]:
        """The events messages sent after ``since_timestamp``, oldest first."""
        return [self.fetch(h) for h in self.list_event_handles(since_timestamp)]

    def fetch_all_events(self) -> list[bytes]:
        """Every events message in the event log, oldest first."""
        return self.fetch_events_since(None)

    def count_events_since(self, since_timestamp: float | None) -> int:
        """How many events messages were sent after ``since_timestamp``."""
        return len(self.list_event_handles(since_timestamp))

    # -- checkpoints -----------------------------------------------------------
    @abstractmethod
    def put_checkpoint(self, name: str, data: bytes) -> str:
        """Store a full checkpoint; it replaces every earlier one."""

    @abstractmethod
    def latest_checkpoint(self) -> bytes | None:
        """The newest full checkpoint, or None."""

    @abstractmethod
    def put_incremental(self, name: str, data: bytes) -> str:
        """Store an incremental checkpoint."""

    @abstractmethod
    def all_incrementals(self) -> list[bytes]:
        """Every incremental checkpoint since the last full one."""

    @abstractmethod
    def incremental_count(self) -> int:
        """How many incremental checkpoints are stored, without fetching them."""

    @abstractmethod
    def next_incremental_sequence(self) -> int:
        """The sequence number the next incremental checkpoint takes."""

    @abstractmethod
    def delete_incrementals(self) -> None:
        """Remove every incremental checkpoint."""

    # -- rolling state ---------------------------------------------------------
    @abstractmethod
    def put_rolling_state(self, name: str, data: bytes) -> str:
        """Store the rolling state, overwriting the previous one."""

    @abstractmethod
    def get_rolling_state(self) -> bytes | None:
        """The rolling state, or None."""

    @abstractmethod
    def delete_rolling_state(self) -> None:
        """Remove the rolling state."""


class WatcherStore(ABC):
    """The data scientist's memory of what each owner sent it.

    One area per owner, in the scientist's own storage, never shared, so it
    survives a lost local disk such as a Colab runtime reset.
    """

    @abstractmethod
    def put_checkpoint(self, peer: str, name: str, data: bytes) -> str:
        """Store a snapshot of every file received from ``peer``; it replaces
        the previous one."""

    @abstractmethod
    def latest_checkpoint(self, peer: str) -> bytes | None:
        """The newest snapshot of files received from ``peer``, or None."""

    @abstractmethod
    def put_rolling_state(self, peer: str, name: str, data: bytes) -> str:
        """Store the changes received from ``peer`` since the snapshot, and the
        ids of the messages already processed."""

    @abstractmethod
    def get_rolling_state(self, peer: str) -> bytes | None:
        """The rolling state for ``peer``, or None."""

    @abstractmethod
    def reset(self, peer: str) -> None:
        """Forget everything received from ``peer``, before asking it to send
        everything again."""


class Collections(ABC):
    """Sets of files a data owner shares with named readers (datasets)."""

    @abstractmethod
    def create(self, prefix: str, tag: str, content_hash: str, owner_email: str) -> str:
        """Create the collection if it does not exist; return its handle."""

    @abstractmethod
    def mark_shared_with_any(self, prefix: str, tag: str, content_hash: str) -> None:
        """Flag the collection as meant for every approved peer."""

    @abstractmethod
    def grant_read(
        self, prefix: str, tag: str, content_hash: str, users: list[str]
    ) -> None:
        """Let ``users`` read the collection. Granting twice is harmless."""

    @abstractmethod
    def upload(
        self, prefix: str, tag: str, content_hash: str, files: dict[str, bytes]
    ) -> None:
        """Store ``files`` in the collection."""

    @abstractmethod
    def list_mine_with_permissions(self, prefix: str) -> list[FileCollection]:
        """The collections I published under ``prefix``, with their flags."""

    def list_mine(self, prefix: str) -> list[str]:
        """The tags of the collections I published under ``prefix``, each once."""
        tags = (c.tag for c in self.list_mine_with_permissions(prefix))
        return list(dict.fromkeys(tags))

    @abstractmethod
    def delete(self, prefix: str, tag: str) -> None:
        """Remove every layout of ``tag`` published under ``prefix``."""

    @abstractmethod
    def list_shared_with_me(self, prefix: str) -> list[dict]:
        """Collections peers shared with me, each with owner_email, tag,
        content_hash and variant."""

    @abstractmethod
    def list_files(
        self, prefix: str, tag: str, content_hash: str, owner_email: str
    ) -> list[dict]:
        """The files of one shared collection, each with file_id and file_name."""

    @abstractmethod
    def fetch_file(self, handle: str) -> bytes:
        """The bytes of one collection file."""

    def download(
        self, prefix: str, tag: str, content_hash: str, owner_email: str
    ) -> dict[str, bytes]:
        """Every file of one collection a peer shared with me, by file name."""
        files = self.list_files(prefix, tag, content_hash, owner_email)
        return {f["file_name"]: self.fetch_file(f["file_id"]) for f in files}


class Peering(ABC):
    """Finding and trusting a peer: identity, channels, peer list, published
    documents."""

    @abstractmethod
    def whoami(self) -> str:
        """The email of the account behind this backend's credentials."""

    @abstractmethod
    def open_channels(self, peer: str) -> None:
        """Create the message channels with ``peer`` and let it write to them."""

    @abstractmethod
    def incoming_requests(self) -> list[str]:
        """Emails of peers who opened channels towards me."""

    # -- peer list (SYFT_peers.json) -------------------------------------------
    @abstractmethod
    def read_peer_registry(self, force_download: bool = False) -> dict:
        """My record of every peer and its state, keyed by email."""

    @abstractmethod
    def update_peer_registry(
        self, peer: str, state: str, public_encryption_bundle: dict | None = None
    ) -> None:
        """Record ``peer``'s state, and its key bundle when given."""

    # -- key bundles -----------------------------------------------------------
    @abstractmethod
    def publish_key_bundle(self, peer: str, bundle_json: str) -> None:
        """Publish my public key bundle for ``peer``."""

    @abstractmethod
    def grant_key_bundle_read(self, peer: str) -> None:
        """Let ``peer`` read the key bundles I publish."""

    @abstractmethod
    def read_own_key_bundle(self, peer: str) -> str | None:
        """The key bundle I published for ``peer``, or None."""

    @abstractmethod
    def read_peer_key_bundle(self, peer: str) -> str | None:
        """The key bundle ``peer`` published for me, or None."""

    # -- version documents -----------------------------------------------------
    @abstractmethod
    def publish_peer_info(self, peer_info: PeerInfo) -> None:
        """Publish my version document."""

    @abstractmethod
    def read_own_peer_info(self) -> PeerInfo | None:
        """My published version document, or None.

        Raises:
            TransportUnavailableError: when the backend cannot be reached.
            ValueError: when the document is not one this client reads.
        """

    @abstractmethod
    def read_peer_info(self, peer: str) -> PeerInfo | None:
        """``peer``'s published version document, or None.

        Raises:
            TransportUnavailableError: when the backend cannot be reached.
            ValueError: when the document is not one this client reads.
        """

    @abstractmethod
    def grant_peer_info_read(self, peer: str, peer_info: PeerInfo) -> None:
        """Let ``peer`` read my version document.

        When I have no document yet, publish ``peer_info`` first.
        """


class Transport(BaseModel):
    """One implementation of each job. The parts may come from different
    backends, though a peer's ``p2p`` and ``peering`` go together."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    p2p: P2PConnection
    owner_store: OwnerStore
    watcher_store: WatcherStore
    collections: Collections
    peering: Peering
    # Recorded on each Peer so a peer list says which backend reaches it.
    platform: BasePlatform
