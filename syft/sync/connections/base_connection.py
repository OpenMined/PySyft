from typing import TYPE_CHECKING, ClassVar, Type
from pydantic import BaseModel

if TYPE_CHECKING:
    from syft.sync.transport.protocols import Transport


class FileCollection(BaseModel):
    """A collection of files that can be shared and synced."""

    folder_id: str
    tag: str
    content_hash: str
    has_any_permission: bool = False
    # The layout this collection holds, as written into the folder name. An owner
    # publishes one collection per layout its audience reads; "" is the original
    # layout. See CollectionSyncSpec.
    variant: str = ""


class ConnectionConfig(BaseModel):
    connection_type: ClassVar[Type["SyftboxPlatformConnection"]]


class SyftboxPlatformConnection(BaseModel):
    config: ConnectionConfig | None = None

    @classmethod
    def from_config(cls, config: ConnectionConfig):
        return config.connection_type.from_config(config)

    def as_transport(self) -> "Transport":
        """This connection as the transport jobs: P2P messages, owner and
        watcher storage, collections and peering."""
        raise NotImplementedError()

    def clone_for_thread(self) -> "SyftboxPlatformConnection":
        """A connection that another thread may use.

        This one, unless the backend's client cannot be shared across threads.
        """
        return self

    # =========================================================================
    # RAW BYTES TRANSPORT (used by ConnectionRouter for encryption)
    # =========================================================================

    def owner_download_next_raw_proposed_message_from_inbox(
        self, sender_email: str
    ) -> tuple[bytes, str] | None:
        """Download next message from inbox as raw bytes. Returns (data, file_id) or None."""
        raise NotImplementedError()

    def watcher_download_raw_events_from_outbox(
        self, peer_email: str, since_timestamp: float | None
    ) -> list[bytes]:
        raise NotImplementedError()
