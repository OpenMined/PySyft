"""Messages between peers on Google Drive: the inbox and outbox folders."""

from syft.sync.connections.drive.gdrive_layout import DriveLayout
from syft.sync.connections.drive.gdrive_retry import execute_with_retries
from syft.sync.transport.protocols import P2PConnection
from syft.sync.transport.types import MessageKind, RawMessage, SendStatus


class DriveP2PConnection(P2PConnection):
    """Messages through the shared inbox and outbox folders.

    A proposed change goes to the data owner's inbox and file events go to the
    data scientist's outbox. ``msg_id`` is the file name. A written file is
    ``DELIVERED``: it waits in the folder until the peer reads it.

    Only sending goes through this class so far. The router still reads and
    archives messages with the Drive methods directly, because doing it through
    ``receive`` and ``ack`` changes how they behave (one rule for both folders,
    delete on ack, no read cursor); that lands with send-and-forget.
    """

    def __init__(self, layout: DriveLayout):
        self._layout = layout

    def send(
        self, peer: str, msg_id: str, data: bytes, kind: MessageKind
    ) -> SendStatus:
        if kind is MessageKind.PROPOSED_CHANGES:
            self.watcher_send_raw_bytes_to_inbox(peer, msg_id, data)
        elif kind is MessageKind.FILE_EVENTS:
            self.owner_write_raw_bytes_to_outbox(peer, msg_id, data)
        else:
            raise ValueError(f"Drive has no folder for messages of kind {kind!r}")
        return SendStatus.DELIVERED

    def delivered(self) -> list[str]:
        # Every Drive send is DELIVERED when it returns; none waits for a peer.
        return []

    def receive(self, max_messages: int | None = None) -> list[RawMessage]:
        raise NotImplementedError(
            "Messages are still read through the Drive methods; receiving "
            "through P2PConnection lands with send-and-forget."
        )

    def ack(self, message: RawMessage) -> None:
        raise NotImplementedError(
            "Messages are still archived through the Drive methods; "
            "acknowledging through P2PConnection lands with send-and-forget."
        )

    # -- Drive -----------------------------------------------------------------

    def watcher_send_raw_bytes_to_inbox(
        self, recipient: str, filename: str, data: bytes
    ) -> None:
        inbox_id = self._layout.get_peer_datasite_inbox_id(recipient)
        if inbox_id is None:
            raise Exception(f"Inbox folder for {recipient}'s datasite not found")

        payload, _ = self._layout.client.create_file_payload(data)
        file_metadata = {"name": filename, "parents": [inbox_id]}

        result = self._layout.client.create_syft_file(
            file_metadata, payload, fields="id, parents"
        )

        file_id = result.get("id")
        actual_parents = result.get("parents", [])
        if inbox_id not in actual_parents:
            print(
                f"WARNING: Message file {file_id} was not placed in inbox folder "
                f"{inbox_id}. Actual parents: {actual_parents}. "
                f"Moving file to correct folder..."
            )
            execute_with_retries(
                self._layout.client.drive_service.files().update(
                    fileId=file_id,
                    addParents=inbox_id,
                    fields="id, parents",
                )
            )

    def owner_write_raw_bytes_to_outbox(
        self, recipient: str, filename: str, data: bytes
    ) -> None:
        outbox_folder_id = self._layout.get_own_datasite_outbox_id(recipient)
        if outbox_folder_id is None:
            raise ValueError(f"Outbox folder for {recipient} not found")

        file_payload, _ = self._layout.client.create_file_payload(data)
        file_metadata = {"name": filename, "parents": [outbox_folder_id]}

        result = self._layout.client.create_syft_file(
            file_metadata, file_payload, fields="id, parents"
        )

        file_id = result.get("id")
        actual_parents = result.get("parents", [])
        if outbox_folder_id not in actual_parents:
            print(
                f"WARNING: Event message {file_id} was not placed in outbox folder "
                f"{outbox_folder_id}. Actual parents: {actual_parents}. "
                f"Moving file to correct folder..."
            )
            execute_with_retries(
                self._layout.client.drive_service.files().update(
                    fileId=file_id,
                    addParents=outbox_folder_id,
                    fields="id, parents",
                )
            )
