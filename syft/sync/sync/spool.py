"""The messages a DS saves before it sends them.

A dropped connection or a stopped kernel must not lose a job submission. Each
message to a peer is saved to ``<syftbox>/<me>/private/spool/<peer>/`` before
any network call, and it is removed when the peer has it. ``private/`` is never
synced.

A saved message is the message before the downgrade and the encryption for its
peer. The upload does both, so it uses the protocol and the key that the peer
has at that time. An enclave gets a new key at every boot.
"""

import json
from contextlib import AbstractContextManager
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from syft.sync.messages.proposed_filechange import ProposedFileChangesMessage
from syft.sync.peers.peer_store import PRIVATE_DIR_NAME
from syft.sync.sync.caches.persisted_dict import locked_file, write_text_atomic

SPOOL_DIR_NAME = "spool"
SPOOL_ENTRY_SUFFIX = ".json"
# Format of a spool entry. Raise it when the layout of an entry changes. An
# entry of a later version is refused.
SPOOL_ENTRY_VERSION = 1


def datasite_spool_path(syftbox_folder: Path | str, email: str) -> Path:
    """Per-datasite spool: ``<syftbox_folder>/<email>/private/spool``."""
    return Path(syftbox_folder) / email / PRIVATE_DIR_NAME / SPOOL_DIR_NAME


class SpoolEntry(BaseModel):
    """One saved message."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    version: int
    # The compressed message, before the downgrade and the encryption for its peer.
    compressed_message: bytes

    @classmethod
    def read(cls, path: Path) -> "SpoolEntry":
        """Read the entry at ``path``.

        Raises:
            ValueError: if a later client wrote the entry, or it has another
                shape.
        """
        text = path.read_text()
        data = json.loads(text)
        # Checked before the fields, so a later layout is refused by its version.
        if data["version"] > SPOOL_ENTRY_VERSION:
            raise ValueError(
                f"The spool entry at {path} has version {data['version']}, and "
                f"this client reads up to version {SPOOL_ENTRY_VERSION}."
            )
        return cls.model_validate_json(text)


class Spool:
    """The saved messages of one DS, in one folder per recipient. An entry is
    named after the ``msg_id`` its message gets."""

    def __init__(self, folder: Path):
        self.folder = folder

    def add(self, recipient: str, message: ProposedFileChangesMessage) -> None:
        """Save ``message`` to ``recipient``."""
        entry = SpoolEntry(
            version=SPOOL_ENTRY_VERSION, compressed_message=message.as_compressed_data()
        )
        msg_id = message.message_filename.as_string()
        write_text_atomic(self._entry_path(recipient, msg_id), entry.model_dump_json())

    def recipients(self) -> list[str]:
        """The recipients that have a folder in the spool."""
        if not self.folder.is_dir():
            return []
        return sorted(p.name for p in self.folder.iterdir() if p.is_dir())

    def pending_messages(self, recipient: str) -> list[ProposedFileChangesMessage]:
        """The saved messages to ``recipient``, oldest first.

        Raises:
            ValueError: if an entry is of a later version, or has another shape.
        """
        paths = self.recipient_folder(recipient).glob(f"*{SPOOL_ENTRY_SUFFIX}")
        messages = [
            ProposedFileChangesMessage.from_compressed_data(
                SpoolEntry.read(p).compressed_message
            )
            for p in paths
        ]
        return sorted(messages, key=lambda m: m.message_filename.submitted_timestamp)

    def remove(self, recipient: str, msg_id: str) -> None:
        """Remove the saved message ``msg_id`` to ``recipient``."""
        self._entry_path(recipient, msg_id).unlink()

    def locked(self) -> AbstractContextManager[None]:
        """A lock that one process at a time holds while it uploads."""
        return locked_file(self.folder)

    def recipient_folder(self, recipient: str) -> Path:
        return self.folder / recipient

    def _entry_path(self, recipient: str, msg_id: str) -> Path:
        return self.recipient_folder(recipient) / f"{msg_id}{SPOOL_ENTRY_SUFFIX}"
