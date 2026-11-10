"""The messages a DS saves before it sends them.

A dropped connection or a stopped kernel must not lose a job submission. Each
message to a peer is saved to ``<syftbox>/<me>/private/spool/<peer>/<msg_id>/``
before any network call, and it is removed when the peer has it. ``private/``
is never synced.

An entry is a folder. ``entry.json`` holds the message without the content of
its files, and each content is in its own file next to it, so only an upload
reads the content. ``add`` writes ``entry.json`` last and ``remove`` deletes it
first: a folder without it is not an entry.

A saved message is the message before the downgrade and the encryption for its
peer. The upload does both, so it uses the protocol and the key that the peer
has at that time. An enclave gets a new key at every boot.
"""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from syft.sync.messages.proposed_filechange import (
    MessageFileName,
    ProposedFileChangesMessage,
)
from syft.sync.peers.peer_store import PRIVATE_DIR_NAME
from syft.sync.sync.caches.persisted_dict import locked_file, write_text_atomic
from syft.sync.utils.versioned_file import read_versioned_json

SPOOL_DIR_NAME = "spool"
SPOOL_ENTRY_FILE_NAME = "entry.json"
CONTENT_FILE_SUFFIX = ".content"
# Format of a spool entry. Raise it when the layout of an entry changes. An
# entry of a later version is refused.
SPOOL_ENTRY_VERSION = 2

ContentType = Literal["text", "binary"]


def datasite_spool_path(syftbox_folder: Path | str, email: str) -> Path:
    """Per-datasite spool: ``<syftbox_folder>/<email>/private/spool``."""
    return Path(syftbox_folder) / email / PRIVATE_DIR_NAME / SPOOL_DIR_NAME


class SpoolEntry(BaseModel):
    """One saved message, without the content of its files."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    # The message before the downgrade and the encryption for its peer. Its
    # changes have no content.
    message: ProposedFileChangesMessage
    # The type of the content of each change, in order. None for a delete.
    content_types: list[ContentType | None]

    @classmethod
    def read(cls, path: Path) -> "SpoolEntry":
        """Read the entry at ``path``.

        Raises:
            ValueError: if a later client wrote the entry, or it has another
                shape.
        """
        return read_versioned_json(path, cls, SPOOL_ENTRY_VERSION, "spool entry")


def _content_path(entry_folder: Path, index: int) -> Path:
    """The file that holds the content of change ``index`` of an entry."""
    return entry_folder / f"{index}{CONTENT_FILE_SUFFIX}"


def _content_bytes(content: str | bytes) -> bytes:
    return content.encode("utf-8") if isinstance(content, str) else content


def _read_content(path: Path, content_type: ContentType | None) -> str | bytes | None:
    if content_type is None:
        return None
    data = path.read_bytes()
    return data.decode("utf-8") if content_type == "text" else data


class Spool:
    """The saved messages of one DS, in one folder per recipient. An entry is
    a folder named after the ``msg_id`` its message gets."""

    def __init__(self, folder: Path):
        self.folder = folder

    def add(self, recipient: str, message: ProposedFileChangesMessage) -> None:
        """Save ``message`` to ``recipient``."""
        msg_id = message.message_filename.as_string()
        entry_folder = self._entry_folder(recipient, msg_id)
        entry_folder.mkdir(parents=True)
        changes = message.proposed_file_changes
        for index, change in enumerate(changes):
            if change.content is not None:
                _content_path(entry_folder, index).write_bytes(
                    _content_bytes(change.content)
                )
        changes_without_content = [
            change.model_copy(update={"content": None}) for change in changes
        ]
        entry = SpoolEntry(
            version=SPOOL_ENTRY_VERSION,
            message=message.model_copy(
                update={"proposed_file_changes": changes_without_content}
            ),
            content_types=[change.content_type for change in changes],
        )
        write_text_atomic(entry_folder / SPOOL_ENTRY_FILE_NAME, entry.model_dump_json())

    def recipients(self) -> list[str]:
        """The recipients that have a folder in the spool."""
        if not self.folder.is_dir():
            return []
        return sorted(p.name for p in self.folder.iterdir() if p.is_dir())

    def msg_ids(self, recipient: str) -> list[str]:
        """The ids of the saved messages to ``recipient``, oldest first."""
        msg_ids = [
            p.name
            for p in self.recipient_folder(recipient).iterdir()
            if (p / SPOOL_ENTRY_FILE_NAME).is_file()
        ]
        return sorted(
            msg_ids, key=lambda m: MessageFileName.from_string(m).submitted_timestamp
        )

    def read_entry(self, recipient: str, msg_id: str) -> SpoolEntry:
        """The saved message ``msg_id`` to ``recipient``, without its content.

        Raises:
            ValueError: if the entry is of a later version, or has another shape.
        """
        return SpoolEntry.read(
            self._entry_folder(recipient, msg_id) / SPOOL_ENTRY_FILE_NAME
        )

    def read_message(self, recipient: str, msg_id: str) -> ProposedFileChangesMessage:
        """The saved message ``msg_id`` to ``recipient``, with its content.

        Raises:
            ValueError: if the entry is of a later version, or has another shape.
        """
        entry_folder = self._entry_folder(recipient, msg_id)
        entry = self.read_entry(recipient, msg_id)
        changes_and_types = zip(
            entry.message.proposed_file_changes, entry.content_types, strict=True
        )
        changes = [
            change.model_copy(
                update={
                    "content": _read_content(
                        _content_path(entry_folder, index), content_type
                    )
                }
            )
            for index, (change, content_type) in enumerate(changes_and_types)
        ]
        return entry.message.model_copy(update={"proposed_file_changes": changes})

    def remove(self, recipient: str, msg_id: str) -> None:
        """Remove the saved message ``msg_id`` to ``recipient``."""
        entry_folder = self._entry_folder(recipient, msg_id)
        (entry_folder / SPOOL_ENTRY_FILE_NAME).unlink()
        for content_path in entry_folder.iterdir():
            content_path.unlink()
        entry_folder.rmdir()

    def locked(self) -> AbstractContextManager[None]:
        """A lock that one process at a time holds while it uploads."""
        return locked_file(self.folder)

    def recipient_folder(self, recipient: str) -> Path:
        return self.folder / recipient

    def _entry_folder(self, recipient: str, msg_id: str) -> Path:
        return self.recipient_folder(recipient) / msg_id
