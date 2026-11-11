"""A queue of messages kept on disk, one file per message, so it survives a
restart and adding or removing one message never rewrites the others."""

from collections.abc import Collection
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from syft.sync.connections.types import MessageKind
from syft.sync.sync.caches.persisted_dict import locked_file

# Bytes travel as base64 inside a JSON frame.
BYTES_AS_BASE64 = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")


class QueuedMessage(BaseModel):
    """One message, as the sender's router encoded it."""

    model_config = ConfigDict(frozen=True, **BYTES_AS_BASE64)

    msg_id: str
    kind: MessageKind
    data: bytes


class _Entry(NamedTuple):
    """One message file, named ``<position>.<kind>.<msg_id>``."""

    position: int
    kind: MessageKind
    msg_id: str
    path: Path


class MessageQueue:
    """Messages in one folder, oldest first. Every change holds the folder lock."""

    def __init__(self, folder: Path):
        self.folder = folder

    def messages(
        self, kind: MessageKind | None = None, max_messages: int | None = None
    ) -> list[QueuedMessage]:
        """The messages, of ``kind`` when given, oldest first, at most
        ``max_messages``. Only those files are read."""
        with locked_file(self.folder, exclusive=False):
            entries = [e for e in self._entries() if kind is None or e.kind == kind]
            return [
                QueuedMessage(msg_id=e.msg_id, kind=e.kind, data=e.path.read_bytes())
                for e in entries[:max_messages]
            ]

    def is_empty(self) -> bool:
        with locked_file(self.folder, exclusive=False):
            return not self._entries()

    def add(self, message: QueuedMessage) -> None:
        """Append ``message``, unless one with its ``msg_id`` is already here."""
        _require_file_name(message.msg_id)
        with locked_file(self.folder):
            entries = self._entries()
            if any(e.msg_id == message.msg_id for e in entries):
                return
            position = entries[-1].position + 1 if entries else 0
            name = f"{position:012d}.{message.kind.value}.{message.msg_id}"
            # Written under a hidden name first, so a reader never sees half.
            tmp = self.folder / f".{name}.tmp"
            tmp.write_bytes(message.data)
            tmp.replace(self.folder / name)

    def remove(self, msg_ids: Collection[str]) -> list[str]:
        """Remove the messages with these ids; return the ids that were here."""
        with locked_file(self.folder):
            removed = [e for e in self._entries() if e.msg_id in msg_ids]
            for entry in removed:
                entry.path.unlink()
        return [e.msg_id for e in removed]

    def _entries(self) -> list[_Entry]:
        self.folder.mkdir(parents=True, exist_ok=True)
        entries = []
        for path in self.folder.iterdir():
            if path.name.startswith("."):
                continue
            position, kind, msg_id = path.name.split(".", 2)
            entries.append(_Entry(int(position), MessageKind(kind), msg_id, path))
        return sorted(entries, key=lambda e: e.position)


def _require_file_name(msg_id: str) -> None:
    """``msg_id`` comes from the peer and names a file, so it may not leave the
    queue folder or hide from it."""
    if msg_id != Path(msg_id).name or msg_id.startswith("."):
        raise ValueError(f"msg_id {msg_id!r} is not a plain file name")
