"""A queue of messages kept in one JSON file, so it survives a restart."""

from collections.abc import Collection
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from syft.sync.connections.types import MessageKind
from syft.sync.sync.caches.persisted_dict import locked_file, write_text_atomic

# Bytes travel and are stored as base64 inside JSON.
BYTES_AS_BASE64 = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")


class QueuedMessage(BaseModel):
    """One message, as the sender's router encoded it."""

    model_config = ConfigDict(frozen=True, **BYTES_AS_BASE64)

    msg_id: str
    kind: MessageKind
    data: bytes


class _QueueFile(BaseModel):
    model_config = BYTES_AS_BASE64

    messages: list[QueuedMessage] = []


class MessageQueue:
    """Messages in one file, oldest first. Every change holds the file lock."""

    def __init__(self, path: Path):
        self.path = path

    def all(self) -> list[QueuedMessage]:
        with locked_file(self.path, exclusive=False):
            return self._load()

    def add(self, message: QueuedMessage) -> None:
        """Append ``message``, unless one with its ``msg_id`` is already here."""
        with locked_file(self.path):
            messages = self._load()
            if all(m.msg_id != message.msg_id for m in messages):
                self._save([*messages, message])

    def remove(self, msg_ids: Collection[str]) -> list[str]:
        """Remove the messages with these ids; return the ids that were here."""
        with locked_file(self.path):
            messages = self._load()
            kept = [m for m in messages if m.msg_id not in msg_ids]
            self._save(kept)
        return [m.msg_id for m in messages if m.msg_id in msg_ids]

    def _load(self) -> list[QueuedMessage]:
        if not self.path.exists():
            return []
        return _QueueFile.model_validate_json(self.path.read_text()).messages

    def _save(self, messages: list[QueuedMessage]) -> None:
        write_text_atomic(self.path, _QueueFile(messages=messages).model_dump_json())
