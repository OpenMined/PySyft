"""Messages held until encryption with their recipient is set up.

The list persists to a JSON file in the cache folder of the datasite, so a
held message survives a restart. The file is the only copy: a process takes
the cross-process lock, sends what it can, and writes back the rest, so two
processes on one datasite (a notebook and syft-bg) never send one message
twice.
"""

import base64
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import portalocker

# Format of the file: {"version": N, "messages": [...]}. Raise it when the
# layout of a message entry changes.
HELD_MESSAGES_VERSION = 1

HeldMessage = tuple[str, bytes]  # (recipient email, compressed message)


class HeldMessages:
    """A list of held messages, persisted to ``path``."""

    def __init__(self, path: Path):
        self._path = Path(path)

    @contextmanager
    def edit(self) -> Iterator[list[HeldMessage]]:
        """Hold the lock, yield the list, and save the changes made to it.

        When the block raises, the file stays as it was.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path.with_suffix(".lock"), "a") as lock_file:
            portalocker.lock(lock_file, portalocker.LOCK_EX)
            try:
                messages = self._read()
                yield messages
                self._write(messages)
            finally:
                portalocker.unlock(lock_file)

    def read(self) -> list[HeldMessage]:
        """The held messages, read under the lock, without a write."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path.with_suffix(".lock"), "a") as lock_file:
            portalocker.lock(lock_file, portalocker.LOCK_EX)
            try:
                return self._read()
            finally:
                portalocker.unlock(lock_file)

    def send_all(
        self,
        send: Callable[[str, Any], bool],
        queued: Iterable[tuple[str, Any]] = (),
        encode: Callable[[Any], bytes] = lambda message: message,
    ) -> int:
        """Send the held messages, then ``queued``, in order for each recipient.

        ``send(recipient, message)`` gets the held bytes, or a queued message.
        It returns False while encryption with the recipient is pending: that
        message, and later ones to the same recipient, stay held (``encode``
        gives the bytes of a queued message). True means sent, or dropped on
        purpose. When ``send`` raises, the messages not handled yet stay held,
        so none is lost and none is sent twice.

        Returns the number of messages held after the call.
        """
        with self.edit() as held:
            items = list(held) + list(queued)
            remaining: list[tuple[str, Any]] = []
            waiting: set[str] = set()
            done = 0
            try:
                for recipient, message in items:
                    if recipient in waiting or not send(recipient, message):
                        waiting.add(recipient)
                        remaining.append((recipient, message))
                    done += 1
            finally:
                held[:] = [(r, encode(m)) for r, m in remaining + items[done:]]
                self._write(held)
            return len(held)

    def stamp(self) -> str | None:
        """Changes when the content changes: a hash of the file, or None without one.

        Not the modification time: some mounts keep it only to the second.
        """
        try:
            return hashlib.sha256(self._path.read_bytes()).hexdigest()
        except FileNotFoundError:
            return None

    def __len__(self) -> int:
        return len(self.read())

    def _read(self) -> list[HeldMessage]:
        if not self._path.exists():
            return []
        data = json.loads(self._path.read_text())
        version = data.get("version")
        if version != HELD_MESSAGES_VERSION:
            # Held messages are user data, so a file this client cannot read
            # is refused, not dropped.
            raise ValueError(
                f"The held messages in {self._path} have version {version}, and "
                f"this client reads version {HELD_MESSAGES_VERSION}."
            )
        return [(m["recipient"], base64.b64decode(m["data"])) for m in data["messages"]]

    def _write(self, messages: list[HeldMessage]) -> None:
        if not messages:
            self._path.unlink(missing_ok=True)
            return
        data = {
            "version": HELD_MESSAGES_VERSION,
            "messages": [
                {"recipient": r, "data": base64.b64encode(d).decode("ascii")}
                for r, d in messages
            ],
        }
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(data))
            os.replace(tmp, self._path)
        finally:
            Path(tmp).unlink(missing_ok=True)
