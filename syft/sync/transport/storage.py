"""Readable views over ``PrivateStore``: one for each role.

A backend implements ``PrivateStore`` once. Owner code holds an
``OwnerStorage`` and data scientist code holds a ``WatcherStorage``, so each
sees only its own methods and cannot write into the other's area.
"""

from syft.sync.transport.protocols import PrivateStore


class OwnerStorage:
    """The data owner's own storage: event log, checkpoints, rolling state."""

    def __init__(self, store: PrivateStore):
        self._store = store

    def append_event(self, name: str, data: bytes) -> str:
        return self._store.append_event(name, data)

    def fetch(self, handle: str) -> bytes:
        return self._store.fetch(handle)

    def fetch_events_since(self, since_timestamp: float) -> list[bytes]:
        return self._store.fetch_events_since(since_timestamp)

    def put_checkpoint(self, name: str, data: bytes) -> str:
        return self._store.put_checkpoint(name, data)

    def latest_checkpoint(self) -> bytes | None:
        return self._store.latest_checkpoint()

    def put_incremental(self, name: str, data: bytes) -> str:
        return self._store.put_incremental(name, data)

    def all_incrementals(self) -> list[bytes]:
        return self._store.all_incrementals()

    def put_rolling_state(self, name: str, data: bytes) -> str:
        return self._store.put_rolling_state(name, data)

    def get_rolling_state(self) -> bytes | None:
        return self._store.get_rolling_state()


class WatcherStorage:
    """The data scientist's memory of what one owner sent it.

    Stored in the scientist's own private storage, never shared, so it
    survives a lost local disk such as a Colab runtime reset.
    """

    def __init__(self, store: PrivateStore, peer: str):
        self._store = store
        self.peer = peer

    def put_checkpoint(self, name: str, data: bytes) -> str:
        return self._store.put_watcher_checkpoint(self.peer, name, data)

    def latest_checkpoint(self) -> bytes | None:
        return self._store.latest_watcher_checkpoint(self.peer)

    def put_rolling_state(self, name: str, data: bytes) -> str:
        return self._store.put_watcher_rolling_state(self.peer, name, data)

    def get_rolling_state(self) -> bytes | None:
        return self._store.get_watcher_rolling_state(self.peer)

    def reset(self) -> None:
        """Forget everything from this owner, before asking for a full resend."""
        self._store.delete_watcher_state(self.peer)
