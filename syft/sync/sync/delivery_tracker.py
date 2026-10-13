"""What each peer was sent, so the data owner can resend, delete and catch up.

STUB: the names and contracts are agreed; the bodies are to be written (PR 2).
A draft implementation exists from the planning work.

For every peer the tracker keeps a view: datasite path -> hash of the version
that peer holds. Each sync compares what a peer may read with its view:

    the peer holds an old version       -> send the new one
    the peer holds nothing (new peer)   -> send every file it may read
    the peer may no longer read a file  -> send a delete
    the owner deleted the file          -> send a delete

The view changes only after the message carrying a version was written to the
peer's outbox, so a crash before the write leaves it unchanged and the
delivery is retried. Paths in flight are written down before routing and
cleared after the last outbox write, so a crash in between is finished by the
next sync.

Open decisions (see research/merged_transport_plan.md):
- First run with no views: assume current peers hold what they may read
  (proposed), or resend everything.
- A rejected peer approved again: reset its view and resend (proposed).
- Enclaves lose the views at every boot until they are also kept on Drive.
"""

from pathlib import Path
from typing import Iterable

from syft.sync.events.file_change_event import FileChangeEvent

DELIVERY_VIEWS_FILENAME = "owner_delivery_views.json"
# Format of the views file. Raise it when the layout changes, and keep a read
# path for every earlier version.
DELIVERY_VIEWS_VERSION = 1


class DeliveryTracker:
    """Per-peer views of delivered file versions, persisted next to the owner cache.

    ``path`` None keeps the views in memory only, as the in-memory owner cache
    does in tests.
    """

    def __init__(self, path: Path | None = None):
        self.path = path

    # -- reading ---------------------------------------------------------------
    def reload(self) -> None:
        """Read the views from disk once; lookups use them until the next reload."""
        raise NotImplementedError

    @property
    def initialized(self) -> bool:
        """Whether views were ever recorded for this datasite."""
        raise NotImplementedError

    def has_view(self, peer: str) -> bool:
        """Whether ``peer`` was ever delivered to; False means a new peer."""
        raise NotImplementedError

    def delivered_hash(self, peer: str, path: str) -> str | None:
        """The hash of the version of ``path`` that ``peer`` holds, or None."""
        raise NotImplementedError

    def holders(self, path: str, peers: Iterable[str]) -> frozenset[str]:
        """The peers among ``peers`` that hold some version of ``path``."""
        raise NotImplementedError

    # -- writing ---------------------------------------------------------------
    def mark_initialized(self) -> None:
        """Record that views exist, with DELIVERY_VIEWS_VERSION."""
        raise NotImplementedError

    def set_view(self, peer: str, view: dict[str, str]) -> None:
        """Replace ``peer``'s whole view, e.g. to seed it on the first run."""
        raise NotImplementedError

    def record(self, peer: str, events: Iterable[FileChangeEvent]) -> None:
        """Record that ``events`` reached ``peer``'s outbox.

        A delete removes the path from the view; any other event sets its hash.
        Call only after the outbox write succeeded.
        """
        raise NotImplementedError

    def reset(self, peer: str) -> None:
        """Forget ``peer``'s view, so the next sync sends it everything again."""
        raise NotImplementedError

    # -- paths in flight -------------------------------------------------------
    @property
    def pending(self) -> list[str]:
        """Paths whose routing started but did not finish."""
        raise NotImplementedError

    def mark_pending(self, paths: Iterable[str]) -> None:
        """Write ``paths`` down before routing them."""
        raise NotImplementedError

    def clear_pending(self) -> None:
        """Clear the paths in flight once every outbox write is done."""
        raise NotImplementedError
