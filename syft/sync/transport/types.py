"""Types shared by every transport backend."""

from enum import Enum

from pydantic import BaseModel, ConfigDict


class Lane(str, Enum):
    """The two message streams between a pair of peers.

    The two streams run in opposite directions and live in different places,
    and a dual-role pair runs both at once, so a mailbox call always names its
    lane.
    """

    # Data scientist -> data owner: proposed file changes (job submissions).
    PROPOSALS = "proposals"
    # Data owner -> data scientist: accepted file change events (results).
    EVENTS = "events"


class Entry(BaseModel):
    """One stored message or file, as a backend lists it.

    ``name`` is the syft file name. ``handle`` is whatever the backend needs to
    fetch or acknowledge the entry again; callers only pass it back. It is None
    when the entry is known by name only.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    handle: str | None = None
