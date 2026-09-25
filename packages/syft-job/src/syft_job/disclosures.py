"""The items of job data that a data owner can release to the submitter."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from enum import Enum
from typing import Iterable, Union

from .traceback_capture import FRAMES_FILENAME

# The data owner's grant for a job with one data owner. It sits in review/.
DISCLOSURES_FILENAME = "disclosures.json"


class DisclosureItem(str, Enum):
    """A class of job data that a party can release to the other parties.

    - ``TRACEBACK_FRAMES``: the file and the line of each frame, and the
      exception type. Never the exception message.
    - ``LOGS``: stdout and stderr, as the job wrote them.
    - ``RETURN_CODE``: the exact exit code of the job.
    """

    TRACEBACK_FRAMES = "traceback_frames"
    LOGS = "logs"
    RETURN_CODE = "return_code"


DISCLOSURE_ITEMS = frozenset(item.value for item in DisclosureItem)

# The staged files that each item covers.
GATED_ARTIFACTS = {
    DisclosureItem.LOGS.value: ("stdout.txt", "stderr.txt"),
    DisclosureItem.TRACEBACK_FRAMES.value: (FRAMES_FILENAME,),
    DisclosureItem.RETURN_CODE.value: ("returncode.txt",),
}

DisclosuresArg = Union[str, DisclosureItem, Iterable[str], Mapping[str, bool], None]


def normalize_disclosures(items: DisclosuresArg) -> dict[str, bool]:
    """Return the known items in ``items`` as a map. Unknown names are dropped.

    A single name is accepted on its own, because iterating a string would
    produce its characters and grant nothing.

    A map is accepted in the form that this function returns, therefore a
    caller can read the current grant, set an item to False, and send it back
    to drop that item.
    """
    if not items:
        return {}
    if isinstance(items, (str, DisclosureItem)):
        items = [items]
    elif isinstance(items, Mapping):
        items = [name for name, allowed in items.items() if allowed]
    names = {str(getattr(i, "value", i)) for i in items}
    return {name: True for name in sorted(names & DISCLOSURE_ITEMS)}


def restrict_to_request(
    disclosures: DisclosuresArg, requested: Iterable[str]
) -> dict[str, bool]:
    """Return the known items in ``disclosures`` that ``requested`` also holds.

    The submitter can edit its request after the approval. A grant that is
    stored in this form cannot grow when the request grows.
    """
    requested = set(requested)
    return {
        name: True for name in normalize_disclosures(disclosures) if name in requested
    }


def check_approval_reason(reason: object) -> None:
    """Refuse a reason that is not a string, such as a list of items.

    ``approve(["logs"])`` would record the list as the reason and grant nothing.
    """
    if reason is not None and not isinstance(reason, str):
        raise TypeError(
            f"reason must be a string, not {type(reason).__name__}. "
            f"Give the items as disclosures=[...]."
        )


def gated_names(granted: Iterable[str]) -> list[str]:
    """The filenames that the granted disclosure items cover."""
    granted = set(granted)
    return [
        name
        for item, names in GATED_ARTIFACTS.items()
        if item in granted
        for name in names
    ]


LOGS_WARNING = (
    "This grant can send stdout and stderr to the submitter, as the job wrote "
    "them. The job code can print private data into them. Grant 'logs' only "
    "for code that you reviewed for this."
)


def warn_on_logs_release(
    granted: Iterable[str], requested: Iterable[str], stacklevel: int = 3
) -> None:
    """Warn when a grant releases the logs. An unrequested item releases nothing."""
    item = DisclosureItem.LOGS.value
    if item in set(granted) and item in set(requested):
        warnings.warn(LOGS_WARNING, UserWarning, stacklevel=stacklevel)


def format_items(items: Iterable[str]) -> str:
    """The item names for display, or "none"."""
    return ", ".join(sorted(items)) or "none"
