# stdlib
from typing import Optional

# relative
from ....logger import traceback_and_raise
from ....util import random_name
from ...common.serde.serializable import serializable
from ...common.uid import UID


@serializable(recursive_serde=True)
class Location:
    """This represents the location of a node, including
    location-relevant metadata (such as how long it takes
    for us to communicate with this location, etc.)"""

    __attr_allowlist__ = ["name"]

    def __init__(self, name: Optional[str] = None) -> None:
        if name is None:
            name = random_name()
        self.name = name
        super().__init__()

    @property
    def id(self) -> UID:
        traceback_and_raise(NotImplementedError)

    def repr_short(self) -> str:
        """Returns a SHORT human-readable version of the ID

        Return a SHORT human-readable version of the ID which
        makes it print nicer when embedded (often alongside other
        UID objects) within other object __repr__ methods."""

        return self.__repr__()
