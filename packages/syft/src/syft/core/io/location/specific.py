# stdlib
from typing import Optional

# relative
from ...common.object import ObjectWithID
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .location import Location


@serializable(recursive_serde=True)
class SpecificLocation(ObjectWithID, Location):
    """This represents the location of a single Node object
    represented by a single UID. It may not have any functionality
    beyond Location but there is logic, which interprets it differently."""

    __attr_allowlist__ = ["id", "name"]

    def __init__(self, id: Optional[UID] = None, name: Optional[str] = None):
        ObjectWithID.__init__(self, id=id)
        self.name = name if name is not None else self.name

    @property
    def icon(self) -> str:
        return "📌"

    @property
    def pprint(self) -> str:
        output = f"{self.icon} {self.name} ({str(self.__class__.__name__)})@{self.id.emoji()}"
        return output
