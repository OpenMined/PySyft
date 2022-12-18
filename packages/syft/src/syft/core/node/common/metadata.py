# stdlib
from typing import Optional

# relative
from ...common.serde.serializable import serializable
from ...common.uid import UID
from ...io.location import Location


@serializable(recursive_serde=True)
class Metadata:
    __attr_allowlist__ = ["name", "node", "id", "node_type", "version"]

    def __init__(
        self,
        node: Location,
        name: str = "",
        id: Optional[UID] = None,
        node_type: str = "",
        version: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.node = node
        if isinstance(id, UID):
            self.id = id
        else:
            self.id = UID()
        self.node_type = node_type
        self.version = version
