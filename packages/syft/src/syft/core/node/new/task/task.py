# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# third party
from pydantic import BaseModel

# relative
from .....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from .....core.node.common.node_table.syft_object import SyftObject
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ...abstract.node import AbstractNodeClient


# TODO: 🟡 Duplication of PyPrimitive Dict
# This is emulated since the action store curently accepts  only SyftObject types
@serializable(recursive_serde=True)
class DictObject(SyftObject):
    # version
    __canonical_name__ = "Dict"
    __version__ = SYFT_OBJECT_VERSION_1

    base_dict: Dict[Any, Any] = {}

    # serde / storage rules
    __attr_state__ = ["id", "base_dict"]

    __attr_searchable__ = []
    __attr_unique__ = ["id"]

    def __repr__(self) -> str:
        return self.base_dict.__repr__()


@serializable(recursive_serde=True)
class NodeView(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str
    node_uid: UID

    @staticmethod
    def from_client(client: AbstractNodeClient):
        return NodeView(name=client.name, node_uid=client.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NodeView):
            return False
        return self.name == other.name and self.node_uid == other.node_uid

    def __hash__(self) -> int:
        return hash((self.name, self.node_uid))


@serializable(recursive_serde=True)
class Task(SyftObject):
    # version
    __canonical_name__ = "Task"
    __version__ = SYFT_OBJECT_VERSION_1

    # fields
    user: str
    inputs: Dict[NodeView, dict]
    owners: List[NodeView]
    code: str
    status: Dict[NodeView, str]
    created_at: str
    updated_at: str
    reviewed_by: str
    execution: str
    outputs: Dict
    reason: str = ""
    oblv_metadata: Optional[Dict] = None

    # serde / storage rules
    __attr_state__ = [
        "id",
        "owners",
        "code",
        "user",
        "status",
        "inputs",
        "outputs",
        "created_at",
        "updated_at",
        "reviewed_by",
        "execution",
        "reason",
        "oblv_metadata",
    ]

    __attr_searchable__ = ["id", "user"]
    __attr_unique__ = ["id"]
