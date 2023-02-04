# stdlib
from typing import Dict

# relative
from .....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from .....core.node.common.node_table.syft_object import SyftObject
from ....common.serde.serializable import serializable


@serializable(recursive_serde=True)
class Task(SyftObject):
    # version
    __canonical_name__ = "Task"
    __version__ = SYFT_OBJECT_VERSION_1

    # fields
    user: str
    inputs: Dict[str, str]
    owner: Dict[str, str]
    code: str
    status: str
    created_at: str
    updated_at: str
    reviewed_by: str
    execution: str
    outputs: Dict[str, str]
    reason: str = ""

    # serde / storage rules
    __attr_state__ = [
        "id",
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
        "owner",
    ]

    __attr_searchable__ = ["id", "status", "user"]
    __attr_unique__ = ["id"]
