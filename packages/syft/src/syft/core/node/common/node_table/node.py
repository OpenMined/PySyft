# stdlib
from typing import Dict
from typing import List
from typing import Optional

# relative
from .syft_object import SyftObject


class NoSQLNode(SyftObject):
    # version
    __canonical_name__ = "Node"
    __version__ = 1

    # fields
    node_uid: str
    node_name: str
    node_type: Optional[str]
    verify_key: Optional[str]
    keep_connected: Optional[bool] = True
    node_route: List[Dict] = []

    # serde / storage rules
    __attr_state__ = [
        "node_uid",
        "node_name",
        "node_type",
        "verify_key",
        "keep_connected",
        "node_route",
    ]

    __attr_searchable__ = ["node_uid", "verify_key"]
    __attr_unique__ = ["node_uid"]
