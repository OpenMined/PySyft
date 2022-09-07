# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# relative
from .user import SyftObject


class NoSQLAssociationRequest(SyftObject):
    # version
    __canonical_name__ = "AssociationRequest"
    __version__ = 1

    # fields
    id_int: int
    requested_date: str
    processed_date: str = ""
    node_name: str = ""
    node_address: str = ""
    name: str = ""
    email: str = ""
    reason: Optional[str] = ""
    status: str = ""
    source: str = ""
    target: str = ""

    # serde / storage rules
    __attr_state__ = [
        "id_int",
        "requested_date",
        "processed_date",
        "node_name",
        "node_address",
        "name",
        "email",
        "reason",
        "status",
        "source",
        "target",
    ]
    __attr_searchable__: List[str] = ["source", "target", "id_int"]
    __attr_unique__: List[str] = []

    def to_dict(self) -> Dict[Any, Any]:
        attr_dict = super().to_dict()
        del attr_dict["id"]
        attr_dict["association_id"] = str(attr_dict["id_int"])
        del attr_dict["id_int"]
        return attr_dict
