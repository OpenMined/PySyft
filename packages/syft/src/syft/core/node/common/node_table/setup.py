# stdlib
from typing import Any
from typing import Dict
from typing import List

# relative
from .user import SyftObject


class NoSQLSetup(SyftObject):
    # version
    __canonical_name__ = "Setup"
    __version__ = 1

    # fields
    id_int: int
    domain_name: str = ""
    description: str = ""
    contact: str = ""
    daa: bool = False
    node_uid: str = ""
    daa_document: bytes = b""
    tags: List[str] = []
    deployed_on: str
    signing_key: str

    # serde / storage rules
    __attr_state__ = [
        "id_int",
        "domain_name",
        "description",
        "contact" "daa" "node_uid",
        "daa_document" "tags",
        "deployed_on",
        "signing_key",
    ]
    __attr_searchable__ = ["node_uid", "id_int", "signing_key", "domain_name"]
    __attr_unique__ = ["node_uid"]

    def to_dict(self) -> Dict[Any, Any]:
        attr_dict = super().to_dict()
        del attr_dict["id"]
        del attr_dict["id_int"]
        del attr_dict["daa_document"]
        return attr_dict
