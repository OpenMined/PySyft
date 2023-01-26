# stdlib
from typing import List
from typing import Optional

# relative
from ....common.serde.serializable import serializable
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject


@serializable(recursive_serde=True)
class NoSQLObjectMetadata(SyftObject):
    # version
    __canonical_name__ = "ObjectMetadata"
    __version__ = SYFT_OBJECT_VERSION_1

    # fields
    obj: str
    tags: List[str] = []
    description: str
    name: Optional[str]
    read_permissions: str
    search_permissions: str
    write_permissions: str
    is_proxy_dataset: bool = False

    # serde / storage rules
    __attr_state__ = [
        "obj"
        "tags"
        "description"
        "name"
        "read_permissions"
        "search_permissions"
        "write_permissions"
        "is_proxy_dataset"
    ]
    __attr_searchable__: List[str] = ["obj"]
    __attr_unique__: List[str] = []
