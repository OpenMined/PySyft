# relative
from .....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from .....core.node.common.node_table.syft_object import SyftObject
from ....common.serde.serializable import serializable


@serializable(recursive_serde=True)
class OblvKeys(SyftObject):
    # version
    __canonical_name__ = "OblvKeys"
    __version__ = SYFT_OBJECT_VERSION_1

    # fields
    public_key: bytes
    private_key: bytes

    # serde / storage rules
    __attr_state__ = ["public_key", "private_key"]
    __attr_searchable__ = ["private_key", "public_key"]
    __attr_unique__ = ["private_key", "public_key"]
