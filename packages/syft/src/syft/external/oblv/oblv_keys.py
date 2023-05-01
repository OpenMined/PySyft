# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject


@serializable()
class OblvKeys(SyftObject):
    # version
    __canonical_name__ = "OblvKeys"
    __version__ = SYFT_OBJECT_VERSION_1

    # fields
    public_key: bytes
    private_key: bytes

    # serde / storage rules
    __attr_searchable__ = ["private_key", "public_key"]
    __attr_unique__ = ["private_key", "public_key"]
