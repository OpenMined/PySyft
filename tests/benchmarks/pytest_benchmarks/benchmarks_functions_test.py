# stdlib
from typing import List

# syft absolute
from syft.lib.python import List as SyList
from syft.lib.python.string import String


def string_serde(data: str) -> None:
    syft_string = String(data)

    serialized = syft_string._object2proto()
    String._proto2object(proto=serialized)


def list_serde(data: List[str]) -> None:
    syft_list = SyList(data)

    serialized = syft_list._object2proto()
    SyList._proto2object(proto=serialized)
