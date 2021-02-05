# syft absolute
from syft.lib.python.string import String


def string_serde(data: str) -> None:
    syft_string = String(data)

    serialized = syft_string._object2proto()
    String._proto2object(proto=serialized)
