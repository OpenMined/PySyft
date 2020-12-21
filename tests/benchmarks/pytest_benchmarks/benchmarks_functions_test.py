# syft absolute
from syft.lib.python.string import String


def string_serde() -> None:
    syft_string = String("Hello OpenMined")

    serialized = syft_string._object2proto()
    String._proto2object(proto=serialized)
