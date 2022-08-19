# relative
from ....util import validate_type
from .deserialize import PROTOBUF_START_MAGIC_HEADER
from .recursive import recursive_serde
from .recursive import rs_object2proto
from .types import Deserializeable


def create_protobuf_magic_header() -> str:
    return f"{PROTOBUF_START_MAGIC_HEADER}"


def _serialize(
    obj: object,
    to_proto: bool = True,
    to_bytes: bool = False,
) -> Deserializeable:
    # capnp_bytes=True
    if hasattr(obj, "_object2bytes"):
        # capnp proto
        return validate_type(obj._object2bytes(), bytes)

    proto = rs_object2proto(obj)

    if to_bytes:
        return proto.SerializeToString()
    else:
        return proto
