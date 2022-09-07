# relative
from ....util import validate_type
from .recursive import rs_object2proto
from .types import Deserializeable


def _serialize(
    obj: object,
    to_proto: bool = True,
    to_bytes: bool = False,
) -> Deserializeable:
    # capnp_bytes=True
    if hasattr(obj, "_object2bytes"):
        # capnp proto
        return validate_type(obj._object2bytes(), bytes)  # type: ignore

    proto = rs_object2proto(obj)

    if to_bytes:
        return proto.SerializeToString()
    else:
        return proto
