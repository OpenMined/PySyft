# stdlib
from typing import Any
from typing import Type


def _serialize(
    obj: object,
    to_proto: bool = True,
    to_bytes: bool = False,
    class_type: Type = type(None),
) -> Any:
    # relative
    from .recursive import rs_object2proto

    proto = rs_object2proto(obj, class_type=class_type)

    if to_bytes:
        return proto.to_bytes()

    if to_proto:
        return proto
