# stdlib
from typing import Any


def _serialize(
    obj: object,
    to_proto: bool = True,
    to_bytes: bool = False,
) -> Any:
    # relative
    from .recursive import rs_object2proto

    proto = rs_object2proto(obj)

    if to_bytes:
        return proto.to_bytes()

    if to_proto:
        return proto
