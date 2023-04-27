# stdlib
from typing import Any
from pydantic.networks import EmailStr


def _serialize(
    obj: object,
    to_proto: bool = True,
    to_bytes: bool = False,
) -> Any:
    # relative
    from .recursive import rs_object2proto

    if isinstance(obj, EmailStr):
        obj = str(obj)

    proto = rs_object2proto(obj)

    if to_bytes:
        return proto.to_bytes()

    if to_proto:
        return proto
