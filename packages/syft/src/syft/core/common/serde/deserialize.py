# stdlib
from typing import Any

# third party
from capnp.lib.capnp import _DynamicStructBuilder


def _deserialize(
    blob: Any,
    from_proto: bool = True,
    from_bytes: bool = False,
) -> Any:
    # relative
    from .recursive import rs_bytes2object
    from .recursive import rs_proto2object

    if (
        (from_bytes and not isinstance(blob, bytes))
        or (
            from_proto
            and not from_bytes
            and not isinstance(blob, _DynamicStructBuilder)
        )
        or not (from_bytes or from_proto)
    ):
        raise TypeError("Wrong deserialization format.")

    if from_bytes:
        # try to decode capnp first
        if isinstance(blob, bytes):
            # relative
            from .capnp import deserialize_capnp

            try:
                return deserialize_capnp(buf=blob)
            except Exception:  # nosec
                pass

        return rs_bytes2object(blob)

    if from_proto:
        return rs_proto2object(blob)
