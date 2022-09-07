# stdlib
from typing import Any

# relative
from .recursive import RecursiveSerde_PB
from .recursive import rs_proto2object
from .types import Deserializeable


def _deserialize(
    blob: Deserializeable,
    from_proto: bool = True,
    from_bytes: bool = False,
) -> Any:
    if from_bytes:
        # try to decode capnp first
        if isinstance(blob, bytes):
            # relative
            from .capnp import CapnpMagicBytesNotFound
            from .capnp import deserialize_capnp

            try:
                return deserialize_capnp(buf=blob)
            except CapnpMagicBytesNotFound:  # nosec
                # probably not capnp bytes
                pass
            except Exception as e:
                # capnp magic bytes found but another problem has occured
                print("failed capnp deserialize", e)
                raise e

        message = RecursiveSerde_PB()
        message.ParseFromString(blob)
        blob = message

    if not isinstance(blob, RecursiveSerde_PB):
        raise TypeError("Wrong deserialization format.")

    return rs_proto2object(blob)
