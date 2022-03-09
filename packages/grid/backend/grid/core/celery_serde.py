# stdlib
from typing import Any

# third party
from kombu import serialization

# syft absolute
import syft as sy
from syft.logger import error


def loads(data: bytes) -> Any:
    # original payload might have nested bytes in the args
    org_payload = sy.deserialize(data, from_bytes=True).upcast()
    # original payload is found at org_payload[0][0]
    if (
        len(org_payload) > 0
        and len(org_payload[0]) > 0
        and isinstance(org_payload[0][0], bytes)
    ):
        try:
            nested_data = org_payload[0][0]
            org_obj = sy.deserialize(nested_data, from_bytes=True)
            org_payload[0][0] = org_obj
        except Exception as e:
            error(f"Unable to deserialize nested payload. {e}")
            raise e

    return org_payload


def dumps(obj: Any) -> bytes:
    # this is usually a Tuple of args where the first one is what we send to the task
    # but it can also get other arbitrary data which we need to serde
    # since we might get bytes directly from the web endpoint we can avoid double
    # unserializing it by keeping it inside the nested args list org_payload[0][0]
    return sy.serialize(obj, to_bytes=True)


serialization.register(
    "syft", dumps, loads, content_type="application/syft", content_encoding="binary"
)
