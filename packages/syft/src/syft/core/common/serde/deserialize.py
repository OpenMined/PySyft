# stdlib
import re
from typing import Any

# third party
from google.protobuf.message import Message

# relative
from ....logger import traceback_and_raise
from ....proto.util.data_message_pb2 import DataMessage
from ....util import index_syft_by_module_name
from .capnp import CAPNP_END_MAGIC_HEADER_BYTES
from .capnp import CAPNP_REGISTRY
from .capnp import CAPNP_START_MAGIC_HEADER
from .capnp import CAPNP_START_MAGIC_HEADER_BYTES
from .types import Deserializeable
from .recursive import RecursiveSerde_PB
from .recursive import recursive_serde
from .recursive import rs_proto2object

PROTOBUF_START_MAGIC_HEADER = "protobuf:"
PROTOBUF_START_MAGIC_HEADER_BYTES = PROTOBUF_START_MAGIC_HEADER.encode("utf-8")


def _deserialize(
    blob: Deserializeable,
    from_proto: bool = True,
    from_bytes: bool = False,
) -> Any:

    if from_bytes:
        message = RecursiveSerde_PB()
        message.ParseFromString(blob)
        blob = message

    if not isinstance(blob, RecursiveSerde_PB):
        raise TypeError(f"Wrong deserialization format.")

    return rs_proto2object(blob)


class CapnpMagicBytesNotFound(Exception):
    pass


def deserialize_capnp(buf: bytes) -> Any:
    # only search 1000 bytes to prevent wasting time on large files
    search_range = 1000
    header_bytes = buf[0:search_range]
    chars = bytearray()
    # filter header bytes
    for i in header_bytes:
        # only allow ascii letters or : in headers and class name to prevent lookup
        # breaking somehow, when packing weird stuff like \x03 ends up in the string
        # e.g. PhiTensor -> ND\x03imEntityPhiTensor
        if i in range(65, 91) or i in range(97, 123) or i == 58:
            chars.append(i)
    header_bytes = bytes(chars)

    proto_start_index = header_bytes.find(PROTOBUF_START_MAGIC_HEADER_BYTES)
    start_index = header_bytes.find(CAPNP_START_MAGIC_HEADER_BYTES)
    if proto_start_index != -1 and (proto_start_index < start_index):
        # we have protobuf on the outside
        raise CapnpMagicBytesNotFound(
            f"protobuf Magic Header {PROTOBUF_START_MAGIC_HEADER} found in bytes"
        )
    if start_index == -1:
        raise CapnpMagicBytesNotFound(
            f"capnp Magic Header {CAPNP_START_MAGIC_HEADER}" + "not found in bytes"
        )
    start_index += len(CAPNP_START_MAGIC_HEADER_BYTES)
    end_index = start_index + header_bytes[start_index:].index(
        CAPNP_END_MAGIC_HEADER_BYTES
    )
    class_name_bytes = header_bytes[start_index:end_index]
    class_name = class_name_bytes.decode("utf-8")

    if end_index <= start_index:
        raise ValueError("End Index should always be greater than Start index")

    if class_name not in CAPNP_REGISTRY:
        raise Exception(
            f"Found capnp Magic Header: {CAPNP_START_MAGIC_HEADER} "
            + f"and Class {class_name} but no mapping in capnp registry {CAPNP_REGISTRY}"
        )
    return CAPNP_REGISTRY[class_name](buf=buf)
