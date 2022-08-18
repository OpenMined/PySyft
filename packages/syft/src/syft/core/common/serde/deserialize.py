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

PROTOBUF_START_MAGIC_HEADER = "protobuf:"
PROTOBUF_START_MAGIC_HEADER_BYTES = PROTOBUF_START_MAGIC_HEADER.encode("utf-8")


def _deserialize(
    blob: Deserializeable,
    from_proto: bool = True,
    from_bytes: bool = False,
) -> Any:
    deserialization_error = TypeError(
        "You tried to deserialize an unsupported type. This can be caused by "
        "several reasons. Either you are actively writing Syft code and forgot "
        "to create one, or you are trying to deserialize an object which was "
        "serialized using a different version of Syft and the object you tried "
        "to deserialize is not supported in this version."
    )

    # try to decode capnp first
    if isinstance(blob, bytes):
        try:
            return deserialize_capnp(buf=blob)
        except CapnpMagicBytesNotFound:  # nosec
            # probably not capnp bytes
            pass
        except Exception as e:
            # capnp magic bytes found but another problem has occured
            print("failed capnp deserialize", e)
            raise e

    # relative
    from .recursive import RecursiveSerde_PB
    from .recursive import recursive_serde
    from .recursive import rs_proto2object

    def parse_recursive_serde_object(recursive_serde_blob: bytes) -> object:
        message = RecursiveSerde_PB()
        message.ParseFromString(recursive_serde_blob)
        return rs_proto2object(message)

    if from_bytes:
        # stdlib
        import sys

        data_message = DataMessage()
        data_message.ParseFromString(blob)
        module_parts = data_message.obj_type.split(".")
        klass = module_parts.pop()

        if klass == "NoneType":
            obj_type = None
        else:
            obj_type = getattr(sys.modules[".".join(module_parts)], klass)

        if recursive_serde(obj_type):
            return parse_recursive_serde_object(data_message.content)
        else:
            get_protobuf_schema = getattr(obj_type, "get_protobuf_schema", None)

        if not callable(get_protobuf_schema):
            traceback_and_raise(deserialization_error)

        protobuf_type = get_protobuf_schema()
        blob = protobuf_type()

        if not isinstance(blob, Message):
            traceback_and_raise(deserialization_error)

        blob.ParseFromString(data_message.content)

    obj_type = getattr(type(blob), "schema2type", None)

    if obj_type is not None:
        _proto2object = getattr(obj_type, "_proto2object", None)
        res = _proto2object(proto=blob)
        return res

    if isinstance(blob, RecursiveSerde_PB):
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
