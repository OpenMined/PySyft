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


# WARNING: This code has more 🐉 Dragons than a game of D&D 🗡🧙🎲
# you were warned...
# enter at your own peril...
# seriously, get some 🧪 HP Potions and 📜 TP Scrolls ready...
def _deserialize(
    blob: Deserializeable,
    from_proto: bool = True,
    from_bytes: bool = False,
) -> Any:
    """We assume you're deserializing a protobuf object by default

    This function deserializes from encoding to a Python object. There are a few ways of
    using this function:
    1. An Message object is passed,q this will transform a protobuf message into its associated
    class.
    the from_proto has to be set (it is by default).
    2. Bytes are passed. This requires the from_bytes flag set the schema_type specified.
    We cannot (and we should not) be able to get the schema_type from the binary representation.

    Note: The only format that does not require the schema_type is when we are passing
    Messages directly.

    Raises: ValueError if you are not setting one from_<protocol> flag.
            ValueError if you are deserializing a data type that requires a schema type and not
            providing one.
            TypeError if you are trying to deserialize an unsupported type.

    :param blob: this parameter is the data to be deserialized from various formats.
    :type blob: Union[str, dict, bytes, Messages]
    :param from_proto: set this flag to True if you want to deserialize a protobuf message.
    :param from_bytes: set this flag to True if you want to deserialize a binary object.
    :type from_bytes: bool
    :return: a deserialized form of the object on which _deserialize() is called.
    :rtype: Serializable
    """

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

    if from_bytes:
        data_message = DataMessage()
        data_message.ParseFromString(blob)
        obj_type = index_syft_by_module_name(fully_qualified_name=data_message.obj_type)
        get_protobuf_schema = getattr(obj_type, "get_protobuf_schema", None)

        if not callable(get_protobuf_schema):
            traceback_and_raise(deserialization_error)

        protobuf_type = get_protobuf_schema()
        blob = protobuf_type()

        if not isinstance(blob, Message):
            traceback_and_raise(deserialization_error)

        blob.ParseFromString(data_message.content)

    # lets try to lookup the type we are deserializing
    # TODO: This needs to be cleaned up in GenerateWrapper and made more consistent.
    # There are serveral code paths that come through here and use different ways to
    # match and overload protobuf -> deserialize type
    obj_type = getattr(type(blob), "schema2type", None)
    # relative
    from .recursive import rs_get_protobuf_schema
    from .recursive import rs_proto2object

    if obj_type is None:
        # TODO: This can probably be removed now we have lists of obj_types
        obj_type = getattr(blob, "obj_type", None)
        if isinstance(blob, rs_get_protobuf_schema()):
            res = rs_proto2object(proto=blob)
            if getattr(res, "temporary_box", False) and hasattr(res, "upcast"):
                return res.upcast()
            return res

        if obj_type is None:
            traceback_and_raise(deserialization_error)

        obj_type = index_syft_by_module_name(fully_qualified_name=obj_type)  # type: ignore
        obj_type = getattr(obj_type, "_sy_serializable_wrapper_type", obj_type)
    elif isinstance(obj_type, list):
        if isinstance(blob, rs_get_protobuf_schema()):
            res = rs_proto2object(proto=blob)
            if getattr(res, "temporary_box", False) and hasattr(res, "upcast"):
                return res.upcast()
            return res
        elif len(obj_type) == 1:
            obj_type = obj_type[0]
        else:
            # this means we have multiple classes that use the same proto but use the
            # obj_type field to differentiate, so lets figure out which one in the list
            obj_type_re = r'obj_type: "(.+)"'
            # the first obj_type in the protobuf will be the main outer type
            obj_types = re.findall(obj_type_re, str(blob))
            if len(obj_types) > 0:
                real_obj_type = obj_types[0]
                for possible_type in obj_type:
                    possible_type_match = possible_type
                    if hasattr(possible_type, "wrapped_type"):
                        possible_type_match = possible_type.wrapped_type()
                    # get the str inside <class ...>, fqn in sympy is different
                    real_obj_type_str = str(possible_type_match).split("'")[1]
                    if real_obj_type.endswith("Wrapper"):
                        real_obj_type = real_obj_type[:-7]  # remove the last Wrapper

                    if real_obj_type == real_obj_type_str or real_obj_type.endswith(
                        real_obj_type_str
                    ):
                        # found it, lets overwrite obj_type and break
                        obj_type = possible_type
                        break

    if not isinstance(obj_type, type):
        traceback_and_raise(f"{deserialization_error}. {type(blob)}")

    _proto2object = getattr(obj_type, "_proto2object", None)
    if not callable(_proto2object):
        traceback_and_raise(deserialization_error)

    res = _proto2object(proto=blob)

    # if its a temporary_box upcast
    if getattr(res, "temporary_box", False) and hasattr(res, "upcast"):
        return res.upcast()

    return res


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
