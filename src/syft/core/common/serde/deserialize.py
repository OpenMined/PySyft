# stdlib
from typing import Any
from typing import Union

# third party
from google.protobuf.message import Message

# syft relative
from ....logger import traceback_and_raise
from ....proto.util.data_message_pb2 import DataMessage
from ....util import index_syft_by_module_name


def _deserialize(
    blob: Union[str, dict, bytes, Message],
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

    try:
        # lets try to lookup the type we are deserializing
        obj_type = getattr(type(blob), "schema2type", None)

    # uh-oh! Looks like the type doesn't exist. Let's throw an informative error.
    except AttributeError:
        traceback_and_raise(deserialization_error)

    _proto2object = getattr(obj_type, "_proto2object", None)

    if not callable(_proto2object):
        traceback_and_raise(deserialization_error)

    obj = _proto2object(proto=blob)
    # the real object is nested in CTypeWrapper's
    if type(_proto2object).__name__.endswith("CTypeWrapper"):
        obj = _proto2object.obj
    return obj
