# stdlib
from typing import Union
from typing import cast

# third party
from google.protobuf import json_format
from google.protobuf.message import Message

# syft relative
from ....decorators.syft_decorator_impl import syft_decorator
from ....proto.util.json_message_pb2 import JsonMessage
from ....util import index_syft_by_module_name
from .serializable import Serializable


@syft_decorator(typechecking=True)
def _deserialize(
    blob: Union[str, dict, bytes, Message],
    from_proto: bool = True,
    from_json: bool = False,
    from_binary: bool = False,
    from_hex: bool = False,
) -> Union[Serializable, object]:
    """We assume you're deserializing a protobuf object by default

    This function deserializes from an encoding to a Python object. There are a few ways of
    using this function:
        1. An Message object is passed, this will transform a protobuf message into its
        associated class. the from_proto has to be set (it is by default).
        2. Bytes are passed. This requires the from_bytes flag set the schema_type specified.
        We cannot (and we should not) be able to get the schema_type from the binary
        representation.
        3. A hex string is passed. This will be transformed to binary and afterwards the
        second step is applied. The from_hex flag should be set and the schema_type should be
        specified.
        4. A json object is passed. The from_json flag must be set and the schema_type should
        be specified.

    Note: The only format that does not require the schema_type is when we are passing
    Messages directly.

    Raises: ValueError if you are not setting one from_<protocol> flag.
            ValueError if you are deserializing a data type that requires a schema type and not
            providing one.
            TypeError if you are are trying to deserialize an unsupported type.

    :param blob: this parameter is the data to be deserialized from various formats.
    :type blob: Union[str, dict, bytes, Messages]
    :param from_proto: set this flag to True if you want to deserialize a protobuf message.
    :type from_json: bool
    :param from_binary: set this flag to True if you want to deserialize a binary object.
    :type from_binary: bool
    :param from_hex: set this flag to True if you want to deserialize a hex string object
    :type from_hex: bool
    :return: a deserialized form of the object on which _deserialize() is called.
    :rtype: Serializable
    """

    import syft as sy


    if from_hex:

        blob = str(bytes.fromhex(cast(str, blob)), "utf-8")

    elif from_binary:

        blob = str(blob, "utf-8")  # type: ignore

    sy.logger.debug("Deserializing blob")
    sy.logger.debug(blob)
    if from_json or from_binary or from_hex:
        import syft as sy
        sy.logger.debug(blob)
        json_message = json_format.Parse(text=blob, message=JsonMessage())

        obj_type = index_syft_by_module_name(fully_qualified_name=json_message.obj_type)
        protobuf_type = obj_type.get_protobuf_schema()
        schema_data = json_message.content
        blob = json_format.Parse(text=schema_data, message=protobuf_type())

    elif not from_proto:
        raise ValueError("Please pick the format of the data on the deserialization")

    try:
        # lets try to lookup the type we are deserializing
        obj_type = type(blob).schema2type  # type: ignore

    # uh-oh! Looks like the type doesn't exist. Let's throw an informative error.
    except AttributeError:
        raise TypeError(
            "You tried to deserialize an unsupported type. This can be caused by "
            "several reasons. Either you are actively writing Syft code and forgot "
            "to create one, or you are trying to deserialize an object which was "
            "serialized using a different version of Syft and the object you tried "
            "to deserialize is not supported in this version."
        )

    return obj_type._proto2object(proto=blob)
