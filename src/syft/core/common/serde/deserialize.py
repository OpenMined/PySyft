from google.protobuf import json_format

from .serializable import Serializable
from ....decorators.syft_decorator_impl import syft_decorator
from google.protobuf.message import Message
from google.protobuf import json_format
from .serializable import serde_store

from ....proto.util.json_message_pb2 import JsonMessage


@syft_decorator(typechecking=True)
def _deserialize(
    blob: (str, dict, bytes, Message),
    from_proto: bool = True,
    from_json: bool = False,
    from_binary: bool = False,
    from_hex: bool = False,
    schema_type: type = None
) -> (Serializable, object):
    """We assume you're deserializing a protobuf object by default"""

    if from_hex:
        schematic = schema_type()
        schematic.ParseFromString(bytes.fromhex(blob))
        blob = schematic

    if from_binary:
        schematic = schema_type()
        schematic.ParseFromString(blob)
        blob = schematic

    if from_json:
        json_message = json_format.Parse(text=blob, message=JsonMessage())
        obj_type = serde_store.qual_name2type[json_message.obj_type]
        protobuf_type = obj_type.get_protobuf_schema()
        schema_data = json_message.content
        blob = json_format.Parse(text=schema_data, message=protobuf_type())

    if from_proto:
        proto_obj = blob

    try:
        # lets try to lookup the type we are deserializing
        obj_type = serde_store.schema2type[type(blob)]

    # uh-oh! Looks like the type doesn't exist. Let's throw an informative error.
    except KeyError:

        raise KeyError(
            """You tried to deserialize an unsupported type. This can be caused by
            several reasons. Either you are actively writing Syft code and forgot
            to create one, or you are trying to deserialize an object which was
            serialized using a different version of Syft and the object you tried
            to deserialize is not supported in this version."""
        )

    return obj_type._proto2object(proto=proto_obj)
