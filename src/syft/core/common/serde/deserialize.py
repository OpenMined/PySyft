from .serializable import Serializable
from ....decorators.syft_decorator_impl import syft_decorator
from google.protobuf.message import Message
from google.protobuf import json_format
from .util import fully_qualified_name2type  # noqa: F401
import json


@syft_decorator(typechecking=True)
def _deserialize(
    blob: (str, dict, bytes, Message),
    from_proto: bool = True,
    from_json: bool = False,
    from_binary: bool = False,
    from_hex: bool = False,
) -> (Serializable, object):
    """We assume you're deserializing a protobuf object by default"""

    global fully_qualified_name2type

    if from_hex:
        from_binary = True
        blob = bytes.fromhex(blob)

    if from_binary:
        from_json = True
        blob = str(blob, "utf-8")

    if from_json:
        from_proto = False
        blob = json.loads(s=blob)
        obj_type_str = blob["objType"]

    if from_proto:
        obj_type_str = blob.obj_type
        proto_obj = blob

    try:
        # lets try to lookup the type we are deserializing
        obj_type = fully_qualified_name2type[obj_type_str]

    # uh-oh! Looks like the type doesn't exist. Let's throw an informative error.
    except KeyError:

        raise KeyError(
            """You tried to deserialize an unsupported type. This can be caused by
            several reasons. Either you are actively writing Syft code and forgot
            to create one, or you are trying to deserialize an object which was
            serialized using a different version of Syft and the object you tried
            to deserialize is not supported in this version."""
        )

    protobuf_type = obj_type.protobuf_type

    if not from_proto:
        proto_obj = json_format.ParseDict(js_dict=blob, message=protobuf_type())

    return obj_type._proto2object(proto_obj)
