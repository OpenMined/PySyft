# stdlib
from typing import Union

# third party
from google.protobuf.message import Message

# syft relative
from ....logger import debug
from ....logger import traceback_and_raise
from ....proto.util.data_message_pb2 import DataMessage
from ....util import get_fully_qualified_name
from ....util import validate_type
from .serializable import Serializable


def _serialize(
    obj: object,
    to_proto: bool = True,
    to_bytes: bool = False,
) -> Union[str, bytes, Message]:
    """Serialize the object according to the parameters.

    This method can be called directly on the syft module::

        import syft as sy
        serialized_obj = sy.serialize(obj=my_object_to_serialize)

    This is the primary serialization method, which processes the above
    flags in a particular order. In general, it is not expected that people
    will set multiple to_<type> flags to True at the same time. We don't
    currently have logic which prevents this, because this may affect
    runtime performance, but if several flags are True, then we will simply
    take return the type of latest supported flag from the following list:

        - proto
        - binary

    TODO: we could also add "dict" to this list but it's not clear if it would be used.

    :param to_proto: set this flag to TRUE if you want to return a protobuf object
    :type to_proto: bool
    :param to_bytes: set this flag to TRUE if you want to return a binary object
    :type to_bytes: bool
    :return: a serialized form of the object on which serialize() is called.
    :rtype: Union[str, bytes, Message]
    """

    is_serializable: Serializable
    if not isinstance(obj, Serializable):
        if hasattr(obj, "_sy_serializable_wrapper_type"):
            is_serializable = obj._sy_serializable_wrapper_type(value=obj)  # type: ignore
        else:
            traceback_and_raise(
                Exception(
                    f"Object {type(obj)} is not serializable and has no _sy_serializable_wrapper_type"
                )
            )
    else:
        is_serializable = obj

    if to_bytes:
        debug(f"Serializing {type(is_serializable)}")
        # indent=None means no white space or \n in the serialized version
        # this is compatible with json.dumps(x, indent=None)
        serialized_data = is_serializable._object2proto().SerializeToString()
        blob: Message = DataMessage(
            obj_type=get_fully_qualified_name(obj=is_serializable),
            content=serialized_data,
        )
        return validate_type(blob.SerializeToString(), bytes)
    elif to_proto:
        return validate_type(is_serializable._object2proto(), Message)
    else:
        traceback_and_raise(
            Exception(
                """You must specify at least one deserialization format using
                        one of the arguments of the serialize() method such as:
                        to_proto, to_bytes."""
            )
        )
