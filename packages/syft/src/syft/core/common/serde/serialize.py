# third party
from google.protobuf.message import Message

# relative
from ....logger import traceback_and_raise
from ....proto.util.data_message_pb2 import DataMessage
from ....util import get_fully_qualified_name
from ....util import validate_type
from .deserialize import PROTOBUF_START_MAGIC_HEADER
from .types import Deserializeable


def create_protobuf_magic_header() -> str:
    return f"{PROTOBUF_START_MAGIC_HEADER}"


def _serialize(
    obj: object,
    to_proto: bool = True,
    to_bytes: bool = False,
) -> Deserializeable:
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

    # relative
    from ....lib.python.primitive_factory import isprimitive

    # we have an unboxed primitive type so we need to mirror that on deserialize
    if isprimitive(obj):
        # relative
        from ....lib.python.primitive_factory import PrimitiveFactory

        obj = PrimitiveFactory.generate_primitive(value=obj, temporary_box=True)
        if hasattr(obj, "temporary_box"):
            # TODO: can remove this once all of PrimitiveFactory.generate_primitive
            # supports temporary_box and is tested
            obj.temporary_box = True  # type: ignore

    if hasattr(obj, "_sy_serializable_wrapper_type"):
        is_serializable = obj._sy_serializable_wrapper_type(value=obj)  # type: ignore
    else:
        is_serializable = obj

    # traceback_and_raise(
    #     Exception(
    #         f"Object {type(obj)} is not serializable and has no _sy_serializable_wrapper_type"
    #     )
    # )

    # capnp_bytes=True
    if hasattr(is_serializable, "_object2bytes"):
        # capnp proto
        return validate_type(is_serializable._object2bytes(), bytes)

    if to_bytes:
        # debug(f"Serializing {type(is_serializable)}")
        # indent=None means no white space or \n in the serialized version
        # this is compatible with json.dumps(x, indent=None)
        serialized_data = is_serializable._object2proto().SerializeToString()
        obj_type = get_fully_qualified_name(obj=is_serializable)
        blob: Message = DataMessage(
            magic_header=create_protobuf_magic_header(),
            obj_type=obj_type,
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
