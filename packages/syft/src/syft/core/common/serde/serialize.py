# relative
from ....util import validate_type
from .deserialize import PROTOBUF_START_MAGIC_HEADER
from .recursive import recursive_serde
from .recursive import rs_object2proto
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

    :param to_proto: set this flag to TRUE if you want to return a protobuf object
    :type to_proto: bool
    :param to_bytes: set this flag to TRUE if you want to return a binary object
    :type to_bytes: bool
    :return: a serialized form of the object on which serialize() is called.
    :rtype: Union[str, bytes, Message]
    """

    # capnp_bytes=True
    if hasattr(obj, "_object2bytes"):
        # capnp proto
        return validate_type(obj._object2bytes(), bytes)

    proto = rs_object2proto(obj)

    if to_bytes:
        return proto.SerializeToString()
    else:
        return proto
