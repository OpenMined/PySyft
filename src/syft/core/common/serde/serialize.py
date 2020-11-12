# stdlib
from typing import Union

# third party
from google.protobuf.message import Message

# syft relative
from ....decorators.syft_decorator_impl import syft_decorator
from .serializable import Serializable


@syft_decorator(typechecking=True)
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
        if hasattr(obj, "serializable_wrapper_type"):
            is_serializable = obj.serializable_wrapper_type(value=obj)  # type: ignore
        else:
            raise Exception(f"Object {type(obj)} has no serializable_wrapper_type")
    else:
        is_serializable = obj

    return is_serializable.serialize(to_proto=to_proto, to_bytes=to_bytes)
