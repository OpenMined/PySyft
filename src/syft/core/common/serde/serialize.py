from .serializable import Serializable
from ....decorators.syft_decorator_impl import syft_decorator
from google.protobuf.message import Message
from typing import Union


# QUESTION: Is runtime type checking the best idea here?
@syft_decorator(typechecking=True)
def _serialize(
    obj: object,
    to_proto: bool = True,
    to_json: bool = False,
    to_binary: bool = False,
    to_hex: bool = False,
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
        - json
        - binary
        - hex

    TODO: we could also add "dict" to this list but it's not clear if it would be used.

    :param to_proto: set this flag to TRUE if you want to return a protobuf object
    :type to_proto: bool
    :param to_json: set this flag to TRUE if you want to return a json object
    :type to_json: bool
    :param to_binary: set this flag to TRUE if you want to return a binary object
    :type to_binary: bool
    :param to_hex: set this flag to TRUE if you want to return a hex string object
    :type to_hex: bool
    :return: a serialized form of the object on which serialize() is called.
    :rtype: Union[str, bytes, Message]
    """

    is_serializable: Serializable
    if not isinstance(obj, Serializable):
        if hasattr(obj, "serializable_wrapper_type"):
            is_serializable = obj.serializable_wrapper_type(  # type: ignore
                value=obj, as_wrapper=True
            )
        else:
            raise Exception(f"Object {type(obj)} has no serializable_wrapper_type")
    else:
        is_serializable = obj

    return is_serializable.serialize(
        to_proto=to_proto, to_json=to_json, to_binary=to_binary, to_hex=to_hex
    )
