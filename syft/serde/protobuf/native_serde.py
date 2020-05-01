"""
This file exists to provide a common place for all Protobuf
serialisation for native Python objects. If you're adding
something here that isn't for `None`, think twice and either
use an existing sub-class of Message or add a new one.
"""
import pydoc
import warnings
from collections import OrderedDict

from google.protobuf.empty_pb2 import Empty
from syft.workers.abstract import AbstractWorker

from syft_proto.execution.v1.type_wrapper_pb2 import InputTypeDescriptor as InputTypeDescriptorPB


def _bufferize_none(worker: AbstractWorker, obj: "type(None)") -> "Empty":
    """
    This function converts None into an empty Protobuf message.

    Args:
        obj (None): makes signature match other bufferize methods

    Returns:
        protobuf_obj: Empty Protobuf message
    """
    return Empty()


def _unbufferize_none(worker: AbstractWorker, obj: "Empty") -> "type(None)":
    """
    This function converts an empty Protobuf message back into None.

    Args:
        obj (Empty): Empty Protobuf message

    Returns:
        obj: None
    """
    return None


def _bufferize_type(worker: AbstractWorker, obj) -> InputTypeDescriptorPB:
    """
    This function gets the type object and returns the ClassType Protobuf message containing the string with the path
    of that that and the actual type..

    Args:
        obj_type (s.g builtins.str, builtins.int, torch.tensor): a type

    Returns:
        ClassTypePB: the Protobuf message type containg the path where to find the type + type.

    Examples:
          str_type_representation = _bufferize_type(worker, type("i'm a string"))
    """

    proto_type = InputTypeDescriptorPB()

    if isinstance(obj, type):
        module_path = obj.__module__
        full_path_type = module_path + "." + obj.__name__
        proto_type.type_name = full_path_type

    return proto_type


def _unbufferize_type(worker: AbstractWorker, class_type_msg: InputTypeDescriptorPB):
    """
    This function receives the ClassType Protobuf message containing the string with the path + type, decodes the string
    and locates the type in a module, returning the type object.

    Args:
        class_type_msg: message encoding the type.

    Returns:
        type: the type of an object (e.g: builtins.str, builtins.int).

    Warning: if pydoc can't locate the type in the current process, might mean that the file layout is different between
    sender and receiver.

    TODO:
        As syft-protobuf grows in type support, we should change the type serialization by using those types, enabling cross
        language typechecking/type validation.
    """
    result = pydoc.locate(class_type_msg.type_name)
    if result is None:
        warnings.warn(
            f"{class_type_msg.type_name} can't be located in the current process, the layout of the modules has been changed.",
            Warning,
        )
        return object
    return result


# Maps a type to its bufferizer and unbufferizer functions
MAP_NATIVE_PROTOBUF_TRANSLATORS = OrderedDict(
    {type(None): (_bufferize_none, _unbufferize_none), type: (_bufferize_type, _unbufferize_type)}
)
