"""
This file exists to provide a common place for all Protobuf
serialisation for native Python objects. If you're adding
something here that isn't for `None`, think twice and either
use an existing sub-class of Message or add a new one.
"""

from collections import OrderedDict
from google.protobuf.empty_pb2 import Empty
from syft.workers.abstract import AbstractWorker


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


# Maps a type to its bufferizer and unbufferizer functions
MAP_NATIVE_PROTOBUF_TRANSLATORS = OrderedDict({type(None): (_bufferize_none, _unbufferize_none)})
