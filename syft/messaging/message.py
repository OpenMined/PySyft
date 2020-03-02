"""
This file exists as the Python encoding of all Message types that Syft sends over the network. It is
an important bottleneck in the system, impacting both security, performance, and cross-platform
compatability. As such, message types should strive to not be framework specific (i.e., Torch,
Tensorflow, etc.).

All Syft message types extend the Message class.
"""

import syft as sy
from syft.workers.abstract import AbstractWorker

from syft.execution.operation import Operation
from syft.frameworks.torch.tensors.interpreters.placeholder import PlaceHolder

from syft_proto.execution.v1.operation_pb2 import Operation as OperationPB
from syft_proto.messaging.v1.message_pb2 import ObjectMessage as ObjectMessagePB
from syft_proto.messaging.v1.message_pb2 import OperationMessage as OperationMessagePB
from syft_proto.types.syft.v1.arg_pb2 import Arg as ArgPB


class Message:
    """All syft message types extend this class

    All messages in the pysyft protocol extend this class. This abstraction
    requires that every message has an integer type, which is important because
    this integer is what determines how the message is handled when a BaseWorker
    receives it.

    Additionally, this type supports a default simplifier and detailer, which are
    important parts of PySyft's serialization and deserialization functionality.
    You can read more abouty detailers and simplifiers in syft/serde/serde.py.
    """

    def __init__(self, contents=None):

        # saves us a write op but costs us a check op to only sometimes
        # set ._contents
        if contents is not None:
            self._contents = contents

    @property
    def contents(self):
        """Return a tuple with the contents of the message (backwards compatability)

        Some of our codebase still assumes that all message types have a .contents attribute. However,
        the contents attribute is very opaque in that it doesn't put any constraints on what the contents
        might be. Some message types can be more efficient by storing their contents more explicitly (see
        Operation). They can override this property to return a tuple view on their other properties.
        """
        if hasattr(self, "_contents"):
            return self._contents
        else:
            return None

    def _simplify(self):
        return (self.contents,)

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: "Message") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            ptr (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(ptr)
        """

        return (sy.serde.msgpack.serde._simplify(worker, ptr.contents),)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "Message":
        """
        This function takes the simplified tuple version of this message and converts
        it into a message. The simplify() method runs the inverse of this method.

        This method shouldn't get called very often. It exists as a backup but in theory
        every message type should have its own detailer.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (Message): a Operation.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """

        # TODO: attempt to use the msg_tuple[0] to return the correct type instead of Message
        # https://github.com/OpenMined/PySyft/issues/2514
        # TODO: as an alternative, this detailer could raise NotImplementedException
        # https://github.com/OpenMined/PySyft/issues/2514

        return Message(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({type(self).__name__} {self.contents})"

    def __repr__(self):
        """Return a human readable version of this message"""
        return self.__str__()
