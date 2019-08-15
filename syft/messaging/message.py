"""
This file exists as the Python encoding of all Message types that Syft sends over the network. It is
an important bottleneck in the system, impacting both security, performance, and cross-platform
compatability. As such, message types should strive to not be framework specific (i.e., Torch,
Tensorflow, etc.).

All Syft message types extend the Message class.
"""

import syft as sy

from syft.workers import AbstractWorker
from syft import codes


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

    def __init__(self, msg_type: int, contents=None):
        self.msg_type = msg_type
        if contents is not None:
            self.contents = contents

    def _simplify(self):
        return (self.msg_type, self.contents)

    @staticmethod
    def simplify(ptr: "Message") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            ptr (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(ptr)
        """

        return (ptr.msg_type, sy.serde._simplify(ptr.contents))

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "Message":
        """
        This function takes the simplified tuple version of this message and converts
        it into a message. The simplify() method runs the inverse of this method.

        This method shouldn't get called very often. It exists as a backup but in theory
        every message type should have its own detailer.

        Args:
            ptr (tuple): a tuple holding the unique attributes of the message
        Returns:
            ptr (Message): a Message.
        Examples:
            data = simplify(ptr)
        """

        # TODO: attempt to use the tensor_tuple[0] to return the correct type instead of Message
        # TODO: as an alternative, this detailer could raise NotImplementedException

        return Message(tensor_tuple[0], sy.serde._detail(worker, tensor_tuple[1]))

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({codes.code2MSGTYPE[self.msg_type]} {self.contents})"

    def __repr__(self):
        """Return a human readable version of this message"""
        return self.__str__()


class CommandMessage(Message):
    def __init__(self, message, return_ids):
        super().__init__(codes.MSGTYPE.CMD)

        self.message = message
        self.return_ids = return_ids

    @property
    def contents(self):  # need this just because some methods assume the tuple form (legacy)
        return (self.message, self.return_ids)

    @staticmethod
    def simplify(ptr: "CommandMessage") -> tuple:
        """
        This function takes the attributes of a CommandMessage and saves them in a tuple
        Args:
            ptr (CommandMessage): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(ptr)
        """
        # NOTE: we can skip calling _simplify on return_ids because they should already be
        # a list of simple types.
        return (ptr.msg_type, (sy.serde._simplify(ptr.message), ptr.return_ids))

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "Message":
        return CommandMessage(sy.serde._detail(worker, tensor_tuple[1][0]), tensor_tuple[1][1])


class ObjectMessage(Message):
    def __init__(self, contents):
        super().__init__(codes.MSGTYPE.OBJ, contents)

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "Message":
        return ObjectMessage(sy.serde._detail(worker, tensor_tuple[1]))


class ObjectRequestMessage(Message):
    # TODO: add more efficient detalier and simplifier custom for this type

    def __init__(self, contents):
        super().__init__(codes.MSGTYPE.OBJ_REQ, contents)

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "Message":
        return ObjectRequestMessage(sy.serde._detail(worker, tensor_tuple[1]))


class IsNoneMessage(Message):
    # TODO: add more efficient detalier and simplifier custom for this type

    def __init__(self, contents):
        super().__init__(codes.MSGTYPE.IS_NONE, contents)

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "Message":
        return IsNoneMessage(sy.serde._detail(worker, tensor_tuple[1]))


class GetShapeMessage(Message):
    # TODO: add more efficient detalier and simplifier custom for this type

    def __init__(self, contents):
        super().__init__(codes.MSGTYPE.GET_SHAPE, contents)

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "Message":
        return GetShapeMessage(sy.serde._detail(worker, tensor_tuple[1]))


class ForceObjectDeleteMessage(Message):
    def __init__(self, contents):
        super().__init__(codes.MSGTYPE.FORCE_OBJ_DEL, contents)

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "Message":
        return ForceObjectDeleteMessage(sy.serde._detail(worker, tensor_tuple[1]))


class SearchMessage(Message):
    # TODO: add more efficient detalier and simplifier custom for this type

    def __init__(self, contents):
        super().__init__(codes.MSGTYPE.SEARCH, contents)

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "Message":
        return SearchMessage(sy.serde._detail(worker, tensor_tuple[1]))
