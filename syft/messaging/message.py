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

        # saves us a write op but costs us a check op to only sometimes
        # set ._contents
        if contents is not None:
            self._contents = contents

    @property
    def contents(self):
        """Return a tuple with the contents of the command (backwards compatability)

        Some of our codebase still assumes that all message types have a .contents attribute. However,
        the contents attribute is very opaque in that it doesn't put any constraints on what the contents
        might be. Some message types can be more efficient by storing their contents more explicitly (see
        CommandMessage). They can override this property to return a tuple view on their other properties.
        """
        if hasattr(self, "_contents"):
            return self._contents
        else:
            return None

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
            ptr (Message): a CommandMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """

        # TODO: attempt to use the tensor_tuple[0] to return the correct type instead of Message
        # TODO: as an alternative, this detailer could raise NotImplementedException

        return Message(msg_tuple[0], sy.serde._detail(worker, msg_tuple[1]))

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({codes.code2MSGTYPE[self.msg_type]} {self.contents})"

    def __repr__(self):
        """Return a human readable version of this message"""
        return self.__str__()


class CommandMessage(Message):
    """All syft commands use this message type

    In Syft, a command is when one worker wishes to tell another worker to do something with
    objects contained in the worker._objects registry (or whatever the official object store is
    backed with in the case that it's been overridden). Semantically, one could view all Messages
    as a kind of command, but when we say command this is what we mean. For example, telling a
    worker to take two tensors and add them together is a command. However, sending an object
    from one worker to another is not a command (and would instead use the ObjectMessage type)."""

    def __init__(self, message, return_ids):
        """Initialize a command message

        Args:
            message (Tuple): this is typically the args and kwargs of a method call on the client, but it
                can be any information necessary to execute the command properly.
            return_ids (Tuple): primarily for our async infrastructure (Plan, Protocol, etc.), the id of
                command results are set by the client. This allows the client to be able to predict where
                the results will be ahead of time. Importantly, this allows the client to pre-initalize the
                pointers to the future data, regardless of whether the command has yet executed. It also
                reduces the size of the response from the command (which is very often empty).

        """

        # call the parent constructor - setting the type integer correctly
        super().__init__(codes.MSGTYPE.CMD)

        self.message = message
        self.return_ids = return_ids

    @property
    def contents(self):
        """Return a tuple with the contents of the command (backwards compatability)

        Some of our codebase still assumes that all message types have a .contents attribute. However,
        the contents attribute is very opaque in that it doesn't put any constraints on what the contents
        might be. Since we know this message is a command, we instead choose to store contents in two pieces,
        self.message and self.return_ids, which allows for more efficient simplification (we don't have to
        simplify return_ids because they are always a list of integers, meaning they're already simplified)."""

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
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "CommandMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into a CommandMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (CommandMessage): a CommandMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return CommandMessage(sy.serde._detail(worker, msg_tuple[1][0]), msg_tuple[1][1])


class ObjectMessage(Message):
    """Send an object to another worker using this message type.

    When a worker has an object in its local object repository (such as a tensor) and it wants
    to send that object to another worker (and delete its local copy), it uses this message type
    to do so.
    """

    def __init__(self, contents):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
        super().__init__(codes.MSGTYPE.OBJ, contents)

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "ObjectMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into an ObjectMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (ObjectMessage): a ObjectMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return ObjectMessage(sy.serde._detail(worker, tensor_tuple[1]))


class ObjectRequestMessage(Message):
    """Request another worker to send one of its objects

    If ObjectMessage pushes an object to another worker, this Message type pulls an
    object from another worker. It also assumes that the other worker will delete it's
    local copy of the object after sending it to you."""

    # TODO: add more efficient detalier and simplifier custom for this type

    def __init__(self, contents):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
        super().__init__(codes.MSGTYPE.OBJ_REQ, contents)

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "ObjectRequestMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into an ObjectRequestMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (ObjectRequestMessage): a ObjectRequestMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return ObjectRequestMessage(sy.serde._detail(worker, tensor_tuple[1]))


class IsNoneMessage(Message):
    """Check if a worker does not have an object with a specific id.

    Occasionally we need to verify whether or not a remote worker has a specific
    object. To do so, we send an IsNoneMessage, which returns True if the object
    (such as a tensor) does NOT exist."""

    # TODO: add more efficient detalier and simplifier custom for this type

    def __init__(self, contents):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
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
