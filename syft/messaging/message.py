"""
This file exists as the Python encoding of all Message types that Syft sends over the network. It is
an important bottleneck in the system, impacting both security, performance, and cross-platform
compatability. As such, message types should strive to not be framework specific (i.e., Torch,
Tensorflow, etc.).

All Syft message types extend the Message class.
"""

import syft as sy
from syft.workers.abstract import AbstractWorker

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


class OperationMessage(Message):
    """All syft operations use this message type

    In Syft, an operation is when one worker wishes to tell another worker to do something with
    objects contained in the worker._objects registry (or whatever the official object store is
    backed with in the case that it's been overridden). Semantically, one could view all Messages
    as a kind of operation, but when we say operation this is what we mean. For example, telling a
    worker to take two tensors and add them together is an operation. However, sending an object
    from one worker to another is not an operation (and would instead use the ObjectMessage type)."""

    def __init__(self, cmd_name, cmd_owner, cmd_args, cmd_kwargs, return_ids):
        """Initialize an operation message

        Args:
            message (Tuple): this is typically the args and kwargs of a method call on the client, but it
                can be any information necessary to execute the operation properly.
            return_ids (Tuple): primarily for our async infrastructure (Plan, Protocol, etc.), the id of
                operation results are set by the client. This allows the client to be able to predict where
                the results will be ahead of time. Importantly, this allows the client to pre-initalize the
                pointers to the future data, regardless of whether the operation has yet executed. It also
                reduces the size of the response from the operation (which is very often empty).

        """

        # call the parent constructor - setting the type integer correctly
        super().__init__()

        self.cmd_name = cmd_name
        self.cmd_owner = cmd_owner
        self.cmd_args = cmd_args
        self.cmd_kwargs = cmd_kwargs
        self.return_ids = return_ids

    @property
    def contents(self):
        """Return a tuple with the contents of the operation (backwards compatability)

        Some of our codebase still assumes that all message types have a .contents attribute. However,
        the contents attribute is very opaque in that it doesn't put any constraints on what the contents
        might be. Since we know this message is a operation, we instead choose to store contents in two pieces,
        self.message and self.return_ids, which allows for more efficient simplification (we don't have to
        simplify return_ids because they are always a list of integers, meaning they're already simplified)."""

        message = (self.cmd_name, self.cmd_owner, self.cmd_args, self.cmd_kwargs)

        return (message, self.return_ids)

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: "OperationMessage") -> tuple:
        """
        This function takes the attributes of a OperationMessage and saves them in a tuple
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            ptr (OperationMessage): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(ptr)
        """
        # NOTE: we can skip calling _simplify on return_ids because they should already be
        # a list of simple types.
        message = (ptr.cmd_name, ptr.cmd_owner, ptr.cmd_args, ptr.cmd_kwargs)

        return (
            sy.serde.msgpack.serde._simplify(worker, message),
            sy.serde.msgpack.serde._simplify(worker, ptr.return_ids),
        )

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "OperationMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into a Operation. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (OperationMessage): an OperationMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        message = msg_tuple[0]
        return_ids = msg_tuple[1]

        detailed_msg = sy.serde.msgpack.serde._detail(worker, message)
        detailed_ids = sy.serde.msgpack.serde._detail(worker, return_ids)

        cmd_name = detailed_msg[0]
        cmd_owner = detailed_msg[1]
        cmd_args = detailed_msg[2]
        cmd_kwargs = detailed_msg[3]

        return OperationMessage(cmd_name, cmd_owner, cmd_args, cmd_kwargs, detailed_ids)

    @staticmethod
    def bufferize(worker: AbstractWorker, operation: "OperationMessage") -> "OperationMessagePB":
        """
        This function takes the attributes of a OperationMessage and saves them in Protobuf
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            ptr (OperationMessage): an OperationMessage
        Returns:
            protobuf_obj: a Protobuf message holding the unique attributes of the message
        Examples:
            data = bufferize(message)
        """
        protobuf_op_msg = OperationMessagePB()
        protobuf_op = OperationPB()
        protobuf_op.command = operation.cmd_name

        if type(operation.cmd_owner) == sy.generic.pointers.pointer_tensor.PointerTensor:
            protobuf_owner = protobuf_op.owner_pointer
        elif (
            type(operation.cmd_owner)
            == sy.frameworks.torch.tensors.interpreters.placeholder.PlaceHolder
        ):
            protobuf_owner = protobuf_op.owner_placeholder
        else:
            protobuf_owner = protobuf_op.owner_tensor

        if operation.cmd_owner is not None:
            protobuf_owner.CopyFrom(sy.serde.protobuf.serde._bufferize(worker, operation.cmd_owner))

        if operation.cmd_args:
            protobuf_op.args.extend(OperationMessage._bufferize_args(worker, operation.cmd_args))

        if operation.cmd_kwargs:
            for key, value in operation.cmd_kwargs.items():
                protobuf_op.kwargs.get_or_create(key).CopyFrom(
                    OperationMessage._bufferize_arg(worker, value)
                )

        if operation.return_ids is not None:
            if type(operation.return_ids) == PlaceHolder:
                return_ids = list((operation.return_ids,))
            else:
                return_ids = operation.return_ids

            for return_id in return_ids:
                if type(return_id) == PlaceHolder:
                    protobuf_op.return_placeholders.extend(
                        [sy.serde.protobuf.serde._bufferize(worker, return_id)]
                    )
                else:
                    sy.serde.protobuf.proto.set_protobuf_id(protobuf_op.return_ids.add(), return_id)

        protobuf_op_msg.operation.CopyFrom(protobuf_op)
        return protobuf_op_msg

    @staticmethod
    def unbufferize(
        worker: AbstractWorker, protobuf_obj: "OperationMessagePB"
    ) -> "OperationMessage":
        """
        This function takes the Protobuf version of this message and converts
        it into an OperationMessage. The bufferize() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            protobuf_obj (OperationMessagePB): the Protobuf message

        Returns:
            obj (OperationMessage): an OperationMessage

        Examples:
            message = unbufferize(sy.local_worker, protobuf_msg)
        """

        command = protobuf_obj.operation.command
        protobuf_owner = protobuf_obj.operation.WhichOneof("owner")
        if protobuf_owner:
            owner = sy.serde.protobuf.serde._unbufferize(
                worker, getattr(protobuf_obj.operation, protobuf_obj.operation.WhichOneof("owner"))
            )
        else:
            owner = None
        args = OperationMessage._unbufferize_args(worker, protobuf_obj.operation.args)

        kwargs = {}
        for key in protobuf_obj.operation.kwargs:
            kwargs[key] = OperationMessage._unbufferize_arg(
                worker, protobuf_obj.operation.kwargs[key]
            )

        return_ids = [
            sy.serde.protobuf.proto.get_protobuf_id(pb_id)
            for pb_id in protobuf_obj.operation.return_ids
        ]

        return_placeholders = [
            sy.serde.protobuf.serde._unbufferize(worker, placeholder)
            for placeholder in protobuf_obj.operation.return_placeholders
        ]

        if return_placeholders:
            if len(return_placeholders) == 1:
                operation_msg = OperationMessage(
                    command, owner, tuple(args), kwargs, return_placeholders[0]
                )
            else:
                operation_msg = OperationMessage(
                    command, owner, tuple(args), kwargs, return_placeholders
                )
        else:
            operation_msg = OperationMessage(command, owner, tuple(args), kwargs, tuple(return_ids))

        return operation_msg

    @staticmethod
    def _bufferize_args(worker: AbstractWorker, args: list) -> list:
        protobuf_args = []
        for arg in args:
            protobuf_args.append(OperationMessage._bufferize_arg(worker, arg))
        return protobuf_args

    @staticmethod
    def _bufferize_arg(worker: AbstractWorker, arg: object) -> ArgPB:
        protobuf_arg = ArgPB()
        try:
            setattr(protobuf_arg, "arg_" + type(arg).__name__.lower(), arg)
        except:
            getattr(protobuf_arg, "arg_" + type(arg).__name__.lower()).CopyFrom(
                sy.serde.protobuf.serde._bufferize(worker, arg)
            )
        return protobuf_arg

    @staticmethod
    def _unbufferize_args(worker: AbstractWorker, protobuf_args: list) -> list:
        args = []
        for protobuf_arg in protobuf_args:
            args.append(OperationMessage._unbufferize_arg(worker, protobuf_arg))
        return args

    @staticmethod
    def _unbufferize_arg(worker: AbstractWorker, protobuf_arg: ArgPB) -> object:
        protobuf_arg_field = getattr(protobuf_arg, protobuf_arg.WhichOneof("arg"))
        try:
            arg = sy.serde.protobuf.serde._unbufferize(worker, protobuf_arg_field)
        except:
            arg = protobuf_arg_field
        return arg


class ObjectMessage(Message):
    """Send an object to another worker using this message type.

    When a worker has an object in its local object repository (such as a tensor) and it wants
    to send that object to another worker (and delete its local copy), it uses this message type
    to do so.
    """

    def __init__(self, contents):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
        super().__init__(contents)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "ObjectMessage":
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
        return ObjectMessage(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))

    @staticmethod
    def bufferize(worker: AbstractWorker, message: "ObjectMessage") -> "ObjectMessagePB":
        """
        This function takes the attributes of an Object Message and saves them in a protobuf object
        Args:
            message (ObjectMessage): an ObjectMessage
        Returns:
            protobuf: a protobuf object holding the unique attributes of the object message
        Examples:
            data = bufferize(object_message)
        """

        protobuf_obj_msg = ObjectMessagePB()
        bufferized_contents = sy.serde.protobuf.serde._bufferize(worker, message.contents)
        protobuf_obj_msg.tensor.CopyFrom(bufferized_contents)
        return protobuf_obj_msg

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_obj: "ObjectMessagePB") -> "ObjectMessage":
        protobuf_contents = protobuf_obj.tensor
        contents = sy.serde.protobuf.serde._unbufferize(worker, protobuf_contents)
        object_msg = ObjectMessage(contents)

        return object_msg


class ObjectRequestMessage(Message):
    """Request another worker to send one of its objects

    If ObjectMessage pushes an object to another worker, this Message type pulls an
    object from another worker. It also assumes that the other worker will delete it's
    local copy of the object after sending it to you."""

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, contents):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
        super().__init__(contents)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "ObjectRequestMessage":
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
        return ObjectRequestMessage(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))


class IsNoneMessage(Message):
    """Check if a worker does not have an object with a specific id.

    Occasionally we need to verify whether or not a remote worker has a specific
    object. To do so, we send an IsNoneMessage, which returns True if the object
    (such as a tensor) does NOT exist."""

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, contents):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
        super().__init__(contents)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "IsNoneMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into an IsNoneMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (IsNoneMessage): a IsNoneMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return IsNoneMessage(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))


class GetShapeMessage(Message):
    """Get the shape property of a tensor in PyTorch

    We needed to have a special message type for this because .shape had some
    constraints in the older version of PyTorch."""

    # TODO: remove this message type and use ObjectRequestMessage instead.
    # note that the above to do is likely waiting for custom tensor type support in PyTorch
    # https://github.com/OpenMined/PySyft/issues/2513

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, contents):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
        super().__init__(contents)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "GetShapeMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into an GetShapeMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (GetShapeMessage): a GetShapeMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return GetShapeMessage(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))


class ForceObjectDeleteMessage(Message):
    """Garbage collect a remote object

    This is the dominant message for garbage collection of remote objects. When
    a pointer is deleted, this message is triggered by default to tell the object
    being pointed to to also delete itself.
    """

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, contents):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
        super().__init__(contents)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "ForceObjectDeleteMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into an ForceObjectDeleteMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (ForceObjectDeleteMessage): a ForceObjectDeleteMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return ForceObjectDeleteMessage(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))


class SearchMessage(Message):
    """A client queries for a subset of the tensors on a remote worker using this type

    For some workers like SocketWorker we split a worker into a client and a server. For
    this configuration, a client can request to search for a subset of tensors on the server
    using this message type (this could also be called a "QueryMessage").
    """

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, contents):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
        super().__init__(contents)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "SearchMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into an SearchMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (SearchMessage): a SearchMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return SearchMessage(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))


class PlanCommandMessage(Message):
    """Message used to execute a command related to plans."""

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, command_name: str, message: tuple):
        """Initialize a PlanCommandMessage.

        Args:
            command_name (str): name used to identify the command.
            message (Tuple): this is typically the args and kwargs of a method call on the client, but it
                can be any information necessary to execute the command properly.
        """

        # call the parent constructor - setting the type integer correctly
        super().__init__()

        self.command_name = command_name
        self.message = message

    @property
    def contents(self):
        """Returns a tuple with the contents of the operation (backwards compatability)."""
        return (self.command_name, self.message)

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: "PlanCommandMessage") -> tuple:
        """
        This function takes the attributes of a PlanCommandMessage and saves them in a tuple

        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            ptr (PlanCommandMessage): a Message

        Returns:
            tuple: a tuple holding the unique attributes of the message
        """
        return (
            sy.serde.msgpack.serde._simplify(worker, ptr.command_name),
            sy.serde.msgpack.serde._simplify(worker, ptr.message),
        )

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "PlanCommandMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into a PlanCommandMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (PlanCommandMessage): a PlanCommandMessage.
        """
        command_name, message = msg_tuple
        return PlanCommandMessage(
            sy.serde.msgpack.serde._detail(worker, command_name),
            sy.serde.msgpack.serde._detail(worker, message),
        )


class ExecuteWorkerFunctionMessage(Message):
    """Message used to execute a function of the remote worker."""

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, command_name: str, message: tuple):
        """Initialize a ExecuteWorkerFunctionMessage.

        Args:
            command_name (str): name used to identify the command.
            message (Tuple): this is typically the args and kwargs of a method call on the client, but it
                can be any information necessary to execute the command properly.
        """

        # call the parent constructor - setting the type integer correctly
        super().__init__()

        self.command_name = command_name
        self.message = message

    @property
    def contents(self):
        """Returns a tuple with the contents of the operation (backwards compatability)."""
        return (self.command_name, self.message)

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: "ExecuteWorkerFunctionMessage") -> tuple:
        """
        This function takes the attributes of a ExecuteWorkerFunctionMessage and saves them in a tuple

        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            ptr (ExecuteWorkerFunctionMessage): a Message

        Returns:
            tuple: a tuple holding the unique attributes of the message
        """
        return (
            sy.serde.msgpack.serde._simplify(worker, ptr.command_name),
            sy.serde.msgpack.serde._simplify(worker, ptr.message),
        )

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "ExecuteWorkerFunctionMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into a ExecuteWorkerFunctionMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (ExecuteWorkerFunctionMessage): a ExecuteWorkerFunctionMessage.
        """
        command_name, message = msg_tuple
        return ExecuteWorkerFunctionMessage(
            sy.serde.msgpack.serde._detail(worker, command_name),
            sy.serde.msgpack.serde._detail(worker, message),
        )
