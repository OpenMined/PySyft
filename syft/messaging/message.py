"""
This file exists as the Python encoding of all Message types that Syft sends over the network. It is
an important bottleneck in the system, impacting both security, performance, and cross-platform
compatability. As such, message types should strive to not be framework specific (i.e., Torch,
Tensorflow, etc.).

All Syft message types extend the Message class.
"""

from abc import ABC
from abc import abstractmethod

import syft as sy
from syft.workers.abstract import AbstractWorker
from syft.generic.abstract.syft_serializable import SyftSerializable

from syft.execution.action import Action
from syft.execution.computation import ComputationAction
from syft.execution.communication import CommunicationAction

from syft_proto.messaging.v1.message_pb2 import ObjectMessage as ObjectMessagePB
from syft_proto.messaging.v1.message_pb2 import TensorCommandMessage as CommandMessagePB
from syft_proto.messaging.v1.message_pb2 import (
    ForceObjectDeleteMessage as ForceObjectDeleteMessagePB,
)
from syft_proto.messaging.v1.message_pb2 import GetShapeMessage as GetShapeMessagePB
from syft_proto.messaging.v1.message_pb2 import IsNoneMessage as IsNoneMessagePB

# TODO: uncomment this when solving the WorkerCommandMessage issue.
# from syft_proto.messaging.v1.message_pb2 import WorkerCommandMessage as WorkerCommandMessagePB
from syft_proto.messaging.v1.message_pb2 import SearchMessage as SearchMessagePB
from syft_proto.messaging.v1.message_pb2 import ObjectRequestMessage as ObjectRequestMessagePB
from syft_proto.messaging.v1.message_pb2 import PlanCommandMessage as PlanCommandMessagePB
from syft_proto.types.syft.v1.id_pb2 import Id as IdPB


class Message(ABC, SyftSerializable):
    """All syft message types extend this class

    All messages in the pysyft protocol extend this class. This abstraction
    requires that every message has an integer type, which is important because
    this integer is what determines how the message is handled when a BaseWorker
    receives it.

    Additionally, this type supports a default simplifier and detailer, which are
    important parts of PySyft's serialization and deserialization functionality.
    You can read more abouty detailers and simplifiers in syft/serde/serde.py.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        """Return a human readable version of this message"""
        pass

    # Intentionally not abstract
    def __repr__(self):
        """Return a human readable version of this message"""
        return self.__str__()


class TensorCommandMessage(Message):
    """All syft actions use this message type

    In Syft, an action is when one worker wishes to tell another worker to do something with
    objects contained in the worker.object_store registry (or whatever the official object store is
    backed with in the case that it's been overridden). Semantically, one could view all Messages
    as a kind of action, but when we say action this is what we mean. For example, telling a
    worker to take two tensors and add them together is an action. However, sending an object
    from one worker to another is not an action (and would instead use the ObjectMessage type)."""

    def __init__(self, action: Action):
        """Initialize an action message

        Args:
            message (Tuple): this is typically the args and kwargs of a method call on the client,
                but it can be any information necessary to execute the action properly.
            return_ids (Tuple): primarily for our async infrastructure (Plan, Protocol, etc.),
                the id of action results are set by the client. This allows the client to be able
                to predict where the results will be ahead of time. Importantly, this allows the
                client to pre-initalize the pointers to the future data, regardless of whether
                the action has yet executed. It also reduces the size of the response from the
                action (which is very often empty).

        """

        self.action = action

    @property
    def name(self):
        return self.action.name

    @property
    def target(self):
        return self.action.target

    @property
    def args(self):
        return self.action.args

    @property
    def kwargs(self):
        return self.action.kwargs

    @property
    def return_ids(self):
        return self.action.return_ids

    @property
    def return_value(self):
        return self.action.return_value

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({type(self).__name__} {self.action})"

    @staticmethod
    def computation(name, target, args_, kwargs_, return_ids, return_value=False):
        """Helper function to build a TensorCommandMessage containing a ComputationAction
        directly from the action arguments.
        """
        action = ComputationAction(name, target, args_, kwargs_, return_ids, return_value)
        return TensorCommandMessage(action)

    @staticmethod
    def communication(name, target, args_, kwargs_, return_ids):
        """Helper function to build a TensorCommandMessage containing a CommunicationAction
        directly from the action arguments.
        """
        action = CommunicationAction(name, target, args_, kwargs_, return_ids)
        return TensorCommandMessage(action)

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: "TensorCommandMessage") -> tuple:
        """
        This function takes the attributes of a TensorCommandMessage and saves them in a tuple
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            ptr (TensorCommandMessage): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(ptr)
        """
        return (sy.serde.msgpack.serde._simplify(worker, ptr.action),)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "TensorCommandMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into a TensorCommandMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (TensorCommandMessage): an TensorCommandMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        simplified_action = msg_tuple[0]

        detailed_action = sy.serde.msgpack.serde._detail(worker, simplified_action)

        return TensorCommandMessage(detailed_action)

    @staticmethod
    def bufferize(
        worker: AbstractWorker, action_message: "TensorCommandMessage"
    ) -> "CommandMessagePB":
        """
        This function takes the attributes of a TensorCommandMessage and saves them in Protobuf
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            action_message (TensorCommandMessage): an TensorCommandMessage
        Returns:
            protobuf_obj: a Protobuf message holding the unique attributes of the message
        Examples:
            data = bufferize(message)
        """
        protobuf_action_msg = CommandMessagePB()

        protobuf_action = sy.serde.protobuf.serde._bufferize(worker, action_message.action)

        if isinstance(action_message.action, ComputationAction):
            protobuf_action_msg.computation.CopyFrom(protobuf_action)
        elif isinstance(action_message.action, CommunicationAction):
            protobuf_action_msg.communication.CopyFrom(protobuf_action)

        return protobuf_action_msg

    @staticmethod
    def unbufferize(
        worker: AbstractWorker, protobuf_obj: "CommandMessagePB"
    ) -> "TensorCommandMessage":
        """
        This function takes the Protobuf version of this message and converts
        it into an TensorCommandMessage. The bufferize() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            protobuf_obj (CommandMessagePB): the Protobuf message

        Returns:
            obj (TensorCommandMessage): an TensorCommandMessage

        Examples:
            message = unbufferize(sy.local_worker, protobuf_msg)
        """
        action = getattr(protobuf_obj, protobuf_obj.WhichOneof("action"))
        detailed_action = sy.serde.protobuf.serde._unbufferize(worker, action)
        return TensorCommandMessage(detailed_action)

    @staticmethod
    def get_protobuf_schema():
        """
        Returns the protobuf schema used for TensorCommandMessage.

        Returns:
            Protobuf schema for torch.Size.
        """
        return CommandMessagePB


class ObjectMessage(Message):
    """Send an object to another worker using this message type.

    When a worker has an object in its local object repository (such as a tensor) and it wants
    to send that object to another worker (and delete its local copy), it uses this message type
    to do so.
    """

    def __init__(self, object_):
        """Initialize the message."""

        self.object = object_

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({type(self).__name__} {self.object})"

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "ObjectMessage") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(msg)
        """
        return (sy.serde.msgpack.serde._simplify(worker, msg.object),)

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
        bufferized_obj = sy.serde.protobuf.serde._bufferize(worker, message.object)
        protobuf_obj_msg.tensor.CopyFrom(bufferized_obj)
        return protobuf_obj_msg

    @staticmethod
    def unbufferize(worker, protobuf_obj):
        """
        This method deserializes ObjectMessagePB into ObjectMessage.

        Args:
            protobuf_obj (ObjectMessagePB): input serialized ObjectMessagePB.

        Returns:
            object_msg (ObjectMessage): deserialized ObjectMessagePB.
        """
        protobuf_obj = protobuf_obj.tensor
        object_ = sy.serde.protobuf.serde._unbufferize(worker, protobuf_obj)
        object_msg = ObjectMessage(object_)

        return object_msg

    @staticmethod
    def get_protobuf_schema():
        """
        Returns the protobuf schema used for ObjectMessage.

        Returns:
            Protobuf schema for ObjectMessage.
        """
        return ObjectMessagePB


class ObjectRequestMessage(Message):
    """Request another worker to send one of its objects

    If ObjectMessage pushes an object to another worker, this Message type pulls an
    object from another worker. It also assumes that the other worker will delete it's
    local copy of the object after sending it to you."""

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, obj_id, user, reason):
        """Initialize the message."""

        self.object_id = obj_id
        self.user = user
        self.reason = reason

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({type(self).__name__} {self.object_id, self.user, self.reason})"

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "ObjectRequestMessage") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(msg)
        """
        return (
            sy.serde.msgpack.serde._simplify(worker, msg.object_id),
            sy.serde.msgpack.serde._simplify(worker, msg.user),
            sy.serde.msgpack.serde._simplify(worker, msg.reason),
        )

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
        return ObjectRequestMessage(
            sy.serde.msgpack.serde._detail(worker, msg_tuple[0]),
            sy.serde.msgpack.serde._detail(worker, msg_tuple[1]),
            sy.serde.msgpack.serde._detail(worker, msg_tuple[2]),
        )

    @staticmethod
    def bufferize(worker, msg):
        """
        This method serializes a ObjectRequestMessage using ObjectRequestMessagePB.

        Args:
            msg (ObjectRequestMessage): input ObjectRequestMessage to be serialized.

        Returns:
            proto_msg (ObjectRequestMessagePB): serialized ObjectRequestMessage.
        """
        proto_msg = ObjectRequestMessagePB()
        sy.serde.protobuf.proto.set_protobuf_id(proto_msg.object_id, msg.object_id)
        proto_msg.reason = msg.reason
        return proto_msg

    @staticmethod
    def unbufferize(worker, proto_msg):
        """
        This method deserializes ObjectRequestMessagePB into ObjectRequestMessage.

        Args:
            protobuf_msg (ObjectRequestMessagePB): input serialized ObjectRequestMessagePB.

        Returns:
           ObjectRequestMessage: deserialized ObjectRequestMessagePB.
        """
        obj_id = sy.serde.protobuf.proto.get_protobuf_id(proto_msg.object_id)
        # add worker support when it will be available
        return ObjectRequestMessage(obj_id=obj_id, user=None, reason=proto_msg.reason)

    @staticmethod
    def get_protobuf_schema():
        """
        Returns the protobuf schema used for ObjectRequestMessage.

        Returns:
            Protobuf schema for ObjectRequestMessage.
        """
        return ObjectRequestMessagePB


class IsNoneMessage(Message):
    """Check if a worker does not have an object with a specific id.

    Occasionally we need to verify whether or not a remote worker has a specific
    object. To do so, we send an IsNoneMessage, which returns True if the object
    (such as a tensor) does NOT exist."""

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, obj_id):
        """Initialize the message."""

        self.object_id = obj_id

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({type(self).__name__} {self.object_id})"

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "IsNoneMessage") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(msg)
        """
        return (sy.serde.msgpack.serde._simplify(worker, msg.object_id),)

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

    @staticmethod
    def bufferize(worker, msg):
        """
        This method serializes a IsNoneMessage using IsNoneMessagePB.

        Args:
            msg (IsNoneMessage): input IsNoneMessage to be serialized.

        Returns:
            protobuf_script (IsNoneMessagePB): serialized IsNoneMessage.
        """
        proto_msg = IsNoneMessagePB()
        sy.serde.protobuf.proto.set_protobuf_id(proto_msg.object_id, msg.object_id)
        return proto_msg

    @staticmethod
    def unbufferize(worker, proto_msg):
        """
        This method deserializes IsNoneMessagePB into IsNoneMessage.

        Args:
            protobuf_msg (IsNoneMessagePB): input serialized IsNoneMessagePB.

        Returns:
            IsNoneMessage: deserialized IsNoneMessagePB.
        """
        obj_id = sy.serde.protobuf.proto.get_protobuf_id(proto_msg.object_id)
        return IsNoneMessage(obj_id=obj_id)

    @staticmethod
    def get_protobuf_schema():
        """
        Returns the protobuf schema used for IsNoneMessage.

        Returns:
            Protobuf schema for ObjectRequestMessage.
        """
        return IsNoneMessagePB


class GetShapeMessage(Message):
    """Get the shape property of a tensor in PyTorch

    We needed to have a special message type for this because .shape had some
    constraints in the older version of PyTorch."""

    # TODO: remove this message type and use ObjectRequestMessage instead.
    # note that the above to do is likely waiting for custom tensor type support in PyTorch
    # https://github.com/OpenMined/PySyft/issues/2513

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, tensor_id):
        """Initialize the message."""

        self.tensor_id = tensor_id

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({type(self).__name__} {self.tensor_id})"

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "GetShapeMessage") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(msg)
        """
        return (sy.serde.msgpack.serde._simplify(worker, msg.tensor_id),)

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
            msg (GetShapeMessage): a GetShapeMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return GetShapeMessage(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))

    @staticmethod
    def bufferize(worker, msg):
        """
        This method serializes a GetShapeMessage using GetShapeMessagePB.

        Args:
            msg (GetShapeMessage): input GetShapeMessage to be serialized.

        Returns:
            proto_msg (GetShapeMessagePB): serialized GetShapeMessage.
        """
        proto_msg = GetShapeMessagePB()
        sy.serde.protobuf.proto.set_protobuf_id(proto_msg.object_id, msg.tensor_id)
        return proto_msg

    @staticmethod
    def unbufferize(worker, proto_obj):
        """
        This method deserializes GetShapeMessagePB into GetShapeMessage.

        Args:
            protobuf_obj (GetShapeMessage): input serialized GetShapeMessagePB.

        Returns:
            GetShapeMessage: deserialized GetShapeMessagePB.
        """
        tensor_id = sy.serde.protobuf.proto.get_protobuf_id(proto_obj.object_id)
        return GetShapeMessage(tensor_id=tensor_id)

    @staticmethod
    def get_protobuf_schema():
        """
        Returns the protobuf schema used for GetShapeMessage.

        Returns:
            Protobuf schema for GetShapeMessage.
        """
        return GetShapeMessagePB


class ForceObjectDeleteMessage(Message):
    """Garbage collect a remote object

    This is the dominant message for garbage collection of remote objects. When
    a pointer is deleted, this message is triggered by default to tell the object
    being pointed to to also delete itself.
    """

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, obj_ids):
        """Initialize the message."""

        self.object_ids = obj_ids

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({type(self).__name__} {self.object_ids})"

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "ForceObjectDeleteMessage") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(msg)
        """
        return sy.serde.msgpack.serde._simplify(worker, msg.object_ids)

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
            msg (ForceObjectDeleteMessage): a ForceObjectDeleteMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return ForceObjectDeleteMessage(sy.serde.msgpack.serde._detail(worker, msg_tuple))

    @staticmethod
    def bufferize(worker, msg):
        """
        This method serializes a ForceObjectDeleteMessage using ForceObjectDeleteMessagePB.

        Args:
            msg (ForceObjectDeleteMessage): input ForceObjectDeleteMessage to be serialized.

        Returns:
            proto_msg (ForceObjectDeleteMessagePB): serialized ForceObjectDeleteMessage.
        """
        proto_msg = ForceObjectDeleteMessagePB()
        for elem in msg.object_ids:
            id = IdPB()
            if isinstance(elem, str):
                id.id_str = elem
            else:
                id.id_int = elem
            proto_msg.object_ids.append(id)
        return proto_msg

    @staticmethod
    def unbufferize(worker, proto_msg):
        """
        This method deserializes ForceObjectDeleteMessagePB into ForceObjectDeleteMessage.

        Args:
            proto_msg (ForceObjectDeleteMessagePB): input serialized ForceObjectDeleteMessagePB.

        Returns:
            ForceObjectDeleteMessage: deserialized ForceObjectDeleteMessagePB.
        """
        obj_ids = []
        for elem in proto_msg.object_ids:
            obj_ids.append(getattr(elem, elem.WhichOneof("id")))

        return ForceObjectDeleteMessage(obj_ids=obj_ids)

    @staticmethod
    def get_protobuf_schema():
        """
        Returns the protobuf schema used for ForceObjectDeleteMessage.

        Returns:
            Protobuf schema for ForceObjectDeleteMessage.
        """
        return ForceObjectDeleteMessagePB


class SearchMessage(Message):
    """A client queries for a subset of the tensors on a remote worker using this type

    For some workers like SocketWorker we split a worker into a client and a server. For
    this configuration, a client can request to search for a subset of tensors on the server
    using this message type (this could also be called a "QueryMessage").
    """

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, query):
        """Initialize the message."""

        self.query = query

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({type(self).__name__} {self.query})"

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "SearchMessage") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(msg)
        """
        return (sy.serde.msgpack.serde._simplify(worker, msg.query),)

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

    @staticmethod
    def bufferize(worker, msg):
        """
        This method serializes a SearchMessage using SearchMessagePB.

        Args:
            msg (SearchMessage): input SearchMessage to be serialized.

        Returns:
            proto_msg (SearchMessagePB): serialized SearchMessage.
        """
        proto_msg = SearchMessagePB()
        for elem in msg.query:
            id = IdPB()
            if isinstance(elem, str):
                id.id_str = elem
            else:
                id.id_int = elem
            proto_msg.query.append(id)
        return proto_msg

    @staticmethod
    def unbufferize(worker, proto_obj):
        """
        This method deserializes SearchMessagePB into SearchMessage.

        Args:
            proto_msg (SearchMessagePB): input serialized SearchMessagePB.

        Returns:
            SearchMessage: deserialized SearchMessagePB.
        """
        query = []
        for elem in proto_obj.query:
            query.append(getattr(elem, elem.WhichOneof("id")))

        return SearchMessage(query=query)

    @staticmethod
    def get_protobuf_schema():
        """
        Returns the protobuf schema used for SearchMessage.

        Returns:
            Protobuf schema for SearchMessage.
        """
        return SearchMessagePB


class PlanCommandMessage(Message):
    """Message used to execute a command related to plans."""

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, command_name: str, args_: tuple):
        """Initialize a PlanCommandMessage.

        Args:
            command_name (str): name used to identify the command.
            message (Tuple): this is typically the args and kwargs of a method call on the client,
                but it can be any information necessary to execute the command properly.
        """

        self.command_name = command_name
        self.args = args_

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({type(self).__name__} {(self.command_name, self.args)})"

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "PlanCommandMessage") -> tuple:
        """
        This function takes the attributes of a PlanCommandMessage and saves them in a tuple

        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (PlanCommandMessage): a Message

        Returns:
            tuple: a tuple holding the unique attributes of the message
        """
        return (
            sy.serde.msgpack.serde._simplify(worker, msg.command_name),
            sy.serde.msgpack.serde._simplify(worker, msg.args),
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
        command_name, args_ = msg_tuple
        return PlanCommandMessage(
            sy.serde.msgpack.serde._detail(worker, command_name),
            sy.serde.msgpack.serde._detail(worker, args_),
        )

    @staticmethod
    def bufferize(worker, msg):
        """
        This method serializes a PlanCommandMessage using PlanCommandMessagePB.

        Args:
            msg (PlanCommandMessage): input PlanCommandMessage to be serialized.

        Returns:
            proto_msg (PlanCommandMessagePB): serialized PlanCommandMessage.
        """
        proto_msg = PlanCommandMessagePB()
        proto_msg.command_name = msg.command_name
        for arg in sy.serde.protobuf.serde.bufferize_args(worker, msg.args):
            proto_msg.args.append(arg)
        return proto_msg

    @staticmethod
    def unbufferize(worker, proto_msg):
        """
        This method deserializes PlanCommandMessagePB into PlanCommandMessage.

        Args:
            proto_msg (PlanCommandMessagePB): input serialized PlanCommandMessagePB.

        Returns:
            PlanCommandMessage: deserialized PlanCommandMessagePB.
        """
        args = sy.serde.protobuf.serde.unbufferize_args(worker, proto_msg.args)
        return PlanCommandMessage(command_name=proto_msg.command_name, args_=tuple(args))

    @staticmethod
    def get_protobuf_schema():
        """
        Returns the protobuf schema used for PlanCommandMessage.

        Returns:
            Protobuf schema for PlanCommandMessage.
        """
        return PlanCommandMessagePB


class WorkerCommandMessage(Message):
    """Message used to execute a function of the remote worker."""

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, command_name: str, message: tuple):
        """Initialize a WorkerCommandMessage.

        Args:
            command_name (str): name used to identify the command.
            message (Tuple): this is typically the args and kwargs of a method call on the client,
                but it can be any information necessary to execute the command properly.
        """

        # call the parent constructor - setting the type integer correctly
        super().__init__()

        self.command_name = command_name
        self.message = message

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({type(self).__name__} {(self.command_name, self.message)})"

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: "WorkerCommandMessage") -> tuple:
        """
        This function takes the attributes of a WorkerCommandMessage and saves them in a tuple

        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            ptr (WorkerCommandMessage): a Message

        Returns:
            tuple: a tuple holding the unique attributes of the message
        """
        return (
            sy.serde.msgpack.serde._simplify(worker, ptr.command_name),
            sy.serde.msgpack.serde._simplify(worker, ptr.message),
        )

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "WorkerCommandMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into a WorkerCommandMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (WorkerCommandMessage): a WorkerCommandMessage.
        """
        command_name, message = msg_tuple
        return WorkerCommandMessage(
            sy.serde.msgpack.serde._detail(worker, command_name),
            sy.serde.msgpack.serde._detail(worker, message),
        )

    @staticmethod
    def bufferize(worker, msg):
        """
        This method serializes a WorkerCommandMessage using WorkerCommandMessagePB.

        Args:
            msg (WorkerCommandMessage): input WorkerCommandMessage to be serialized.

        Returns:
            proto_msg (WorkerCommandMessagePB): serialized WorkerCommandMessage.
        """
        proto_msg = WorkerCommandMessage()
        proto_msg.command_name = msg.command_name
        for arg in sy.serde.protobuf.serde.bufferize_args(worker, msg.args):
            proto_msg.args.append(arg)
        return proto_msg

    @staticmethod
    def unbufferize(worker, proto_msg):
        """
        This method deserializes WorkerCommandMessagePB into WorkerCommandMessage.

        Args:
            proto_msg (WorkerCommandMessagePB): input serialized WorkerCommandMessagePB.

        Returns:
            WorkerCommandMessage: deserialized WorkerCommandMessagePB.
        """
        args = sy.serde.protobuf.serde.unbufferize_args(worker, proto_msg.args)
        return WorkerCommandMessage(command_name=proto_msg.command_name, args_=tuple(args))

    # TODO: when testing is fixed, uncomment this to enable worker command message support.
    # @staticmethod
    # def get_protobuf_schema():
    #     """
    #         Returns the protobuf schema used for WorkerCommandMessage.
    #
    #         Returns:
    #             Protobuf schema for WorkerCommandMessage.
    #     """
    #     return WorkerCommandMessagePB


class CryptenInitPlan(Message):
    """Initialize a Crypten party using this message.

    Crypten uses processes as parties, those processes need to be initialized with information
    so they can communicate and exchange tensors and shares while doing computation. This message
    allows the exchange of information such as the ip and port of the master party to connect to,
    as well as the rank of the party to run and the number of parties involved."""

    def __init__(self, crypten_context, model=None):
        # crypten_context = (rank_to_worker_ids, world_size, master_addr, master_port)
        self.crypten_context = crypten_context
        self.model = model

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({type(self).__name__} {self.crypten_context})"

    @staticmethod
    def simplify(worker: AbstractWorker, message: "CryptenInitPlan") -> tuple:
        """
        This function takes the attributes of a CryptenInitPlan and saves them in a tuple

        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            ptr (CryptenInitPlan): a Message

        Returns:
            tuple: a tuple holding the unique attributes of the message
        """
        return (
            sy.serde.msgpack.serde._simplify(worker, (*message.crypten_context, message.model)),
        )

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "CryptenInitPlan":
        """
        This function takes the simplified tuple version of this message and converts
        it into a CryptenInitPlan. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.

        Returns:
            CryptenInitPlan message.

        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        msg_tuple = sy.serde.msgpack.serde._detail(worker, msg_tuple[0])
        *context, model = msg_tuple
        return CryptenInitPlan(tuple(context), model)


class CryptenInitJail(Message):
    """Initialize a Crypten party using this message.

    Crypten uses processes as parties, those processes need to be initialized with information
    so they can communicate and exchange tensors and shares while doing computation. This message
    allows the exchange of information such as the ip and port of the master party to connect to,
    as well as the rank of the party to run and the number of parties involved. Compared to
    CryptenInitPlan, this message also sends two extra fields, a JailRunner and a Crypten model."""

    def __init__(self, crypten_context, jail_runner, model=None):
        # crypten_context = (rank_to_worker_ids, world_size, master_addr, master_port)
        self.crypten_context = crypten_context
        self.jail_runner = jail_runner
        self.model = model

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({type(self).__name__} {self.crypten_context}, {self.jail_runner})"

    @staticmethod
    def simplify(worker: AbstractWorker, message: "CryptenInitJail") -> tuple:
        """
        This function takes the attributes of a CryptenInitJail and saves them in a tuple

        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            ptr (CryptenInitJail): a Message

        Returns:
            tuple: a tuple holding the unique attributes of the message
        """
        return (
            sy.serde.msgpack.serde._simplify(
                worker, (*message.crypten_context, message.jail_runner, message.model)
            ),
        )

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "CryptenInitJail":
        """
        This function takes the simplified tuple version of this message and converts
        it into a CryptenInitJail. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.

        Returns:
            CryptenInitJail message.

        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        msg_tuple = sy.serde.msgpack.serde._detail(worker, msg_tuple[0])
        *context, jail_runner, model = msg_tuple
        return CryptenInitJail(tuple(context), jail_runner, model)
