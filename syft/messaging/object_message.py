import syft as sy

from syft.messaging.message import Message
from syft.workers.abstract import AbstractWorker


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
