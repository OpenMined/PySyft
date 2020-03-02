import syft as sy

from syft.messaging.message import Message
from syft.workers.abstract import AbstractWorker


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
