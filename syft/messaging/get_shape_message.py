import syft as sy

from syft.messaging.message import Message
from syft.workers.abstract import AbstractWorker


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
