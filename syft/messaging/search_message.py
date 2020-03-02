import syft as sy

from syft.messaging.message import Message
from syft.workers.abstract import AbstractWorker


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
