from __future__ import annotations
from abc import ABC
from ...common.object import ObjectWithId
from ..supervisor import syft_supervisor

class AbstractWorker(ABC, ObjectWithId):
    """Base interface for the syft worker.

    A worker is a collection of objects owned by the worker, a list of
    supported frameworks used for remote execution and a message router. The
    objects owned by a worker are placed in an ObjectStore object.

    Attributes
        store (ObjectStore): the object responsible to handle the data owned
        by the worker.

        framworks (Globals): the collection of frameworks that the worker is
        able to use.

        msg_router (dict): mapping from the message types to the services that
        can handle them.

        worker_stats (WorkerStat): object that stores statics about the worker
        usage, mainly used for debugging purposes.

    Args:
        debug (bool): enables debugging and logging on a worker.
        services (ServiceTable): the services that this worker can handle.
    """

    def __init__(self, debug: bool = False, services=None):
        super().__init__()

    @syft_supervisor
    def recv_msg(self, msg) -> "SyftMsgResponse":
        """Method that handles messages received by the worker, by redirecting
        the message to the actual service that handles it through the msg_router
        attribute

        Args:
            msg (syft.core.message.SyftMessage): the message forwarded by the
            Client

        Returns:
            syft.core.message.SyftMsgResponse: the response for the message
            received as input.
        """
        pass

    def _recv_msg(self) -> None:
        """Method that handles the actual implementation for receiving
        messages.

        For example, a websocket worker should define the websocket programming
        logic on receiving messages in this method.
        """
        raise NotImplementedError

    def _send_msg(self) -> None:
        """Method that handles the actual implementation for sending messages.

        For example, a websocket worker should define the websocket programming
        login on sending messages in this method.
        """
        raise NotImplementedError

    def get_info(self) -> str:
        pass
