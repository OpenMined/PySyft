import syft as sy
from syft.workers.abstract import AbstractWorker

from syft.execution.action import Action


class CommunicationAction(Action):
    """Describes communication actions performed on tensors"""

    def __init__(self, object, source: AbstractWorker, *destinations: AbstractWorker, **kwargs):
        """Initialize an communication action

        Args:
        """
        super().__init__()

        self.obj = obj
        self.source = source
        self.destinations = destinations
        self.kwargs = kwargs

    @staticmethod
    def simplify(worker: AbstractWorker, communication: "CommunicationAction") -> tuple:
        """
        This function takes the attributes of a CommunicationAction and saves them in a tuple
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            communication (CommunicationAction): a CommunicationAction
        Returns:
            tuple: a tuple holding the unique attributes of the CommunicationAction
        Examples:
            data = simplify(worker, communication)
        """
        return (
            sy.serde.msgpack.serde._simplify(worker, communication.obj),
            sy.serde.msgpack.serde._simplify(worker, communication.source),
            sy.serde.msgpack.serde._simplify(worker, communication.destinations),
            sy.serde.msgpack.serde._simplify(worker, communication.kwargs),
        )

    @staticmethod
    def detail(worker: AbstractWorker, communication_tuple: tuple) -> "CommunicationAction":
        """
        This function takes the simplified tuple version of this message and converts
        it into a CommunicationAction. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            communication_tuple (Tuple): the raw information being detailed.
        Returns:
            communication (CommunicationAction): a CommunicationAction.
        Examples:
            communication = detail(sy.local_worker, communication_tuple)
        """

        (obj, source, destinations, kwargs) = communication_tuple

        detailed_obj = sy.serde.msgpack.serde._detail(worker, obj)
        detailed_source = sy.serde.msgpack.serde._detail(worker, source)
        detailed_destinations = sy.serde.msgpack.serde._detail(worker, destinations)
        detailed_kwargs = sy.serde.msgpack.serde._detail(worker, kwargs)

        return CommunicationAction(
            detailed_obj, detailed_source, detailed_destinations, detailed_kwargs
        )

    @staticmethod
    def bufferize(
        worker: AbstractWorker, communication: "CommunicationAction"
    ) -> "CommunicationActionPB":
        """
        This function takes the attributes of a CommunicationAction and saves them in Protobuf
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            communication (CommunicationAction): a CommunicationAction
        Returns:
            protobuf_obj: a Protobuf message holding the unique attributes of the communication
        Examples:
            data = bufferize(sy.local_worker, communication)
        """
        pass

    @staticmethod
    def unbufferize(
        worker: AbstractWorker, protobuf_obj: "CommunicationActionPB"
    ) -> "CommunicationAction":
        """
        This function takes the Protobuf version of this message and converts
        it into a CommunicationAction. The bufferize() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            protobuf_obj (CommunicationActionPB): the Protobuf message

        Returns:
            obj (CommunicationAction): a CommunicationAction

        Examples:
            message = unbufferize(sy.local_worker, protobuf_msg)
        """
        pass
