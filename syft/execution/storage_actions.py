from syft.execution.action import Action
from syft.workers.abstract import AbstractWorker

from syft_proto.execution.v1.storage_action_pb2 import (
    StorageAction as StorageActionPB,
    # HACK replace this with the correct syft proto objects and update proto version in reqs
    # and update documentations
)

STORAGE_ACTION_METHODS = ["store", "load"]


class StorageAction(Action):
    """Describes woker actions related to object storage"""

    def __init__(self, name, target, args_, kwargs_, return_ids, return_value=False):
        """Initialize an action

        Args:
            name (String): The name of the method to be invoked (e.g. "send")
            target (Tensor): The object to invoke the method on
            args_ (Tuple): The arguments to the method call
            kwargs_ (Dictionary): The keyword arguments to the method call
            return_ids (Tuple): primarily for our async infrastructure (Plan, Protocol, etc.), the
                id of action results are set by the client. This allows the client to be able to
                predict where the results will be ahead of time. Importantly, this allows the
                client to pre-initalize the pointers to the future data, regardless of whether
                the action has yet executed. It also
                reduces the size of the response from the action (which is very often empty).
            return_value (boolean): return the result or not. If true, the result is directly
                returned, if not, the command sender will create a pointer to the remote result
                using the return_ids and will need to do .get() later to get the result.

        """
        if name not in STORAGE_ACTION_METHODS:
            raise ValueError(
                f"Method `{name}` is not supported by StorageActions. Consider using \
                ComputationAction or CommunicationAction instead."
            )

        super().__init__(name, target, args_, kwargs_, return_ids, return_value=return_value)

    @staticmethod
    def simplify(worker: AbstractWorker, action: "Action") -> tuple:
        """
        This function takes the attributes of a StorageAction and saves them in a tuple
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            action (StorageAction): a StorageAction
        Returns:
            tuple: a tuple holding the unique attributes of the StorageAction
        Examples:
            data = simplify(worker, action)
        """
        return Action.simplify(worker, action)

    @staticmethod
    def detail(worker: AbstractWorker, action_tuple: tuple) -> "Action":
        """
        This function takes the simplified tuple version of this message and converts
        it into a StorageAction. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            StorageAction_tuple (Tuple): the raw information being detailed.
        Returns:
            StorageAction (StorageAction): a StorageAction.
        Examples:
            StorageAction = detail(sy.local_worker,StorageAction_tuple)
        """
        attrs = Action.detail(worker, action_tuple)

        return StorageAction(*attrs)

    @staticmethod
    def bufferize(worker: AbstractWorker, StorageAction: "StorageAction") -> "StorageActionPB":
        """
        This function takes the attributes of a StorageAction and saves them in Protobuf
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            StorageAction (StorageAction): a StorageAction
        Returns:
            protobuf_obj: a Protobuf message holding the unique attributes of the StorageAction
        Examples:
            data = bufferize(sy.local_worker, StorageAction)
        """
        protobuf_action = StorageActionPB()

        return Action.bufferize(worker, StorageAction, protobuf_action)

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_obj: "StorageActionPB") -> "StorageAction":
        """
        This function takes the Protobuf version of this message and converts
        it into an Action. The bufferize() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            protobuf_obj (StorageActionPB): the Protobuf message

        Returns:
            obj (StorageAction): a StorageAction

        Examples:
            message = unbufferize(sy.local_worker, protobuf_msg)
        """
        attrs = Action.unbufferize(worker, protobuf_obj)

        return StorageAction(*attrs)

    @staticmethod
    def get_protobuf_schema() -> StorageActionPB:
        return StorageActionPB
