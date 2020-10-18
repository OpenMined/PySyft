from syft.workers.abstract import AbstractWorker

from syft.execution.action import Action

from syft_proto.execution.v1.computation_action_pb2 import ComputationAction as ComputationActionPB


class ComputationAction(Action):
    """Describes mathematical operations performed on tensors"""

    def __init__(self, name, target, args_, kwargs_, return_ids, return_value=False):
        """Initialize an action

        Args:
            name (String): The name of the method to be invoked (e.g. "__add__")
            target (Tensor): The object to invoke the method on
            args_ (Tuple): The arguments to the method call
            kwargs_ (Dictionary): The keyword arguments to the method call
            return_ids (Tuple): primarily for our async infrastructure (Plan, Protocol, etc.),
                the id of action results are set by the client. This allows the client to be able
                to predict where the results will be ahead of time. Importantly, this allows the
                client to pre-initalize the pointers to the future data, regardless of whether the
                action has yet executed. It also reduces the size of the response from the action
                (which is very often empty).
            return_value (boolean): return the result or not. If true, the result is directly
                returned, if not, the command sender will create a pointer to the remote result
                using the return_ids and will need to do .get() later to get the result.

        """
        super().__init__(name, target, args_, kwargs_, return_ids, return_value=return_value)

    def copy(self):
        return ComputationAction(self.name, self.target, self.args, self.kwargs, self.return_ids)

    @staticmethod
    def simplify(worker: AbstractWorker, action: "Action") -> tuple:
        """
        This function takes the attributes of a ComputationAction and saves them in a tuple
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            action (ComputationAction): a ComputationAction
        Returns:
            tuple: a tuple holding the unique attributes of the ComputationAction
        Examples:
            data = simplify(worker, action)
        """
        return Action.simplify(worker, action)

    @staticmethod
    def detail(worker: AbstractWorker, action_tuple: tuple) -> "Action":
        """
        This function takes the simplified tuple version of this message and converts
        it into a ComputationAction. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            communication_tuple (Tuple): the raw information being detailed.
        Returns:
            communication (ComputationAction): a ComputationAction.
        Examples:
            communication = detail(sy.local_worker, communication_tuple)
        """
        attrs = Action.detail(worker, action_tuple)

        return ComputationAction(*attrs)

    @staticmethod
    def bufferize(
        worker: AbstractWorker, communication: "ComputationAction"
    ) -> "ComputationActionPB":
        """
        This function takes the attributes of a ComputationAction and saves them in Protobuf
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            communication (ComputationAction): a ComputationAction
        Returns:
            protobuf_obj: a Protobuf message holding the unique attributes of the communication
        Examples:
            data = bufferize(sy.local_worker, communication)
        """
        protobuf_action = ComputationActionPB()

        return Action.bufferize(worker, communication, protobuf_action)

    @staticmethod
    def unbufferize(
        worker: AbstractWorker, protobuf_obj: "ComputationActionPB"
    ) -> "ComputationAction":
        """
        This function takes the Protobuf version of this message and converts
        it into an Action. The bufferize() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            protobuf_obj (ComputationActionPB): the Protobuf message

        Returns:
            obj (ComputationAction): a ComputationAction

        Examples:
            message = unbufferize(sy.local_worker, protobuf_msg)
        """
        attrs = Action.unbufferize(worker, protobuf_obj)

        return ComputationAction(*attrs)

    @staticmethod
    def get_protobuf_schema() -> ComputationActionPB:
        return ComputationActionPB
