import syft as sy

from syft.messaging.message import Message
from syft.workers.abstract import AbstractWorker


class OperationMessage(Message):
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
        super().__init__()

        self.operation = Operation(cmd_name, cmd_owner, cmd_args, cmd_kwargs, return_ids)

    @property
    def contents(self):
        return self.operation.contents

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: "OperationMessage") -> tuple:
        return (sy.serde.msgpack.serde._simplify(worker, ptr.operation),)

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
        # TODO: Extract the single element from the tuple and delegate
        # to Operation detailer

        operation = msg_tuple[0]

        message = operation[0]
        return_ids = operation[1]

        detailed_msg = sy.serde.msgpack.serde._detail(worker, message)
        detailed_ids = sy.serde.msgpack.serde._detail(worker, return_ids)

        cmd_name = detailed_msg[0]
        cmd_owner = detailed_msg[1]
        cmd_args = detailed_msg[2]
        cmd_kwargs = detailed_msg[3]

        return OperationMessage(cmd_name, cmd_owner, cmd_args, cmd_kwargs, detailed_ids)
