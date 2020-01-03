import copy
from typing import List
from typing import Tuple
from typing import Union

import syft as sy
from syft.generic.frameworks.types import FrameworkTensorType
from syft.generic.tensor import AbstractTensor
from syft.messaging.plan.state import State
from syft.workers.abstract import AbstractWorker


class Procedure(object):
    """
    A Procedure is a wrapper over a list of operations to execute.

    It provides tools to update the operations to un with new arguments
    on different workers.

    Args:
        operations: the list of operations
        arg_ids: the argument ids present in the operations
        result_ids: the result ids present in the operations
    """

    def __init__(self, operations=None, arg_ids=None, result_ids=None):
        self.operations = operations or []
        self.arg_ids = arg_ids or []
        self.result_ids = result_ids or []
        # promise_out_id id used for plan augmented to be used with promises
        self.promise_out_id = None

    def __str__(self):
        return f"<Procedure #operations:{len(self.operations)}>"

    def __repr__(self):
        return self.__str__()

    def update_args(
        self,
        args: Tuple[Union[FrameworkTensorType, AbstractTensor]],
        result_ids: List[Union[str, int]],
    ):
        """Replace args and result_ids with the ones given.
        Updates the arguments ids and result ids used to execute
        the plan.
        Args:
            args: List of tensors.
            result_ids: Ids where the plan output will be stored.
        """

        arg_ids = tuple(arg.id for arg in args)
        self.update_ids(self.arg_ids, arg_ids)
        self.arg_ids = arg_ids

        self.update_ids(self.result_ids, result_ids)
        self.result_ids = result_ids

    def update_worker_ids(self, from_worker_id: Union[str, int], to_worker_id: Union[str, int]):
        return self.update_ids([], [], from_worker_id, to_worker_id)

    def update_ids(
        self,
        from_ids: Tuple[Union[str, int]] = [],
        to_ids: Tuple[Union[str, int]] = [],
        from_worker_id: Union[str, int] = None,
        to_worker_id: Union[str, int] = None,
    ):
        """Replace ids and worker ids in the list of operations stored

        Args:
            from_ids: Ids to change.
            to_ids: Ids to replace with.
            from_worker: The previous worker that built the plan.
            to_worker: The new worker that is running the plan.
        """
        for idx, operation in enumerate(self.operations):
            # replace ids in the owner
            owner = operation.cmd_owner
            if owner is not None:
                if owner.id in from_ids:
                    owner.id = to_ids[from_ids.index(owner.id)]

                if owner.id_at_location in from_ids:
                    owner.id_at_location = to_ids[from_ids.index(owner.id_at_location)]

            # replace ids in the args
            for arg in operation.cmd_args:
                try:
                    if arg.id in from_ids:
                        arg.id = to_ids[from_ids.index(arg.id)]

                    if arg.id_at_location in from_ids:
                        arg.id_at_location = to_ids[from_ids.index(arg.id_at_location)]
                except:
                    pass

            # replace ids in the returns
            return_ids = list(operation.return_ids)
            for idx, return_id in enumerate(return_ids):
                if return_id in from_ids:
                    return_ids[idx] = to_ids[from_ids.index(return_id)]
            operation.return_ids = return_ids

            # create a dummy worker
            to_worker = sy.workers.virtual.VirtualWorker(None, to_worker_id)

            # replace worker in the owner
            if owner is not None and owner.location is not None:
                if owner.location.id == from_worker_id:
                    owner.location = to_worker

            # replace worker in the args
            for arg in operation.cmd_args:
                try:
                    if arg.location.id == from_worker_id:
                        arg.location = to_worker
                except:
                    pass

            self.operations[idx] = operation

        return self

    @staticmethod
    def simplify(worker: AbstractWorker, procedure: "Procedure") -> tuple:
        return (
            sy.serde.msgpack.serde._simplify(worker, procedure.operations),
            sy.serde.msgpack.serde._simplify(worker, procedure.arg_ids),
            sy.serde.msgpack.serde._simplify(worker, procedure.result_ids),
            sy.serde.msgpack.serde._simplify(worker, procedure.promise_out_id),
        )

    @staticmethod
    def detail(worker: AbstractWorker, procedure_tuple: tuple) -> "State":
        operations, arg_ids, result_ids, promise_out_id = procedure_tuple

        operations = sy.serde.msgpack.serde._detail(worker, operations)
        arg_ids = sy.serde.msgpack.serde._detail(worker, arg_ids)
        result_ids = sy.serde.msgpack.serde._detail(worker, result_ids)

        procedure = Procedure(operations, arg_ids, result_ids)
        procedure.promise_out_id = promise_out_id
        return procedure
