import copy
from typing import List
from typing import Tuple
from typing import Union

import syft as sy
from syft.generic.frameworks.types import FrameworkTensorType
from syft.generic.tensor import AbstractTensor
from syft.messaging.plan.state import State
from syft.workers.abstract import AbstractWorker


class Command(object):
    """A command is a serializable object providing instruction to operate designated tensors."""

    def __init__(self, command):
        self.command = command

    def update_ids(self, from_ids, to_ids):
        self.command = Command.replace_ids(self.command, from_ids, to_ids)

    @staticmethod
    def replace_ids(command, from_ids, to_ids):

        assert isinstance(from_ids, (list, tuple))
        assert isinstance(to_ids, (list, tuple))

        type_obj = type(command)
        command = list(command)
        for i, item in enumerate(command):
            if isinstance(item, (int, str, bytes)) and item in from_ids:
                command[i] = to_ids[from_ids.index(item)]
            elif isinstance(item, (list, tuple)):
                command[i] = Command.replace_ids(command=item, from_ids=from_ids, to_ids=to_ids)
        return type_obj(command)


class Procedure(object):
    """A Procedure is a list of commands."""

    def __init__(self, commands=None, arg_ids=None, result_ids=None):
        self.commands = commands or []
        self.arg_ids = arg_ids or []
        self.result_ids = result_ids or []

    def __str__(self):
        return f"<Procedure #commands:{len(self.commands)}>"

    def __repr__(self):
        return self.__str__()

    def update_ids(
        self,
        from_ids: Tuple[Union[str, int]] = [],
        to_ids: Tuple[Union[str, int]] = [],
        from_worker: Union[str, int] = None,
        to_worker: Union[str, int] = None,
    ):
        """Replaces pairs of tensor ids in the plan stored.

        Args:
            from_ids: Ids to change.
            to_ids: Ids to replace with.
            from_worker: The previous worker that built the plan.
            to_worker: The new worker that is running the plan.
        """
        for idx, command in enumerate(self.commands):
            if from_worker and to_worker:
                from_workers, to_workers = [from_worker], [to_worker]
                if isinstance(from_worker, str):
                    from_workers.append(from_worker.encode("utf-8"))
                    to_workers.append(to_worker)
                command = Command.replace_ids(command, from_workers, to_workers)

            if len(from_ids) and len(to_ids):
                command = Command.replace_ids(command, from_ids, to_ids)

            self.commands[idx] = command

        return self

    def update_worker_ids(self, from_worker_id: Union[str, int], to_worker_id: Union[str, int]):
        return self.update_ids([], [], from_worker_id, to_worker_id)

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

    def copy(self) -> "Procedure":
        procedure = Procedure(
            commands=copy.deepcopy(self.commands), arg_ids=self.arg_ids, result_ids=self.result_ids
        )
        return procedure

    @staticmethod
    def simplify(procedure: "Procedure") -> tuple:
        return (
            tuple(
                procedure.commands
            ),  # We're not simplifying because commands are already simplified
            sy.serde._simplify(procedure.arg_ids),
            sy.serde._simplify(procedure.result_ids),
        )

    @staticmethod
    def detail(worker: AbstractWorker, procedure_tuple: tuple) -> "State":
        commands, arg_ids, result_ids = procedure_tuple
        commands = list(commands)
        arg_ids = sy.serde._detail(worker, arg_ids)
        result_ids = sy.serde._detail(worker, result_ids)

        procedure = Procedure(commands, arg_ids, result_ids)
        return procedure
