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
        operations: the list of (serialized) operations
        arg_ids: the argument ids present in the operations
        result_ids: the result ids present in the operations
    """

    def __init__(self, operations=None, input_placeholders=None, output_placeholders=None):
        self.operations = operations or []
        self.input_placeholders = input_placeholders or []
        self.output_placeholders = output_placeholders or []

    def __str__(self):
        return f"<Procedure #operations:{len(self.operations)}>"

    def __repr__(self):
        return self.__str__()

    def copy(self) -> "Procedure":
        # TODO: Sort out deep copying here
        procedure = Procedure(
            operations=self.operations,
            input_placeholders=self.input_placeholders,
            output_placeholders=self.output_placeholders,
        )
        return procedure

    def update_inputs(self, args: Tuple[Union[FrameworkTensorType, AbstractTensor]]):
        """Replace args and result_placeholders with the ones given.
        Updates the arguments ids and result ids used to execute
        the plan.
        Args:
            args: List of tensors.
            result_ids: Ids where the plan output will be stored.
        """

        for placeholder, arg in zip(self.input_placeholders, args):
            placeholder.instantiate(arg)

    @staticmethod
    def simplify(worker: AbstractWorker, procedure: "Procedure") -> tuple:
        return (
            # We're not simplifying fully because operations are already simplified
            sy.serde.msgpack.serde._simplify(worker, procedure.operations),
            sy.serde.msgpack.serde._simplify(worker, procedure.input_placeholders),
            sy.serde.msgpack.serde._simplify(worker, procedure.output_placeholders),
        )

    @staticmethod
    def detail(worker: AbstractWorker, procedure_tuple: tuple) -> "State":
        operations, input_placeholders, output_placeholders = procedure_tuple

        operations = sy.serde.msgpack.serde._detail(worker, operations)
        input_placeholders = sy.serde.msgpack.serde._detail(worker, input_placeholders)
        output_placeholders = sy.serde.msgpack.serde._detail(worker, output_placeholders)

        procedure = Procedure(operations, input_placeholders, output_placeholders)
        return procedure
