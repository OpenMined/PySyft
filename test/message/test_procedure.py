import syft
import torch

from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.messaging.message import Operation
from syft.messaging.plan.procedure import Procedure


def test_procedure_update_ids(workers):
    worker = workers["me"]

    pointer1 = PointerTensor(
        workers["bob"], 68519530406, workers["me"], 27674294093, True, torch.Size([1])
    )

    pointer2 = PointerTensor(
        workers["bob"], 91383408771, workers["me"], 2843683950, False, torch.Size([1])
    )

    operation = Operation("__add__", pointer1, [pointer2], {}, (75165665059,))

    simplified_operation = syft.serde.msgpack.serde._simplify(worker, operation)

    procedure = Procedure(
        operations=[simplified_operation], arg_ids=[68519530406], result_ids=(75165665059,)
    )

    procedure.update_ids(
        from_ids=[27674294093], to_ids=[73570994542], from_worker="bob", to_worker="alice"
    )

    proc_ops = [
        syft.serde.msgpack.serde._detail(worker, simplified_operation)
        for simplified_operation in procedure.operations
    ]

    assert proc_ops[0].cmd_owner.id == 73570994542
    assert proc_ops[0].cmd_owner.location == workers["alice"]
    assert proc_ops[0].cmd_args[0].location == workers["alice"]

    tensor = torch.tensor([1.0])
    tensor_id = tensor.id
    procedure.update_args(args=(tensor,), result_ids=[8730174527])

    proc_ops = [
        syft.serde.msgpack.serde._detail(worker, simplified_operation)
        for simplified_operation in procedure.operations
    ]

    assert proc_ops[0].cmd_owner.id_at_location == tensor_id
    assert proc_ops[0].return_ids[0] == 8730174527
