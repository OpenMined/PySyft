# stdlib
from typing import Any
from typing import Tuple

# third party
import pytest
import torch

# syft absolute
import syft as sy

test_args = [
    (sy.lib.python.List([1, 2, 3]), "append", (5,)),
    (torch.Tensor([1, 2, 3]), "add_", (torch.Tensor([1, 2, 3]),)),
    (sy.lib.python.collections.OrderedDict(), "__setitem__", ("key", "value")),
]


@pytest.mark.parametrize("obj, op, args", test_args)
def test_inplace_ops_gc_test(
    obj: Any,
    op: str,
    args: Tuple[Any],
    node: sy.VirtualMachine,
    client: sy.VirtualMachineClient,
) -> None:

    object_ptr = obj.send(client, pointable=False)
    initial_id = object_ptr.id_at_location
    assert len(node.store) == 1

    func = getattr(object_ptr, op)
    func(*args)  # inplace op

    assert object_ptr.id_at_location == initial_id
    object_ptr.__del__()

    assert len(node.store) == 0
