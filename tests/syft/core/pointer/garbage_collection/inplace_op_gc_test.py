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


test_args_inplace = [
    (torch.Tensor([1, 2, 3]), "add_", (torch.Tensor([1, 2, 3]),)),
]


@pytest.mark.xfail
@pytest.mark.parametrize("obj, op, args", test_args_inplace)
def test_inplace_chained_ops_gc_test(
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
    ptr = func(*args)  # inplace op

    # currently, inplace operators duplicate the object in store
    assert ptr.id_at_location != initial_id
    assert object_ptr.id_at_location == initial_id
    object_ptr.__del__()

    # even if we delete the first one, the duplicate remains
    assert len(node.store) == 1

    # this is currently considered a bug, but due to the fact that it mutates the original object
    # and it creates a duplicate one. This can only be replicated on objects that mutate inplace
    # and returns (chains) the result (like add_).
