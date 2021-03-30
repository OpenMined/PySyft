# stdlib
import gc

# third party
import pytest
import torch as th

# syft absolute
import syft as sy


@pytest.mark.slow
def test_parameter_vm_remote_operation(
    node: sy.VirtualMachine, client: sy.VirtualMachineClient
) -> None:
    x = th.nn.Parameter(th.randn(3, 3))

    xp = x.send(client, pointable=False)

    y = xp + xp

    assert len(node.store._objects) == 2

    y.get()

    assert len(node.store._objects) == 1

    del xp

    gc.collect()

    assert len(node.store._objects) == 0


def test_get_copy(node: sy.VirtualMachine, client: sy.VirtualMachineClient) -> None:
    x = th.nn.Parameter(th.randn(3, 3))

    xp = x.send(client, pointable=False)

    y = xp + xp

    assert len(node.store._objects) == 2

    y.get_copy()

    # no deletion of the object
    assert len(node.store._objects) == 2

    del xp
    gc.collect()

    assert len(node.store._objects) == 1


def test_parameter_serde() -> None:
    param = th.nn.parameter.Parameter(th.tensor([1.0, 2, 3]), requires_grad=True)
    # Setting grad manually to check it is passed through serialization
    param.grad = th.randn_like(param)

    blob = sy.serialize(param)

    param2 = sy.deserialize(blob=blob)

    assert (param == param2).all()
    assert (param2.grad == param2.grad).all()
    assert param2.requires_grad == param2.requires_grad


def test_linear_grad_serde() -> None:
    # Parameter is created inside Linear module
    linear = th.nn.Linear(5, 1)
    param = linear.weight

    # we dont have auto generated IDs anymore
    # assert hasattr(param, "id")

    # TODO: fix backward is failing
    # Induce grads on linear weights
    # out = linear(th.randn(5, 5))
    # out.backward()
    # assert param.grad is not None

    blob = sy.serialize(param)

    param2 = sy.deserialize(blob=blob)

    assert (param == param2).all()
    # assert param == param2
    # assert (param2.grad == param2.grad).all()
    assert param2.requires_grad == param2.requires_grad
