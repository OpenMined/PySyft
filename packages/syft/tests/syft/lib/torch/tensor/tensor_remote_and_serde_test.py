# stdlib
import gc

# third party
import pytest
import torch as th

# syft absolute
import syft as sy
from syft.core.node.common.service.auth import AuthorizationException


@pytest.mark.slow
def test_torch_remote_tensor_register(
    node: sy.VirtualMachine, client: sy.VirtualMachineClient
) -> None:
    """Test if sending a tensor will be registered on the remote worker."""
    x = th.tensor([-1, 0, 1, 2, 3, 4])
    ptr = x.send(client, pointable=False)

    assert len(node.store) == 1

    ptr = x.send(client, pointable=False)
    gc.collect()

    # the previous objects get deleted because we overwrite
    # ptr - we send a message to delete that object
    assert len(node.store) == 1

    ptr.get()
    assert len(node.store) == 0  # Get removes the object


def test_torch_remote_tensor_with_send(
    node: sy.VirtualMachine, client: sy.VirtualMachineClient
) -> None:
    """Test sending tensor on the remote worker with send method."""

    x = th.tensor([-1, 0, 1, 2, 3, 4])
    ptr = x.send(client)

    assert len(node.store) == 1

    data = ptr.get()

    assert len(node.store) == 0  # Get removes the object

    assert x.equal(data)  # Check if send data and received data are equal


def test_torch_serde() -> None:

    x = th.tensor([1.0, 2, 3, 4], requires_grad=True)

    # This is not working currently:
    # (x * 2).sum().backward()
    # But pretend we have .grad
    x.grad = th.randn_like(x)

    blob = sy.serialize(x)

    x2 = sy.deserialize(blob=blob)

    assert (x == x2).all()


@pytest.mark.slow
def test_torch_no_read_permissions(
    client: sy.VirtualMachineClient, root_client: sy.VirtualMachineClient
) -> None:

    x = th.tensor([1, 2, 3, 4])

    # root user of Bob's machine sends a tensor
    ptr = x.send(root_client)

    # guest creates a pointer to that object (assuming the client can guess/infer the ID)
    ptr.client = client

    # this should trigger an exception
    with pytest.raises(AuthorizationException):
        ptr.get()

    x = th.tensor([1, 2, 3, 4])

    # root user of Bob's machine sends a tensor
    ptr = x.send(root_client)

    # but if root bob asks for it it should be fine
    x2 = ptr.get()

    assert (x == x2).all()

    assert x.grad == x2.grad


def test_torch_garbage_collect(
    node: sy.VirtualMachine, client: sy.VirtualMachineClient
) -> None:
    """
    Test if sending a tensor and then deleting the pointer removes the object
    from the remote worker.
    """
    x = th.tensor([-1, 0, 1, 2, 3, 4])
    ptr = x.send(client, pointable=False)

    assert len(node.store) == 1

    # "del" only decrements the counter and the garbage collector plays the role of the reaper
    del ptr

    # Make sure __del__ from Pointer is called
    gc.collect()

    assert len(node.store) == 0
