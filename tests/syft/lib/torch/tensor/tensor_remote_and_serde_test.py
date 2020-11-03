# stdlib
import gc

# third party
import pytest
import torch as th

# syft absolute
import syft as sy
from syft.core.node.common.service.auth import AuthorizationException


def test_torch_remote_tensor_register() -> None:
    """ Test if sending a tensor will be registered on the remote worker. """

    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    x = th.tensor([-1, 0, 1, 2, 3, 4])
    ptr = x.send(alice_client)

    assert len(alice.store) == 1

    ptr = x.send(alice_client)
    gc.collect()

    # the previous objects get deleted because we overwrite
    # ptr - we send a message to delete that object
    assert len(alice.store) == 1

    ptr.get()
    assert len(alice.store) == 0  # Get removes the object


def test_torch_remote_tensor_with_alias_send() -> None:
    """Test sending tensor on the remote worker with alias send method."""

    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    x = th.tensor([-1, 0, 1, 2, 3, 4])
    ptr = x.send_to(alice_client)

    assert len(alice.store) == 1

    # TODO: Fix this from deleting the object in the store due to the variable
    # see above
    # ptr = x.send_to(alice_client)

    data = ptr.get()

    assert len(alice.store) == 0  # Get removes the object

    assert x.equal(data)  # Check if send data and received data are equal


def test_torch_serde() -> None:

    x = th.tensor([1.0, 2, 3, 4], requires_grad=True)

    # This is not working currently:
    # (x * 2).sum().backward()
    # But pretend we have .grad
    x.grad = th.randn_like(x)

    blob = x.serialize()

    x2 = sy.deserialize(blob=blob)

    assert (x == x2).all()


def test_torch_no_read_permissions() -> None:

    bob = sy.VirtualMachine(name="bob")
    root_bob = bob.get_root_client()
    guest_bob = bob.get_client()

    x = th.tensor([1, 2, 3, 4])

    # root user of Bob's machine sends a tensor
    ptr = x.send(root_bob)

    # guest creates a pointer to that object (assuming the client can guess/infer the ID)
    ptr.client = guest_bob

    # this should trigger an exception
    with pytest.raises(AuthorizationException):
        ptr.get()

    x = th.tensor([1, 2, 3, 4])

    # root user of Bob's machine sends a tensor
    ptr = x.send(root_bob)

    # but if root bob asks for it it should be fine
    x2 = ptr.get()

    assert (x == x2).all()

    assert x.grad == x2.grad


def test_torch_garbage_collect() -> None:
    """
    Test if sending a tensor and then deleting the pointer removes the object
    from the remote worker.
    """

    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    x = th.tensor([-1, 0, 1, 2, 3, 4])
    ptr = x.send(alice_client)

    assert len(alice.store) == 1

    # "del" only decrements the counter and the garbage collector plays the role of the reaper
    del ptr

    # Make sure __del__ from Pointer is called
    gc.collect()

    assert len(alice.store) == 0


def test_torch_garbage_method_creates_pointer() -> None:
    """
    Test if sending a tensor and then deleting the pointer removes the object
    from the remote worker.
    """

    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    x = th.tensor([-1, 0, 1, 2, 3, 4])
    x_ptr = x.send(alice_client)

    assert len(alice.store) == 1

    gc.disable()
    x_ptr + 2

    assert len(alice.store) == 3
