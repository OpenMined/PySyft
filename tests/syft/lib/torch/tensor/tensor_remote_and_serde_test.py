# third party
import pytest
import torch as th

# syft absolute
import syft as sy


def test_torch_remote_tensor_register() -> None:
    """ Test if sending a tensor will be registered on the remote worker. """

    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    x = th.tensor([-1, 0, 1, 2, 3, 4])
    ptr = x.send(alice_client)

    assert len(alice.store) == 1

    ptr = x.send(alice_client)
    assert len(alice.store) == 1  # Same id

    ptr.get()
    assert len(alice.store) == 0  # Get removes the object


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

    # guest creates a pointer to that object (assuming the client can guess/inpher the ID)
    ptr.client = guest_bob

    # this should trigger an exception
    with pytest.raises(Exception) as exception:
        ptr.get()

    assert (
        str(exception.value)
        == "You do not have permission to .get() this tensor. Please submit a request."
    )

    x = th.tensor([1, 2, 3, 4])

    # root user of Bob's machine sends a tensor
    ptr = x.send(root_bob)

    # but if root bob asks for it it should be fine
    x2 = ptr.get()

    assert (x == x2).all()

    assert x.grad == x2.grad
