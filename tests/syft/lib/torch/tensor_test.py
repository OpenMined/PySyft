import syft as sy
import torch as th
import pytest


def test_torch_vm_remote_operation():

    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    x = th.tensor([1, 2, 3, 4])

    xp = x.send(alice_client)

    y = xp + xp

    assert len(alice.store._objects) == 2

    y.get()

    assert len(alice.store._objects) == 1

    # TODO: put thought into garbage collection and then
    #  uncoment this.
    # del xp
    #
    # assert len(alice.store._objects) == 0


def test_torch_serde():

    x = th.tensor([1.0, 2, 3, 4], requires_grad=True)

    # This is not working currently:
    # (x * 2).sum().backward()
    # But pretend we have .grad
    x.grad = th.randn_like(x)

    blob = x.serialize()

    x2 = sy.deserialize(blob=blob)

    assert (x == x2).all()


def test_torch_permissions():

    bob = sy.VirtualMachine(name="bob")
    root_bob = bob.get_root_client()
    guest_bob = bob.get_client()

    import torch as th

    x = th.tensor([1, 2, 3, 4])

    # root user of Bob's machine sends a tensor
    ptr = x.send(root_bob)

    # guest bob creates a pointer to that object (assuming he could guess/inpher the ID)
    ptr.location = guest_bob

    # this should trigger an exception
    with pytest.raises(Exception):
        ptr.get()

    x = th.tensor([1, 2, 3, 4])

    # root user of Bob's machine sends a tensor
    ptr = x.send(root_bob)

    # but if root bob asks for it it should be fine
    x2 = ptr.get()

    assert (x == x2).all()

    assert (x.grad == x2.grad)
