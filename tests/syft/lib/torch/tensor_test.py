import syft as sy
import torch as th


def test_torch_vm_remote_operation():

    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    x = th.tensor([1, 2, 3, 4])

    xp = x.send(alice_client)

    y = xp + xp

    assert len(alice.store._objects) == 2

    y.get()

    assert len(alice.store._objects) == 1

    del xp

    assert len(alice.store._objects) == 0


def test_torch_serde():

    x = th.tensor([1.0, 2, 3, 4], requires_grad=True)

    # This is not working currently:
    # (x * 2).sum().backward()
    # But pretend we have .grad
    x.grad = th.randn_like(x)

    blob = x.serialize()

    x2 = sy.deserialize(blob=blob)

    assert (x == x2).all()
    assert (x.grad == x2.grad).all()
