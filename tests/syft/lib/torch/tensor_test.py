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

    x = th.tensor([1, 2, 3, 4])

    blob = x.serialize()

    x2 = sy.deserialize(blob=blob)

    assert (x == x2).all()