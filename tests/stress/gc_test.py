# stdlib
import gc

# third party
import torch

# syft absolute
import syft as sy


def test_same_var_for_ptr_gc() -> None:
    """
    Test for checking if the gc is correctly triggered
    when the last reference to the ptr is overwritten
    """
    x = torch.tensor([1, 2, 3, 4])

    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    for _ in range(100):
        """
        Override the ptr multiple times to make sure we trigger
        the gc
        """
        ptr = x.send(alice_client)

    gc.collect()

    assert len(alice.store) == 1

    ptr.get()
    gc.collect()

    assert len(alice.store) == 0


def test_send_same_obj_gc() -> None:
    """
    Test if sending the same object multiple times, register the
    object multiple times - because each send generated another
    remote object
    """

    x = torch.tensor([1, 2, 3, 4])
    ptr = []

    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    for _ in range(100):
        ptr.append(x.send(alice_client))

    gc.collect()
    assert len(alice.store) == 100

    del ptr

    gc.collect()
    assert len(alice.store) == 0
