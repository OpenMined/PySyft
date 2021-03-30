# stdlib
from time import sleep

# third party
import torch

# syft absolute
import syft as sy
from syft.core.garbage_collection import GCBatched
from syft.core.garbage_collection import GCSimple
from syft.core.garbage_collection import gc_get_default_strategy
from syft.core.garbage_collection import gc_set_default_strategy


def test_gc_simple_strategy(node: sy.VirtualMachine) -> None:

    client = node.get_client()

    x = torch.tensor([1, 2, 3, 4])
    ptr = x.send(client, pointable=False)

    assert len(node.store) == 1

    del ptr

    assert len(node.store) == 0


def test_gc_batched_strategy_per_client(node: sy.VirtualMachine) -> None:

    client = node.get_client()
    client.gc.gc_strategy = GCBatched(threshold_client=10)

    x = torch.tensor([1, 2, 3, 4])

    for _ in range(9):
        x.send(client, pointable=False)

    assert len(node.store) == 9

    x.send(client, pointable=False)

    assert len(node.store) == 0


def test_gc_batched_strategy_per_total() -> None:
    node1 = sy.VirtualMachine(name="Alice")
    node2 = sy.VirtualMachine(name="Bob")

    client1 = node1.get_client()
    client2 = node2.get_client()

    # We need to set the same instance since we should shar
    # the client_to_obj_ids
    gc_strategy = GCBatched(threshold_client=100, threshold_total=5)
    client1.gc.gc_strategy = gc_strategy
    client2.gc.gc_strategy = gc_strategy

    x = torch.tensor([1, 2, 3, 4])

    x.send(client1, pointable=False)
    x.send(client2, pointable=False)
    x.send(client2, pointable=False)
    x.send(client1, pointable=False)

    assert len(node1.store) == 2
    assert len(node2.store) == 2

    x.send(client1, pointable=False)

    assert len(node2.store) == 0
    assert len(node1.store) == 0


def test_gc_change_default_gc_strategy(node: sy.VirtualMachine) -> None:
    gc_prev_strategy = gc_get_default_strategy()
    gc_set_default_strategy(GCBatched)

    client = node.get_client()

    res = isinstance(client.gc.gc_strategy, GCBatched)
    print(client.gc.gc_strategy)

    # Revert
    gc_set_default_strategy(gc_prev_strategy)
    sy.core.garbage_collection.GC_DEFAULT_STRATEGY = GCSimple

    assert res


def test_gc_batched_force_gc_at_change(node: sy.VirtualMachine) -> None:
    client = node.get_client()

    # Change the strategy
    gc_strategy = GCBatched()
    client.gc.gc_strategy = gc_strategy

    x = torch.tensor([1, 2, 3, 4])

    x.send(client, pointable=False)
    x.send(client, pointable=False)
    x.send(client, pointable=False)

    assert len(node.store) == 3

    # It should for the GCBatched to delete all the cached to-delete objs
    client.gc.gc_strategy = GCSimple()

    assert len(node.store) == 0
