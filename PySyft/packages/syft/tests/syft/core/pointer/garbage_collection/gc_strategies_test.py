# third party
import torch

# syft absolute
import syft as sy
from syft.core.pointer.garbage_collection import GCBatched
from syft.core.pointer.garbage_collection import GCSimple
from syft.core.pointer.garbage_collection import GarbageCollection
from syft.core.pointer.garbage_collection import gc_get_default_strategy
from syft.core.pointer.garbage_collection import gc_set_default_strategy


def test_gc_simple_strategy(node: sy.VirtualMachine) -> None:
    client = node.get_client()

    x = torch.tensor([1, 2, 3, 4])
    ptr = x.send(client, pointable=False)

    assert len(node.store) == 1

    del ptr

    assert len(node.store) == 0


def test_gc_batched_strategy_setter(node: sy.VirtualMachine) -> None:

    client = node.get_client()
    client.gc.gc_strategy = GCBatched(threshold=10)

    x = torch.tensor([1, 2, 3, 4])

    for _ in range(9):
        x.send(client, pointable=False)

    assert len(node.store) == 9

    x.send(client, pointable=False)

    assert len(node.store) == 0


def test_gc_batched_strategy_gc_constructor(node: sy.VirtualMachine) -> None:
    client = node.get_client()
    client.gc = GarbageCollection("gcbatched", 5)

    x = torch.tensor([1, 2, 3, 4])

    for _ in range(4):
        x.send(client, pointable=False)

    assert len(node.store) == 4

    x.send(client, pointable=False)

    assert len(node.store) == 0


def test_gc_change_default_gc_strategy(node: sy.VirtualMachine) -> None:
    gc_prev_strategy = gc_get_default_strategy()
    gc_set_default_strategy("gcbatched")

    client = node.get_client()

    res = isinstance(client.gc.gc_strategy, GCBatched)

    # Revert
    gc_set_default_strategy(gc_prev_strategy)
    sy.core.pointer.garbage_collection.GC_DEFAULT_STRATEGY = GCSimple

    assert res


def test_gc_batched_delete_at_change(node: sy.VirtualMachine) -> None:
    client = node.get_client()

    # Change the strategy
    client.gc.gc_strategy = GCBatched()

    x = torch.tensor([1, 2, 3, 4])

    x.send(client, pointable=False)
    x.send(client, pointable=False)
    x.send(client, pointable=False)

    assert len(node.store) == 3

    # It should for the GCBatched to delete all the cached to-delete objs
    client.gc.gc_strategy = GCSimple()

    assert len(node.store) == 0
