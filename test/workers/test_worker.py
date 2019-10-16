import pytest
import torch

import syft as sy
from syft.workers.virtual import VirtualWorker

from syft.exceptions import WorkerNotFoundException


def test___init__():
    hook = sy.TorchHook(torch)

    tensor = torch.tensor([1, 2, 3, 4])

    worker_id = sy.ID_PROVIDER.pop()
    alice_id = f"alice{worker_id}"
    alice = VirtualWorker(hook, id=alice_id)
    worker_id = sy.ID_PROVIDER.pop()
    bob = VirtualWorker(hook, id=f"bob{worker_id}")
    worker_id = sy.ID_PROVIDER.pop()
    charlie = VirtualWorker(hook, id=f"charlie{worker_id}")
    worker_id = sy.ID_PROVIDER.pop()
    dawson = VirtualWorker(hook, id=f"dawson{worker_id}", data=[tensor])

    # Ensure adding data on signup functionality works as expected
    assert tensor.owner == dawson

    assert bob.get_worker(alice_id).id == alice.id
    assert bob.get_worker(alice).id == alice.id
    assert bob.get_worker(charlie).id == charlie.id

    bob.get_worker("the_unknown_worker")

    bob.add_worker(alice)


def test_get_unknown_worker():

    hook = sy.TorchHook(torch)

    bob = VirtualWorker(hook, id="bob")
    charlie = VirtualWorker(hook, id="charlie")

    # if an unknown string or id representing a worker is given it fails
    with pytest.raises(WorkerNotFoundException):
        bob.get_worker("the_unknown_worker", fail_hard=True)

    with pytest.raises(WorkerNotFoundException):
        bob.get_worker(1, fail_hard=True)

    # if an instance of virtual worker is given it doesn't fail
    assert bob.get_worker(charlie).id == charlie.id
    assert charlie.id in bob._known_workers
