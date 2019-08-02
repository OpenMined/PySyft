import syft as sy
import torch as th


def test_is_client_true(hook):
    me = hook.local_worker
    me.is_client_worker = True
    x = th.tensor([1, 2, 3])
    assert x.id not in me._objects


def test_is_client_false(hook):
    me = hook.local_worker
    me.is_client_worker = False
    x = th.tensor([1, 2, 3])
    assert x.id in me._objects


def test_in_known_workers(hook):
    # Get local worker
    local_worker = hook.local_worker
    id = local_worker.id

    # Get known workers dict
    known_workers = local_worker._known_workers

    assert id in known_workers and local_worker == known_workers[id]
