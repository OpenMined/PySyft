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
