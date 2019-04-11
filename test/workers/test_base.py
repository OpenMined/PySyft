import pytest
import time

import syft as sy
import torch as th

from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker


def test_create_already_existing_worker(hook):
    # Shares tensor with bob
    bob = sy.VirtualWorker(hook, "bob")
    x = th.tensor([1, 2, 3]).send(bob)

    # Recreates bob and shares a new tensor
    bob = sy.VirtualWorker(hook, "bob")
    y = th.tensor([2, 2, 2]).send(bob)

    # Recreates bob and shares a new tensor
    bob = sy.VirtualWorker(hook, "bob")
    z = th.tensor([2, 2, 10]).send(bob)

    # Both workers should be the same, so the following operation should be valid
    _ = x + y * z


def test_create_already_existing_worker_with_different_type(hook, start_proc):
    # Shares tensor with bob
    bob = sy.VirtualWorker(hook, "bob")
    _ = th.tensor([1, 2, 3]).send(bob)

    kwargs = {"id": "fed1", "host": "localhost", "port": 8765, "hook": hook}
    server = start_proc(WebsocketServerWorker, kwargs)

    time.sleep(0.1)

    # Recreates bob as a different type of worker
    kwargs = {"id": "bob", "host": "localhost", "port": 8765, "hook": hook}
    with pytest.raises(RuntimeError):
        bob = WebsocketClientWorker(**kwargs)

    server.terminate()
