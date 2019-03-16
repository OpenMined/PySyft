import pytest
import torch

import syft
from syft import TorchHook


@pytest.fixture(scope="session", autouse=True)
def hook():
    hook = TorchHook(torch)
    return hook


@pytest.fixture(scope="session", autouse=True)
def workers(hook):
    # Define 3 virtual workers
    alice = syft.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
    bob = syft.VirtualWorker(id="bob", hook=hook, is_client_worker=False)
    james = syft.VirtualWorker(id="james", hook=hook, is_client_worker=False)

    # bob.add_workers([alice, james])
    # alice.add_workers([bob, james])
    # james.add_workers([bob, alice])
    # hook.local_worker.add_workers([alice, bob, james])

    output = {"me": hook.local_worker, "alice": alice, "bob": bob, "james": james}

    return output
