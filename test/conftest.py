import pytest
import torch

import syft
from syft import TorchHook


@pytest.fixture(scope="function", autouse=True)
def hook():
    hook = TorchHook(torch)
    return hook


@pytest.fixture(scope="function", autouse=True)
def workers(hook):
    alice = syft.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
    bob = syft.VirtualWorker(id="bob", hook=hook, is_client_worker=False)
    james = syft.VirtualWorker(id="james", hook=hook, is_client_worker=False)

    # TODO: should one set this boolean to true?
    # It was done previously in self.setUp() from `test_hook.py`
    # hook.local_worker.is_client_worker = True

    output = {}
    output["me"] = hook.local_worker
    output["alice"] = alice
    output["bob"] = bob
    output["james"] = james

    return output
