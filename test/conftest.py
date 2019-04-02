import pytest
import torch
from multiprocessing import Process

import syft
from syft import TorchHook


@pytest.fixture()
def start_proc():  # pragma: no cover
    """ helper function for spinning up a websocket participant """

    def _start_proc(participant, kwargs):
        def target():
            server = participant(**kwargs)
            server.start()

        p = Process(target=target)
        p.start()
        return p

    return _start_proc


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

    output = {"me": hook.local_worker, "alice": alice, "bob": bob, "james": james}

    return output
