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
    # And 1 plan worker
    plan_worker = syft.Plan(id="plan_worker", hook=hook, is_client_worker=False)

    bob.add_workers([alice, james, plan_worker])
    alice.add_workers([bob, james, plan_worker])
    james.add_workers([bob, alice, plan_worker])
    plan_worker.add_workers([alice, bob, james])
    hook.local_worker.add_workers([alice, bob, james, plan_worker])

    output = {
        "me": hook.local_worker,
        "alice": alice,
        "bob": bob,
        "james": james,
        "plan_worker": plan_worker,
    }

    return output
