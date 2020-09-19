import pytest
import torch

import syft
from syft import TorchHook
from syft.generic.frameworks.hook import hook_args


@pytest.fixture(scope="session", autouse=True)
def hook():
    hook = TorchHook(torch)
    return hook


@pytest.fixture(scope="function", autouse=True)
def workers(hook):
    # To run a plan locally the local worker can't be a client worker,
    # since it needs to register objects
    # LaRiffle edit: doing this increases the reference count on pointers and
    # breaks the auto garbage collection for pointer of pointers, see #2150
    # hook.local_worker.is_client_worker = False

    # Reset the hook and the local worker
    syft.local_worker.clear_objects()
    hook_args.hook_method_args_functions = {}
    hook_args.hook_method_response_functions = {}
    hook_args.register_response_functions = {}
    hook_args.get_tensor_type_functions = {}

    # Define 4 virtual workers
    alice = syft.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
    bob = syft.VirtualWorker(id="bob", hook=hook, is_client_worker=False)
    charlie = syft.VirtualWorker(id="charlie", hook=hook, is_client_worker=False)
    james = syft.VirtualWorker(id="james", hook=hook, is_client_worker=False)

    workers = {
        "me": hook.local_worker,
        "alice": alice,
        "bob": bob,
        "charlie": charlie,
        "james": james,
    }

    yield workers

    alice.remove_worker_from_local_worker_registry()
    bob.remove_worker_from_local_worker_registry()
    charlie.remove_worker_from_local_worker_registry()
    james.remove_worker_from_local_worker_registry()
