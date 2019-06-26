import pytest
import torch
from multiprocessing import Process
import builtins

import syft
from syft import TorchHook


@pytest.fixture()
def start_proc():  # pragma: no cover
    """ helper function for spinning up a websocket participant """

    def _start_proc(participant, dataset: str = None, **kwargs):
        def target():
            server = participant(**kwargs)
            if dataset is not None:
                data, key = dataset
                server.add_dataset(data, key=key)
            server.start()

        p = Process(target=target)
        p.start()
        return p

    return _start_proc


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
    syft.frameworks.torch.hook.hook_args.hook_method_args_functions = {}
    syft.frameworks.torch.hook.hook_args.hook_method_response_functions = {}
    syft.frameworks.torch.hook.hook_args.get_tensor_type_functions = {}

    # Define 3 virtual workers
    alice = syft.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
    bob = syft.VirtualWorker(id="bob", hook=hook, is_client_worker=False)
    james = syft.VirtualWorker(id="james", hook=hook, is_client_worker=False)

    workers = {"me": hook.local_worker, "alice": alice, "bob": bob, "james": james}

    yield workers

    alice.remove_worker_from_local_worker_registry()
    bob.remove_worker_from_local_worker_registry()
    james.remove_worker_from_local_worker_registry()


@pytest.fixture
def no_tf_encrypted():
    import_orig = builtins.__import__

    def mocked_import(name, globals, locals, fromlist, level):
        if "tf_encrypted" in name:
            raise ImportError()
        return import_orig(name, globals, locals, fromlist, level)

    builtins.__import__ = mocked_import
    yield
    builtins.__import__ = import_orig
