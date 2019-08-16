import time
import pytest
import torch
from multiprocessing import Process
import builtins

import syft
from syft import TorchHook
from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker


def _start_proc(participant, dataset: str = None, **kwargs):
    """Helper function for spinning up a websocket participant."""

    def target():
        server = participant(**kwargs)
        if dataset is not None:
            data, key = dataset
            server.add_dataset(data, key=key)
        server.start()

    p = Process(target=target)
    p.start()
    return p


@pytest.fixture()
def start_proc():  # pragma: no cover
    return _start_proc


@pytest.fixture()
def start_remote_worker():  # pragma: no cover
    """Helper function for starting a websocket worker."""

    def _start_remote_worker(
        id, hook, dataset: str = None, host="localhost", port=8768, max_tries=5, sleep_time=0.01
    ):
        kwargs = {"id": id, "host": host, "port": port, "hook": hook}
        server = _start_proc(WebsocketServerWorker, dataset=dataset, **kwargs)

        retry_counter = 0
        connection_open = False
        while not connection_open:
            try:
                remote_worker = WebsocketClientWorker(**kwargs)
                connection_open = True
            except ConnectionRefusedError as e:
                if retry_counter < max_tries:
                    retry_counter += 1
                    time.sleep(sleep_time)
                else:
                    raise e

        return server, remote_worker

    return _start_remote_worker


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
