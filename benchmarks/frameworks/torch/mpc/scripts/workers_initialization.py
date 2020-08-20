import torch
import syft
from syft import TorchHook
from syft.generic.frameworks.hook import hook_args


def hook():
    hook = TorchHook(torch)
    return hook


def workers(hook):
    """
    This function defines virtual workers to be used in benchmarking functions.
    """

    # Reset the hook and the local worker
    syft.local_worker.clear_objects()
    hook_args.hook_method_args_functions = {}
    hook_args.hook_method_response_functions = {}
    hook_args.register_response_functions = {}
    hook_args.get_tensor_type_functions = {}

    # Define virtual workers
    alice = syft.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
    bob = syft.VirtualWorker(id="bob", hook=hook, is_client_worker=False)
    james = syft.VirtualWorker(id="james", hook=hook, is_client_worker=False)
    charlie = syft.VirtualWorker(id="charlie", hook=hook, is_client_worker=False)
    workers = {
        "me": hook.local_worker,
        "alice": alice,
        "bob": bob,
        "charlie": charlie,
        "james": james,
    }
    return workers
