import pytest
import torch
import syft

from syft.frameworks.torch import TorchHook
from syft.workers import VirtualWorker


def test___init__(hook):
    assert torch.torch_hooked
    assert hook.torch.__version__ == torch.__version__


def test_torch_attributes():
    with pytest.raises(RuntimeError):
        syft.torch._command_guard("false_command", "torch_modules")

    assert syft.torch._is_command_valid_guard("torch.add", "torch_modules")
    assert not syft.torch._is_command_valid_guard("false_command", "torch_modules")

    syft.torch._command_guard("torch.add", "torch_modules", get_native=False)


def test_worker_registration():
    hook = syft.TorchHook(torch, verbose=True)

    me = hook.local_worker
    me.is_client_worker = False

    bob = syft.VirtualWorker(id="bob", hook=hook, is_client_worker=False)

    me.add_workers([bob])

    worker = me.get_worker(bob)

    assert bob == worker
