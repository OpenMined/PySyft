from unittest import TestCase
import torch
import syft

from syft.frameworks.torch import TorchHook
from syft.workers import VirtualWorker


class TestHook(TestCase):
    def test___init__(self):
        hook = TorchHook(torch)
        assert torch.torch_hooked
        assert hook.torch.__version__ == torch.__version__

        alice = VirtualWorker(id="alice")
        hook = TorchHook(torch, local_worker=alice)

    def test_torch_attributes(self):
        with self.assertRaises(RuntimeError):
            syft.torch._command_guard("false_command", "torch_modules")

        assert syft.torch._is_command_valid_guard("torch.add", "torch_modules")
        assert not syft.torch._is_command_valid_guard("false_command", "torch_modules")

        syft.torch._command_guard("torch.add", "torch_modules", get_native=False)
