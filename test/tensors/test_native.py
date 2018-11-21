from syft.frameworks.torch.hook import TorchHook
from syft.frameworks.torch.tensors import TorchTensor
from syft.frameworks.torch.tensors import PointerTensor
from syft.workers.virtual import VirtualWorker
from unittest import TestCase
import torch


class TestTorchTensor(TestCase):
    def test_owner_default(self):
        hook = TorchHook()
        tensor = TorchTensor()

        assert tensor.owner == hook.local_worker

    def test_create_pointer(self):
        hook = TorchHook()
        tensor = TorchTensor()
        bob = VirtualWorker()

        ptr = tensor.create_pointer(
            location=bob, id_at_location=1, register=False, owner=hook.local_worker, ptr_id=2
        )

        assert ptr.owner == hook.local_worker
        assert ptr.location == bob
        assert ptr.id_at_location == 1
        assert ptr.ptr_id == 2

    def test_create_pointer_defaults(self):
        tensor = TorchTensor()
        bob = VirtualWorker()

        ptr = tensor.create_pointer(location=bob)

        assert ptr.owner == tensor.owner
        assert ptr.location == bob

    def test_send(self):
        hook = TorchHook()  # noqa

        bob = VirtualWorker()

        tensor = torch.rand(5, 3)
        tensor.id = 1

        pointer = tensor.send(bob)

        assert type(pointer) == PointerTensor
        assert bob.get_obj(1) == tensor

    def test_get(self):
        hook = TorchHook()  # noqa

        bob = VirtualWorker()

        tensor = torch.rand(5, 3)
        tensor.id = 1

        pointer = tensor.send(bob)

        assert type(pointer) == PointerTensor
        assert pointer.get() == tensor
