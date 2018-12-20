from syft.frameworks.torch.hook import TorchHook
from syft.workers.virtual import VirtualWorker
from unittest import TestCase
import torch


class TestTorchTensor(TestCase):
    def setUp(self):
        self.hook = TorchHook(torch)
        self.bob = VirtualWorker()

    def test_overload_reshape(self):
        tensor = torch.Tensor([1, 2, 3, 4])
        tensor_reshaped = tensor.reshape((2, 2))
        tensor_matrix = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert (tensor_reshaped == tensor_matrix).all()

    def test_owner_default(self):
        tensor = torch.Tensor([1, 2, 3, 4, 5])

        assert tensor.owner == self.hook.local_worker

    def test_create_pointer(self):
        tensor = torch.Tensor([1, 2, 3, 4, 5])

        ptr = tensor.create_pointer(
            location=self.bob,
            id_at_location=1,
            register=False,
            owner=self.hook.local_worker,
            ptr_id=2,
        )

        assert ptr.owner == self.hook.local_worker
        assert ptr.location == self.bob
        assert ptr.id_at_location == 1
        assert ptr.id == 2

        ptr2 = tensor.create_pointer(owner=self.hook.local_worker)
        assert isinstance(ptr2.__str__(), str)
        assert isinstance(ptr2.__repr__(), str)

    def test_create_pointer_defaults(self):
        tensor = torch.Tensor([1, 2, 3, 4, 5])

        ptr = tensor.create_pointer(location=self.bob)

        assert ptr.owner == tensor.owner
        assert ptr.location == self.bob

    # def test_get(self):
    #     tensor = torch.rand(5, 3)
    #     tensor.id = 1

    #     pointer = tensor.send(self.bob)

    #     assert type(pointer) == PointerTensor
    #     assert pointer.get() == tensor
