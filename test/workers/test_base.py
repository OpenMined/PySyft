import random

from syft.frameworks.torch.tensors import PointerTensor
from syft.frameworks.torch.hook import TorchHook
from syft.workers.virtual import VirtualWorker
from unittest import TestCase
import torch


class TestTorchTensor(TestCase):
    def setUp(self):
        self.hook = TorchHook(torch)
        self.me = self.hook.local_worker
        self.bob = VirtualWorker()

    def test_send_tensor(self):
        x = torch.Tensor([1, 2])
        ptr_id = int(10e10 * random.random())
        x_ptr = self.me.send(x, self.bob, ptr_id)

        assert isinstance(x_ptr.child, PointerTensor)
        assert x_ptr.location.id == self.bob.id
        assert x_ptr.id_at_location == ptr_id
        remote_x = self.bob.get_obj(ptr_id)
        assert remote_x is not None
        assert isinstance(remote_x, torch.Tensor)

        x_back = x_ptr.get()

        assert (x == x_back).all()
