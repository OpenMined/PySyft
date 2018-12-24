import random

from syft.frameworks.torch.tensors import PointerTensor
from syft.frameworks.torch.hook import TorchHook
from syft.workers.virtual import VirtualWorker
from unittest import TestCase
import torch


def test_send_tensor(workers):
    x = torch.Tensor([1, 2])
    ptr_id = int(10e10 * random.random())
    x_ptr = workers["me"].send(x, workers["bob"], ptr_id)

    assert isinstance(x_ptr.child, PointerTensor)
    assert x_ptr.location.id == workers["bob"].id
    assert x_ptr.id_at_location == ptr_id
    remote_x = workers["bob"].get_obj(ptr_id)
    assert remote_x is not None
    assert isinstance(remote_x, torch.Tensor)

    x_back = x_ptr.get()

    assert (x == x_back).all()
