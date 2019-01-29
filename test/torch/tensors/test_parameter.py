import random

import torch
import torch.nn.functional as F
from torch.nn import Parameter
import syft

from syft.frameworks.torch.tensors import LoggingTensor


class TestLogTensor(object):
    def setUp(self):
        hook = syft.TorchHook(torch, verbose=True)

        self.me = hook.local_worker
        self.me.is_client_worker = True

        instance_id = str(int(10e10 * random.random()))
        bob = syft.VirtualWorker(id=f"bob{instance_id}", hook=hook, is_client_worker=False)
        alice = syft.VirtualWorker(id=f"alice{instance_id}", hook=hook, is_client_worker=False)
        james = syft.VirtualWorker(id=f"james{instance_id}", hook=hook, is_client_worker=False)

        bob.add_workers([alice, james])
        alice.add_workers([bob, james])
        james.add_workers([bob, alice])

        self.hook = hook

        self.bob = bob
        self.alice = alice
        self.james = james

    def test_param_on_pointer(self):
        """
        """
        self.setUp()
        tensor = torch.tensor([1.0, -1.0, 3.0, 4.0])
        ptr = tensor.send(self.bob)
        param = Parameter(ptr)
        local_param = param.get()
        assert (local_param.data == tensor).all()

    def test_param_send_get(self):
        """
        """
        self.setUp()
        tensor = torch.tensor([1.0, -1.0, 3.0, 4.0])
        param = Parameter(data=tensor)
        param_ptr = param.send(self.bob)
        param_back = param_ptr.get()
        assert (param_back.data == tensor).all()

    def test_param_remote_binary_method(self):
        """
        """
        self.setUp()
        tensor = torch.tensor([1.0, -1.0, 3.0, 4.0])
        param = Parameter(data=tensor)
        param_ptr = param.send(self.bob)
        param_double_ptr = param_ptr + param_ptr
        param_double_back = param_double_ptr.get()
        double_tensor = tensor + tensor
        assert (param_double_back.data == double_tensor).all()
