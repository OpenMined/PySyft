import random

import torch
import syft

from syft.frameworks.torch.tensors import LogTensor


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

    def test_wrap(self):
        """
        Test the .on() wrap functionality for LogTensor
        """
        self.setUp()
        x_tensor = torch.Tensor([1, 2, 3])
        x = LogTensor().on(x_tensor)
        assert isinstance(x, torch.Tensor)
        assert isinstance(x.child, LogTensor)
        assert isinstance(x.child.child, torch.Tensor)

    def test_method_on_log_chain(self):
        """
        Test method call on a chain including a log tensor
        """
        self.setUp()
        # build a long chain tensor Wrapper>LogTensor>TorchTensor
        x_tensor = torch.Tensor([1, 2, 3])
        x = LogTensor().on(x_tensor)
        y = x.add(x)
        assert (y.child.child == x_tensor.add(x_tensor)).all()
