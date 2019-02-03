import random

import pytest
import torch
import torch.nn.functional as F
import syft

from syft.frameworks.torch.tensors import AdditiveSharingTensor


class TestAdditiveSharingTensor(object):
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
        Test the .on() wrap functionality for LoggingTensor
        """
        self.setUp()
        x_tensor = torch.Tensor([1, 2, 3])
        x = AdditiveSharingTensor().on(x_tensor)
        assert isinstance(x, torch.Tensor)
        assert isinstance(x.child, AdditiveSharingTensor)
        assert isinstance(x.child.child, torch.Tensor)

    def test_encode_decode(self):

        self.setUp()

        x = torch.tensor([1,2,3]).share(self.bob, self.alice, self.james)

        x = x.get()

        assert x[0] == 1

    def test_add(self):

        self.setUp()

        x = torch.tensor([1, 2, 3]).share(self.bob, self.alice, self.james)

        y = (x + x).get()

        assert y[0] == 2

