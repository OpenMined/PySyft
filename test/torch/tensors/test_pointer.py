import random

import torch
import syft

from syft.frameworks.torch.tensors import PointerTensor


class TestPointer(object):
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

    def test_init(self):
        self.setUp()
        pointer = PointerTensor(id=1000, location=self.alice, owner=self.me)
        pointer.__str__()

    def test_create_pointer(self):
        self.setUp()
        x = torch.Tensor([1, 2])
        x.create_pointer()
        x.create_pointer(location=self.james)

    def test_send_get(self):
        """Test several send get usages"""
        self.setUp()
        bob = self.bob
        alice = self.alice

        # simple send
        x = torch.Tensor([1, 2])
        x_ptr = x.send(bob)
        x_back = x_ptr.get()
        assert (x == x_back).all()

        # send with variable overwriting
        x = torch.Tensor([1, 2])
        x = x.send(bob)
        x_back = x.get()
        assert (torch.Tensor([1, 2]) == x_back).all()

        # double send
        x = torch.Tensor([1, 2])
        x_ptr = x.send(bob)
        x_ptr_ptr = x_ptr.send(alice)
        x_ptr_back = x_ptr_ptr.get()
        x_back_back = x_ptr_back.get()
        assert (x == x_back_back).all()

        # double send with variable overwriting
        x = torch.Tensor([1, 2])
        x = x.send(bob)
        x = x.send(alice)
        x = x.get()
        x_back = x.get()
        assert (torch.Tensor([1, 2]) == x_back).all()

        # chained double send
        x = torch.Tensor([1, 2])
        x = x.send(bob).send(alice)
        x_back = x.get().get()
        assert (torch.Tensor([1, 2]) == x_back).all()

    def test_repeated_send(self):
        """Tests that repeated calls to .send(bob) works gracefully
        Previously garbage collection deleted the remote object
        when .send() was called twice. This test ensures the fix still
        works."""

        self.setUp()

        # create tensor
        x = torch.Tensor([1, 2])
        print(x.id)

        # send tensor to bob
        x_ptr = x.send(self.bob)

        # send tensor again
        x_ptr = x.send(self.bob)

        # ensure bob has tensor
        assert x.id in self.bob._objects
