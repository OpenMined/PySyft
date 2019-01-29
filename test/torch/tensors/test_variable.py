"""All the tests of things which are exclusively gradient focused. If
you are working on gradients being used by other abstractins, don't
 use this class. Use the abstractin's test class instead. (I.e., if you
 are testing gradients with PointerTensor, use test_pointer.py.)"""

import random

import torch
import syft as sy
from syft.frameworks.torch.tensors import LoggingTensor


class TestVariables(object):
    def setUp(self):
        hook = sy.TorchHook(torch, verbose=True)

        self.me = hook.local_worker
        self.me.is_client_worker = True

        instance_id = str(int(10e10 * random.random()))
        bob = sy.VirtualWorker(id=f"bob{instance_id}", hook=hook, is_client_worker=False)
        alice = sy.VirtualWorker(id=f"alice{instance_id}", hook=hook, is_client_worker=False)
        james = sy.VirtualWorker(id=f"james{instance_id}", hook=hook, is_client_worker=False)

        bob.add_workers([alice, james])
        alice.add_workers([bob, james])
        james.add_workers([bob, alice])

        self.hook = hook

        self.bob = bob
        self.alice = alice
        self.james = james

    def test_gradient_serde(self):

        self.setUp()

        # create a tensor
        x = torch.tensor([1, 2, 3, 4.], requires_grad=True)

        # create gradient on tensor
        x.sum().backward(torch.ones(1))

        # save gradient
        orig_grad = x.grad

        # serialize
        blob = sy.serde.serialize(x)

        # deserialize
        t = sy.serde.deserialize(blob)

        # check that gradient was properly serde
        assert (t.grad == orig_grad).all()