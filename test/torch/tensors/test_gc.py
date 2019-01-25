"""All the tests relative to garbage collection of all kinds of remote or local tensors"""

import random

import torch
import syft
from syft.frameworks.torch.tensors import LoggingTensor


class TestGarbageCollection(object):
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

    # POINTERS

    def test_explicit_garbage_collect_pointer(self):
        """Tests whether deleting a PointerTensor garbage collects the remote object too"""

        self.setUp()

        # create tensor
        x = torch.Tensor([1, 2])

        # send tensor to bob
        x_ptr = x.send(self.bob)

        # ensure bob has tensor
        assert x.id in self.bob._objects

        # delete pointer to tensor, which should
        # automatically garbage collect the remote
        # object on Bob's machine
        del x_ptr

        # ensure bob's object was garbage collected
        assert x.id not in self.bob._objects

    def test_explicit_garbage_collect_double_pointer(self):
        """Tests whether deleting a pointer to a pointer garbage collects
        the remote object too"""

        self.setUp()

        # create tensor
        x = torch.Tensor([1, 2])

        # send tensor to bob and then pointer to alice
        x_ptr = x.send(self.bob)
        x_ptr_ptr = x_ptr.send(self.alice)

        # ensure bob has tensor
        assert x.id in self.bob._objects

        # delete pointer to pointer to tensor, which should automatically
        # garbage collect the remote object on Bob's machine
        del x_ptr_ptr

        # ensure bob's object was garbage collected
        assert x.id not in self.bob._objects

        # Chained version
        x = torch.Tensor([1, 2])
        x_id = x.id
        # send tensor to bob and then pointer to alice
        x = x.send(self.bob).send(self.alice)
        # ensure bob has tensor
        assert x_id in self.bob._objects
        # delete pointer to pointer to tensor
        del x
        # ensure bob's object was garbage collected
        assert x_id not in self.bob._objects

    def test_implicit_garbage_collection_pointer(self):
        """Tests whether GCing a PointerTensor GCs the remote object too."""

        self.setUp()

        # create tensor
        x = torch.Tensor([1, 2])

        # send tensor to bob
        x_ptr = x.send(self.bob)

        # ensure bob has tensor
        assert x.id in self.bob._objects

        # delete pointer to tensor, which should
        # automatically garbage collect the remote
        # object on Bob's machine
        x_ptr = "asdf"

        # ensure bob's object was garbage collected
        assert x.id not in self.bob._objects

    def test_implicit_garbage_collect_double_pointer(self):
        """Tests whether GCing a pointer to a pointer garbage collects
        the remote object too"""

        self.setUp()

        # create tensor
        x = torch.Tensor([1, 2])

        # send tensor to bob and then pointer to alice
        x_ptr = x.send(self.bob)
        x_ptr_ptr = x_ptr.send(self.alice)

        # ensure bob has tensor
        assert x.id in self.bob._objects

        # delete pointer to pointer to tensor, which should automatically
        # garbage collect the remote object on Bob's machine
        x_ptr_ptr = "asdf"

        # ensure bob's object was garbage collected
        assert x.id not in self.bob._objects

        # Chained version
        x = torch.Tensor([1, 2])
        x_id = x.id
        # send tensor to bob and then pointer to alice
        x = x.send(self.bob).send(self.alice)
        # ensure bob has tensor
        assert x_id in self.bob._objects
        # delete pointer to pointer to tensor
        x = "asdf"
        # ensure bob's object was garbage collected
        assert x_id not in self.bob._objects

    # LOGGING TENSORS

    def test_explicit_garbage_collect_logging_on_pointer(self):
        """
        Tests whether deleting a LoggingTensor on a PointerTensor
        garbage collects the remote object too
        """
        self.setUp()

        x = torch.Tensor([1, 2])
        x_id = x.id

        x = x.send(self.bob)
        x = LoggingTensor().on(x)
        assert x_id in self.bob._objects

        del x

        assert x_id not in self.bob._objects

    def test_implicit_garbage_collect_logging_on_pointer(self):
        """
        Tests whether GCing a LoggingTensor on a PointerTensor
        garbage collects the remote object too
        """
        self.setUp()

        x = torch.Tensor([1, 2])
        x_id = x.id

        x = x.send(self.bob)
        x = LoggingTensor().on(x)
        assert x_id in self.bob._objects

        x = "open-source"

        assert x_id not in self.bob._objects
