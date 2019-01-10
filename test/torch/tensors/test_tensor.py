import random

import torch
import syft

from syft.frameworks.torch.tensors import PointerTensor


class TestNative(object):
    def test_init(self):
        hook = syft.TorchHook(torch, verbose=True)
        tensor_extension = torch.Tensor()
        assert tensor_extension.id is not None
        assert tensor_extension.owner is not None


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

    def test_signature_cache_change(self):
        """Tests that calls to the same method using a different
        signature works correctly. We cache signatures in the
        hook.build_hook_args_function dictionary but sometimes they
        are incorrect if we use the same method with different
        parameter types. So, we need to test to make sure that
        this cache missing fails gracefully. This test tests
        that for the .div(tensor) .div(int) method."""

        self.setUp()

        x = torch.Tensor([1, 2, 3])
        y = torch.Tensor([1, 2, 3])

        z = x.div(y)
        z = x.div(2)
        z = x.div(y)

        assert True
