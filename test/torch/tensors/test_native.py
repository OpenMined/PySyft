import torch
import syft
import random

from syft.frameworks.torch.tensors import PointerTensor


class TestNativeTensor(object):
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

    def test___str__(self):
        self.setUp()
        tensor = torch.Tensor([1, 2, 3, 4])
        assert isinstance(tensor.__str__(), str)
        tensor_ptr = tensor.send(self.bob)
        assert isinstance(tensor_ptr.__str__(), str)

    def test___repr__(self):
        self.setUp()
        tensor = torch.Tensor([1, 2, 3, 4])
        assert isinstance(tensor.__repr__(), str)
        tensor_ptr = tensor.send(self.bob)
        assert isinstance(tensor_ptr.__repr__(), str)

    def test_overload_reshape(self):
        tensor = torch.Tensor([1, 2, 3, 4])
        tensor_reshaped = tensor.reshape((2, 2))
        tensor_matrix = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert (tensor_reshaped == tensor_matrix).all()

    def test_owner_default(self):
        self.setUp()
        tensor = torch.Tensor([1, 2, 3, 4, 5])

        assert tensor.owner == self.hook.local_worker

    def test_create_pointer(self):
        self.setUp()
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
        self.setUp()
        tensor = torch.Tensor([1, 2, 3, 4, 5])

        ptr = tensor.create_pointer(location=self.bob)

        assert ptr.owner == tensor.owner
        assert ptr.location == self.bob

    def test_get(self):
        self.setUp()
        tensor = torch.rand(5, 3)
        tensor.owner = self.me
        tensor.id = 1

        pointer = tensor.send(self.bob)

        assert type(pointer.child) == PointerTensor
        assert (pointer.get() == tensor).all()
