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
        self.me.is_client_worker = False

        bob = syft.VirtualWorker(id="bob", hook=hook, is_client_worker=False)
        alice = syft.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
        james = syft.VirtualWorker(id="james", hook=hook, is_client_worker=False)

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
