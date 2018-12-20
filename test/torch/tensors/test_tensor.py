import torch


import syft
from syft.frameworks.torch.tensors import TorchTensor, PointerTensor


# class TestNative(object):
#     def test_init(self):
#         tensor_extension = TorchTensor()
#         assert tensor_extension.id is None
#         assert tensor_extension.owner is None


class TestPointer(object):
    def setUp(self):
        hook = syft.TorchHook(torch, verbose=True)

        me = hook.local_worker
        me.is_client_worker = False

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
        alice = syft.VirtualWorker(id="alice")
        pointer = PointerTensor(id=1000, location=alice, owner=alice)
        pointer.__str__()

    def test_create_pointer(self):
        self.setUp()
        x = torch.Tensor([1, 2])
        x.create_pointer()
        x.create_pointer(location=self.james)
