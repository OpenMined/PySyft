from unittest import TestCase
import torch


import syft
from syft.workers import VirtualWorker
from syft.frameworks.torch.tensors import TorchTensor, PointerTensor


def setUpModule():

    global me
    global bob
    global alice
    global james
    global hook

    hook = syft.TorchHook(torch, verbose=True)

    me = hook.local_worker
    me.is_client_worker = False

    bob = syft.VirtualWorker(id="bob", hook=hook, is_client_worker=False)
    alice = syft.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
    james = syft.VirtualWorker(id="james", hook=hook, is_client_worker=False)

    bob.add_workers([alice, james])
    alice.add_workers([bob, james])
    james.add_workers([bob, alice])


class TestNative(TestCase):
    def test_init(self):
        tensor_extension = TorchTensor()
        assert tensor_extension.id is None
        assert tensor_extension.owner is None


class TestPointer(TestCase):
    def test_init(self):
        alice = VirtualWorker(id="alice")
        pointer = PointerTensor(id=1000, location=alice, owner=alice)
        pointer.__str__()

    def test_create_pointer(self):
        x = torch.Tensor([1, 2])
        x.create_pointer()
        x.create_pointer(location=james)
