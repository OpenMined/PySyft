from unittest import TestCase

import torch
import syft as sy
from syft.exceptions import WorkerNotFoundException
from syft.workers import VirtualWorker


def test___init__():
    hook = sy.TorchHook(torch)

    alice = VirtualWorker(hook, id="alice")
    bob = VirtualWorker(hook, id="bob", known_workers={alice.id: alice})
    charlie = VirtualWorker(hook, id="charlie")

    assert bob.get_worker("alice").id == alice.id
    assert bob.get_worker(alice).id == alice.id
    assert bob.get_worker(charlie).id == charlie.id

    bob.get_worker("the_unknown_worker")

    bob.add_worker(alice)


class TestWorker(TestCase):
    def test_get_unknown_worker(self):

        hook = sy.TorchHook(torch)

        bob = VirtualWorker(hook, id="bob")
        charlie = VirtualWorker(hook, id="charlie")

        # if an unknown string or id representing a worker is given it fails
        try:
            bob.get_worker("the_unknown_worker", fail_hard=True)
            assert False
        except WorkerNotFoundException:
            assert True

        try:
            bob.get_worker(1, fail_hard=True)
            assert False
        except WorkerNotFoundException:
            assert True

        # if an instance of virtual worker is given it doesn't fail
        assert bob.get_worker(charlie).id == charlie.id
        assert charlie.id in bob._known_workers

    def test_search(self):
        bob = VirtualWorker(sy.torch.hook)

        x = (
            torch.tensor([1, 2, 3, 4, 5])
            .tag("#fun", "#mnist")
            .describe("The images in the MNIST training dataset.")
            .send(bob)
        )

        y = (
            torch.tensor([1, 2, 3, 4, 5])
            .tag("#not_fun", "#cifar")
            .describe("The images in the MNIST training dataset.")
            .send(bob)
        )

        z = (
            torch.tensor([1, 2, 3, 4, 5])
            .tag("#fun", "#boston_housing")
            .describe("The images in the MNIST training dataset.")
            .send(bob)
        )

        a = (
            torch.tensor([1, 2, 3, 4, 5])
            .tag("#not_fun", "#boston_housing")
            .describe("The images in the MNIST training dataset.")
            .send(bob)
        )

        assert len(bob.search("#fun")) == 2
        assert len(bob.search("#mnist")) == 1
        assert len(bob.search("#cifar")) == 1
        assert len(bob.search("#not_fun")) == 2
        assert len(bob.search("#not_fun", "#boston_housing")) == 1
