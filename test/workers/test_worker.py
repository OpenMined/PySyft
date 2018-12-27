from unittest import TestCase

from syft.exceptions import WorkerNotFoundException
from syft.workers import VirtualWorker


class TestHook(TestCase):
    def test___init__(self):
        alice = VirtualWorker(id="alice")
        bob = VirtualWorker(id="bob", known_workers={alice.id: alice})
        charlie = VirtualWorker(id="charlie")

        assert bob.get_worker("alice").id == alice.id
        assert bob.get_worker(alice).id == alice.id
        assert bob.get_worker(charlie).id == charlie.id

        bob.get_worker("the_unknown_worker")

        bob.add_worker(alice)


class TestWorker(TestCase):
    def test_get_unknown_worker(self):
        bob = VirtualWorker(id="bob")
        charlie = VirtualWorker(id="charlie")

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
