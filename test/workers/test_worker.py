from syft.workers import VirtualWorker


def test___init__():
    alice = VirtualWorker(id="alice")
    bob = VirtualWorker(id="bob", known_workers={alice.id: alice})
    charlie = VirtualWorker(id="charlie")

    assert bob.get_worker("alice").id == alice.id
    assert bob.get_worker(alice).id == alice.id
    assert bob.get_worker(charlie).id == charlie.id

    bob.get_worker("the_unknown_worker")

    bob.add_worker(alice)
