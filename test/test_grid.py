import torch
import syft as sy


def test_virtual_grid(workers):
    """This tests our ability to simplify tuple types.

    This test is pretty simple since tuples just serialize to
    themselves, with a tuple wrapper with the correct ID (1)
    for tuples so that the detailer knows how to interpret it."""

    print(len(workers))
    print(workers)

    bob = workers["bob"]
    alice = workers["alice"]
    james = workers["james"]

    grid = sy.PrivateGridNetwork(*[bob, alice, james])

    x = torch.tensor([1, 2, 3, 4]).tag("#bob", "#male").send(bob)
    y = torch.tensor([1, 2, 3, 4]).tag("#alice", "#female").send(alice)
    z = torch.tensor([1, 2, 3, 4]).tag("#james", "#male").send(james)

    results = grid.search()
    assert len(results) == 3

    assert "bob" in results.keys()
    assert "alice" in results.keys()
    assert "james" in results.keys()

    results = grid.search("#bob")
    assert len(results["bob"]) == 1
    assert "alice" not in results
    assert "james" not in results

    results = grid.search("#male")
    assert len(results["bob"]) == 1
    assert "alice" not in results
    assert len(results["james"]) == 1
