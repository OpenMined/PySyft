import torch
from syft.generic.pointers.pointer_dataset import PointerDataset
from syft.frameworks.torch.fl.dataset import BaseDataset


def test_create_dataset_pointer(workers):
    alice, bob, me = workers["alice"], workers["bob"], workers["me"]

    data = torch.tensor([1, 2, 3, 4])
    target = torch.tensor([5, 6, 7, 8])
    dataset = BaseDataset(data, target)
    ptr = dataset.send(alice)

    assert type(ptr) == PointerDataset
    assert ptr.location == alice
    assert ptr.owner == me


def test_search_dataset(workers):
    alice, bob, me = workers["alice"], workers["bob"], workers["me"]

    data = torch.tensor([1, 2, 3, 4])
    target = torch.tensor([5, 6, 7, 8])
    dataset = BaseDataset(data, target).tag("#test").describe("test search dataset")
    ptr = dataset.send(alice)
    results = me.request_search(["#test"], location=alice)

    assert results[0].id_at_location == ptr.id_at_location


def test_get_dataset(workers):
    alice, bob, me = workers["alice"], workers["bob"], workers["me"]

    data = torch.tensor([1, 2, 3, 4])
    target = torch.tensor([5, 6, 7, 8])
    dataset = BaseDataset(data, target)
    ptr = dataset.send(alice)
    dataset = ptr.get()

    assert torch.equal(dataset.data, data) == 1
    assert torch.equal(dataset.targets, target) == 1
    assert dataset.owner == me


def test_get_data_targets(workers):
    alice, bob, me = workers["alice"], workers["bob"], workers["me"]

    data = torch.tensor([1, 2, 3, 4])
    target = torch.tensor([5, 6, 7, 8])
    dataset = BaseDataset(data, target)
    ptr = dataset.send(alice)
    remote_targets = ptr.targets.get()
    remote_data = ptr.data.get()

    assert torch.equal(remote_data, data) == 1
    assert torch.equal(remote_targets, target) == 1
