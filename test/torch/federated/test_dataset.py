import pytest
import torch as th
import syft as sy

from syft.frameworks.torch.fl import BaseDataset


def test_base_dataset(workers):

    bob = workers["bob"]
    inputs = th.tensor([1, 2, 3, 4.0])
    targets = th.tensor([1, 2, 3, 4.0])
    dataset = BaseDataset(inputs, targets)

    assert len(dataset) == 4
    assert dataset[2] == (3, 3)

    dataset = dataset.send(bob)
    assert dataset.data.location.id == "bob"
    assert dataset.targets.location.id == "bob"
    assert dataset.location.id == "bob"


def test_base_dataset_transform():

    inputs = th.tensor([1, 2, 3, 4.0])
    targets = th.tensor([1, 2, 3, 4.0])

    transform_dataset = BaseDataset(inputs, targets)

    def func(x):

        return x * 2

    transform_dataset.transform(func)

    expected_val = th.tensor([2, 4, 6, 8])
    transformed_val = [val[0].item() for val in transform_dataset]

    assert expected_val.equal(th.tensor(transformed_val).long())


def test_federated_dataset(workers):
    bob = workers["bob"]
    alice = workers["alice"]

    alice_base_dataset = BaseDataset(th.tensor([3, 4, 5, 6]), th.tensor([3, 4, 5, 6]))
    datasets = [
        BaseDataset(th.tensor([1, 2]), th.tensor([1, 2])).send(bob),
        alice_base_dataset.send(alice),
    ]

    fed_dataset = sy.FederatedDataset(datasets)

    assert fed_dataset.workers == ["bob", "alice"]
    assert len(fed_dataset) == 6

    alice_remote_data = fed_dataset.get_dataset("alice")
    assert (alice_remote_data.data == alice_base_dataset.data).all()
    assert alice_remote_data[2] == (5, 5)
    assert len(alice_remote_data) == 4
    assert len(fed_dataset) == 2

    assert isinstance(fed_dataset.__str__(), str)


def test_dataset_to_federate(workers):
    bob = workers["bob"]
    alice = workers["alice"]

    dataset = BaseDataset(th.tensor([1.0, 2, 3, 4, 5, 6]), th.tensor([1.0, 2, 3, 4, 5, 6]))

    fed_dataset = dataset.federate((bob, alice))

    assert isinstance(fed_dataset, sy.FederatedDataset)

    assert fed_dataset.workers == ["bob", "alice"]
    assert fed_dataset["bob"].location.id == "bob"
    assert len(fed_dataset) == 6


def test_federated_dataset_search(workers):

    bob = workers["bob"]
    alice = workers["alice"]

    grid = sy.PrivateGridNetwork(*[bob, alice])

    train_bob = th.Tensor(th.zeros(1000, 100)).tag("data").send(bob)
    target_bob = th.Tensor(th.zeros(1000, 100)).tag("target").send(bob)

    train_alice = th.Tensor(th.zeros(1000, 100)).tag("data").send(alice)
    target_alice = th.Tensor(th.zeros(1000, 100)).tag("target").send(alice)

    data = grid.search("data")
    target = grid.search("target")

    datasets = [
        BaseDataset(data["bob"][0], target["bob"][0]),
        BaseDataset(data["alice"][0], target["alice"][0]),
    ]

    fed_dataset = sy.FederatedDataset(datasets)
    train_loader = sy.FederatedDataLoader(fed_dataset, batch_size=4, shuffle=False, drop_last=False)

    counter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        counter += 1

    assert counter == len(train_loader), f"{counter} == {len(fed_dataset)}"


def test_abstract_dataset():
    inputs = th.tensor([1, 2, 3, 4.0])
    targets = th.tensor([1, 2, 3, 4.0])
    dataset = BaseDataset(inputs, targets, id=1)

    assert dataset.id == 1
    assert dataset.description is None


def test_get_dataset(workers):
    bob = workers["bob"]
    alice = workers["alice"]

    alice_base_dataset = BaseDataset(th.tensor([3, 4, 5, 6]), th.tensor([3, 4, 5, 6]))
    datasets = [
        BaseDataset(th.tensor([1, 2]), th.tensor([1, 2])).send(bob),
        alice_base_dataset.send(alice),
    ]
    fed_dataset = sy.FederatedDataset(datasets)
    dataset = fed_dataset.get_dataset("alice")

    assert len(fed_dataset) == 2
    assert len(dataset) == 4


def test_illegal_get(workers):
    """
    test getting error message when calling .get() on a
    dataset that's a part of fedratedDataset object
    """
    bob = workers["bob"]
    alice = workers["alice"]

    alice_base_dataset = BaseDataset(th.tensor([3, 4, 5, 6]), th.tensor([3, 4, 5, 6]))
    datasets = [
        BaseDataset(th.tensor([1, 2]), th.tensor([1, 2])).send(bob),
        alice_base_dataset.send(alice),
    ]
    fed_dataset = sy.FederatedDataset(datasets)
    with pytest.raises(ValueError):
        fed_dataset["alice"].get()
