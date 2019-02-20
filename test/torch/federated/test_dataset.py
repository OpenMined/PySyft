import pytest
import torch as th
import syft as sy
from syft.frameworks.torch.federated import FederatedData


def test_federated_data():
    data = {"bob": th.Tensor([1, 2]), "alice": th.Tensor([3, 4, 5, 6])}
    fed_data = FederatedData(data)
    assert (fed_data.alice == th.Tensor([3, 4, 5, 6])).all()
    assert fed_data.alice[2] == 5
    assert (fed_data["alice"] == th.Tensor([3, 4, 5, 6])).all()
    assert fed_data["alice"][2] == 5
    assert fed_data["alice", 2] == 5
    assert len(fed_data.alice) == 4
    assert len(fed_data) == 6

    assert fed_data.workers == {"bob", "alice"}
    fed_data.drop_worker("bob")
    assert fed_data.workers == {"alice"}
    try:
        fed_data.bob
        assert False
    except AttributeError:
        pass

    assert isinstance(fed_data.__str__(), str)


def test_federated_dataset_local():
    inputs = {"bob": th.Tensor([1, 2]), "alice": th.Tensor([3, 4, 5, 6])}
    targets = {"bob": th.Tensor([1, 2]), "alice": th.Tensor([3, 4, 5, 6])}
    fdataset = sy.FederatedDataset(inputs, targets)
    assert len(fdataset) == 6

    inputs = sy.frameworks.torch.federated.FederatedData(
        {"bob": th.Tensor([1, 2]), "alice": th.Tensor([3, 4, 5, 6])}
    )
    targets = sy.frameworks.torch.federated.FederatedData(
        {"bob": th.Tensor([1, 2]), "alice": th.Tensor([3, 4, 5, 6])}
    )
    fdataset = sy.FederatedDataset(inputs, targets)
    assert len(fdataset) == 6

    inputs = {"bob": th.Tensor([1, 2]), "alice": th.Tensor([3, 4, 5, 6])}
    targets = {"bob": th.Tensor([1, 2])}
    fdataset = sy.FederatedDataset(inputs, targets)
    assert fdataset.workers == {"bob"}

    try:
        inputs = {"bob": th.Tensor([1, 2]), "alice": th.Tensor([3, 4, 5, 6])}
        targets = {"bob": th.Tensor([1]), "alice": th.Tensor([3, 4, 5, 6])}
        fdataset = sy.FederatedDataset(inputs, targets)
    except AssertionError:
        pass

    assert isinstance(fdataset.__str__(), str)


def test_federated_dataset_remote(workers):
    bob = workers["bob"]
    alice = workers["alice"]
    inputs = {"bob": th.Tensor([1, 2]).send(bob), "alice": th.Tensor([3, 4, 5, 6]).send(alice)}
    targets = {"bob": th.Tensor([1, 2]).send(bob), "alice": th.Tensor([3, 4, 5, 6]).send(alice)}
    fdataset = sy.FederatedDataset(inputs, targets)
    assert len(fdataset) == 6

    inputs = sy.frameworks.torch.federated.FederatedData(
        {"bob": th.Tensor([1, 2]).send(bob), "alice": th.Tensor([3, 4, 5, 6]).send(alice)}
    )
    targets = sy.frameworks.torch.federated.FederatedData(
        {"bob": th.Tensor([1, 2]).send(bob), "alice": th.Tensor([3, 4, 5, 6]).send(alice)}
    )
    fdataset = sy.FederatedDataset(inputs, targets)
    assert len(fdataset) == 6

    inputs = {"bob": th.Tensor([1, 2]).send(bob), "alice": th.Tensor([3, 4, 5, 6]).send(alice)}
    targets = {"bob": th.Tensor([1, 2]).send(bob)}
    fdataset = sy.FederatedDataset(inputs, targets)
    assert fdataset.workers == {"bob"}

    try:
        inputs = {"bob": th.Tensor([1, 2]).send(bob), "alice": th.Tensor([3, 4, 5, 6]).send(alice)}
        targets = {"bob": th.Tensor([1, 2]).send(bob), "alice": th.Tensor([3, 4, 5, 6]).send(alice)}
        fdataset = sy.FederatedDataset(inputs, targets)
    except AssertionError:
        pass

    assert isinstance(fdataset.__str__(), str)


def test_federated_dataloader(workers):
    bob = workers["bob"]
    alice = workers["alice"]
    inputs = {"bob": th.Tensor([1, 2]), "alice": th.Tensor([3, 4, 5, 6])}
    targets = {"bob": th.Tensor([1, 2]), "alice": th.Tensor([3, 4, 5, 6])}
    fdataset = sy.FederatedDataset(inputs, targets)

    fdataloader = sy.FederatedDataLoader(fdataset, batch_size=2)
    counter = 0
    for batch_idx, (data, target) in enumerate(fdataloader):
        counter += 1

    assert counter == len(fdataloader), f"{counter} == {len(fdataloader)}"

    fdataloader = sy.FederatedDataLoader(fdataset, batch_size=2, drop_last=True)
    counter = 0
    for batch_idx, (data, target) in enumerate(fdataloader):
        counter += 1

    assert counter == len(fdataloader), f"{counter} == {len(fdataloader)}"

    inputs = {"bob": th.Tensor([1, 2]).send(bob), "alice": th.Tensor([3, 4, 5, 6]).send(alice)}
    targets = {"bob": th.Tensor([1, 2]).send(bob), "alice": th.Tensor([3, 4, 5, 6]).send(alice)}
    fdataset = sy.FederatedDataset(inputs, targets)

    fdataloader = sy.FederatedDataLoader(fdataset, batch_size=1, drop_last=True, num_iterators=2)
    counter = 0
    for batch_idx, (data, target) in enumerate(fdataloader):
        counter += 1
    assert counter == len(fdataloader), f"{counter} == {len(fdataloader)}"


def test_federated_dataset_search(workers):

    bob = workers["bob"]
    alice = workers["alice"]

    grid = sy.VirtualGrid(*[bob, alice])

    train_bob = th.Tensor(th.zeros(1000, 100)).tag("data").send(bob)
    target_bob = th.Tensor(th.zeros(1000, 100)).tag("target").send(bob)

    train_alice = th.Tensor(th.zeros(1000, 100)).tag("data").send(alice)
    target_alice = th.Tensor(th.zeros(1000, 100)).tag("target").send(alice)

    data, _ = grid.search("data")
    target, _ = grid.search("target")

    dataset = sy.FederatedDataset(data, target)
    train_loader = sy.FederatedDataLoader(dataset, batch_size=4, shuffle=False, drop_last=False)

    counter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        counter += 1

    assert counter == len(train_loader), f"{counter} == {len(fdataset)}"
