import torch as th
import syft as sy

from syft.frameworks.torch.federated import BaseDataset


def test_base_dataset(workers):
    bob = workers["bob"]

    inputs = th.tensor([1, 2, 3, 4.0])
    targets = th.tensor([1, 2, 3, 4.0])
    dataset = BaseDataset(inputs, targets)
    assert len(dataset) == 4
    assert dataset[2] == (3, 3)

    dataset.send(bob)
    assert dataset.data.location.id == "bob"
    assert dataset.targets.location.id == "bob"
    assert dataset.location.id == "bob"

    dataset.get()
    try:
        assert dataset.data.location.id == 0
        assert dataset.targets.location.id == 0
        assert False
    except AttributeError:
        pass


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

    fed_dataset["alice"].get()
    assert (fed_dataset["alice"].data == alice_base_dataset.data).all()
    assert fed_dataset["alice"][2] == (5, 5)
    assert len(fed_dataset["alice"]) == 4
    assert len(fed_dataset) == 6

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


def test_federated_dataloader(workers):
    bob = workers["bob"]
    alice = workers["alice"]
    datasets = [
        BaseDataset(th.tensor([1, 2]), th.tensor([1, 2])).send(bob),
        BaseDataset(th.tensor([3, 4, 5, 6]), th.tensor([3, 4, 5, 6])).send(alice),
    ]
    fed_dataset = sy.FederatedDataset(datasets)

    fdataloader = sy.FederatedDataLoader(fed_dataset, batch_size=2)
    counter = 0
    for batch_idx, (data, target) in enumerate(fdataloader):
        counter += 1

    assert counter == len(fdataloader), f"{counter} == {len(fdataloader)}"

    fdataloader = sy.FederatedDataLoader(fed_dataset, batch_size=2, drop_last=True)
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
