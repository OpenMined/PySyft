import torch as th
import syft as sy
from syft.frameworks.torch import federated


def test_federated_dataloader(workers):
    bob = workers["bob"]
    alice = workers["alice"]
    datasets = [
        federated.BaseDataset(th.tensor([1, 2]), th.tensor([1, 2])).send(bob),
        federated.BaseDataset(th.tensor([3, 4, 5, 6]), th.tensor([3, 4, 5, 6])).send(alice),
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


def test_federated_dataloader_shuffle(workers):
    bob = workers["bob"]
    alice = workers["alice"]
    datasets = [
        federated.BaseDataset(th.tensor([1, 2]), th.tensor([1, 2])).send(bob),
        federated.BaseDataset(th.tensor([3, 4, 5, 6]), th.tensor([3, 4, 5, 6])).send(alice),
    ]
    fed_dataset = sy.FederatedDataset(datasets)

    fdataloader = sy.FederatedDataLoader(fed_dataset, batch_size=2, shuffle=True)
    for epoch in range(3):
        counter = 0
        for batch_idx, (data, target) in enumerate(fdataloader):
            if counter < 1:  # one batch for bob, two batches for alice (batch_size == 2)
                assert (
                    data.location.id == "bob"
                ), "id should be bob, counter = {0}, epoch = {1}".format(counter, epoch)
            else:
                assert (
                    data.location.id == "alice"
                ), "id should be alice, counter = {0}, epoch = {1}".format(counter, epoch)
            counter += 1
        assert counter == len(fdataloader), f"{counter} == {len(fdataloader)}"

    num_iterators = 2
    fdataloader = sy.FederatedDataLoader(
        fed_dataset, batch_size=2, num_iterators=num_iterators, shuffle=True
    )
    assert (
        fdataloader.num_iterators == num_iterators - 1
    ), f"{fdataloader.num_iterators} == {num_iterators - 1}"


def test_federated_dataloader_num_iterators(workers):
    bob = workers["bob"]
    alice = workers["alice"]
    james = workers["james"]
    datasets = [
        federated.BaseDataset(th.tensor([1, 2]), th.tensor([1, 2])).send(bob),
        federated.BaseDataset(th.tensor([3, 4, 5, 6]), th.tensor([3, 4, 5, 6])).send(alice),
        federated.BaseDataset(th.tensor([7, 8, 9, 10]), th.tensor([7, 8, 9, 10])).send(james),
    ]

    fed_dataset = sy.FederatedDataset(datasets)
    num_iterators = len(datasets)
    fdataloader = sy.FederatedDataLoader(
        fed_dataset, batch_size=2, num_iterators=num_iterators, shuffle=True
    )
    assert (
        fdataloader.num_iterators == num_iterators - 1
    ), f"{fdataloader.num_iterators} == {num_iterators - 1}"
    counter = 0
    for batch_idx, batches in enumerate(fdataloader):
        assert (
            len(batches.keys()) == num_iterators - 1
        ), f"len(batches.keys()) == {num_iterators} - 1"
        if batch_idx < 1:
            data_bob, target_bob = batches[bob]
            assert data_bob.location.id == "bob", "id should be bob, batch_idx = {0}".format(
                batch_idx
            )
        else:  # bob is replaced by james
            data_james, target_james = batches[james]
            assert data_james.location.id == "james", "id should be james, batch_idx = {0}".format(
                batch_idx
            )
        if batch_idx < 2:
            data_alice, target_alice = batches[alice]
            assert data_alice.location.id == "alice", "id should be alice, batch_idx = {0}".format(
                batch_idx
            )
        counter += 1
    epochs = num_iterators - 1
    assert counter * (num_iterators - 1) == epochs * len(
        fdataloader
    ), " == epochs * len(fdataloader)"


def test_federated_dataloader_iter_per_worker(workers):
    bob = workers["bob"]
    alice = workers["alice"]
    james = workers["james"]
    datasets = [
        federated.BaseDataset(th.tensor([1, 2]), th.tensor([1, 2])).send(bob),
        federated.BaseDataset(th.tensor([3, 4, 5, 6]), th.tensor([3, 4, 5, 6])).send(alice),
        federated.BaseDataset(th.tensor([7, 8, 9, 10]), th.tensor([7, 8, 9, 10])).send(james),
    ]

    fed_dataset = sy.FederatedDataset(datasets)
    fdataloader = sy.FederatedDataLoader(
        fed_dataset, batch_size=2, iter_per_worker=True, shuffle=True
    )
    nr_workers = len(datasets)
    assert (
        fdataloader.num_iterators == nr_workers
    ), "num_iterators should be equal to number or workers"
    for batch_idx, batches in enumerate(fdataloader):
        assert len(batches.keys()) == nr_workers, "return a batch for each worker"


def test_federated_dataloader_one_worker(workers):
    bob = workers["bob"]

    datasets = [federated.BaseDataset(th.tensor([3, 4, 5, 6]), th.tensor([3, 4, 5, 6])).send(bob)]

    fed_dataset = sy.FederatedDataset(datasets)
    num_iterators = len(datasets)
    fdataloader = sy.FederatedDataLoader(fed_dataset, batch_size=2, shuffle=True)
    assert fdataloader.num_iterators == 1, f"{fdataloader.num_iterators} == {1}"
