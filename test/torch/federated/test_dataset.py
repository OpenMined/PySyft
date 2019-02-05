import pytest
import torch as th
import syft as sy
import numpy as np


def test_federated_dataset(workers):

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

    epochs = 2
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            pass
