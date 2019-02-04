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

    fd = sy.FederatedDataset(data, target, batch_size=32, num_iterators=1)

    for iter in range(5):
        fd.reset()
        i = 0
        loss_accum = 0
        while fd.keep_going():
            i += 1

            data, target = fd.step()
