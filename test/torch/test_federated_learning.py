"""All the tests relative to garbage collection of all kinds of remote or local tensors"""

import syft as sy
import torch

hook = sy.TorchHook(torch)
from torch import nn
from torch import optim


def test_toy_federated_learning(workers):

    bob, alice = workers["bob"], workers["alice"]

    # A Toy Dataset
    data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1.0]], requires_grad=True)
    target = torch.tensor([[0], [0], [1], [1.0]], requires_grad=True)

    # get pointers to training data on each worker by
    # sending some training data to bob and alice
    data_bob = data[0:2]
    target_bob = target[0:2]

    data_alice = data[2:]
    target_alice = target[2:]

    data_bob = data_bob.send(bob)
    data_alice = data_alice.send(alice)
    target_bob = target_bob.send(bob)
    target_alice = target_alice.send(alice)

    # organize pointers into a list
    datasets = [(data_bob, target_bob), (data_alice, target_alice)]

    # Initialize A Toy Model
    model = nn.Linear(2, 1)

    # Training Logic
    opt = optim.SGD(params=model.parameters(), lr=0.1)
    for iter in range(20):

        # NEW) iterate through each worker's dataset
        for data, target in datasets:
            # NEW) send model to correct worker
            model.send(data.location)

            # 1) erase previous gradients (if they exist)
            opt.zero_grad()

            # 2) make a prediction
            pred = model(data)

            # 3) calculate how much we missed
            loss = ((pred - target) ** 2).sum()

            # 4) figure out which weights caused us to miss
            loss.backward()

            # 5) change those weights
            opt.step()

            # get model (with gradients)
            model.get()

            # 6) print our progress
            print(loss.get())  # NEW) slight edit... need to call .get() on loss


def test_lstm(workers):
    bob, alice = workers["bob"], workers["alice"]

    lstm = nn.LSTM(3, 3)
    inputs = torch.randn(5, 1, 3)
    hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
    out, hidden = lstm(inputs, hidden)
    assert out.shape == torch.Size([5, 1, 3])

    lstm = nn.LSTM(3, 3)
    lstm.send(bob)
    inputs = torch.randn(5, 1, 3).send(bob)
    hidden = (
        torch.randn(1, 1, 3).send(bob),
        torch.randn(1, 1, 3).send(bob),
    )  # clean out hidden state
    out, hidden = lstm(inputs, hidden)
    assert out.shape == torch.Size([5, 1, 3])
