"""All the tests relative to garbage collection of all kinds of remote or local tensors"""

import syft as sy
import torch

hook = sy.TorchHook(torch)
from torch import nn
from torch import optim
import random


class TestFederatedLearning(object):
    def setUp(self):
        hook = sy.TorchHook(torch, verbose=True)

        self.me = hook.local_worker
        self.me.is_client_worker = True

        instance_id = str(int(10e10 * random.random()))
        bob = sy.VirtualWorker(id=f"bob{instance_id}", hook=hook, is_client_worker=False)
        alice = sy.VirtualWorker(id=f"alice{instance_id}", hook=hook, is_client_worker=False)
        james = sy.VirtualWorker(id=f"james{instance_id}", hook=hook, is_client_worker=False)

        bob.add_workers([alice, james])
        alice.add_workers([bob, james])
        james.add_workers([bob, alice])

        self.hook = hook

        self.bob = bob
        self.alice = alice
        self.james = james

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
        self.datasets = [(data_bob, target_bob), (data_alice, target_alice)]

    # POINTERS

    def test_toy_federated_learning(self):

        self.setUp()

        # Initialize A Toy Model
        model = nn.Linear(2, 1)

        # Training Logic
        opt = optim.SGD(params=model.parameters(), lr=0.1)
        for iter in range(20):

            # NEW) iterate through each worker's dataset
            for data, target in self.datasets:
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
