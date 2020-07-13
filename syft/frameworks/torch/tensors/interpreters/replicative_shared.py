import math
import torch
import syft as sy
from syft.generic.abstract.tensor import AbstractTensor
import random


class ReplicatedSharingTensor(AbstractTensor):
    def __init__(
        self, shares=None, owner=None, id=None, crypto_provider=None, tags=None, description=None,
    ):
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.child = shares
        self.ring_size = 2 ** 5
        self.crypto_provider = (
            crypto_provider if crypto_provider is not None else sy.hook.local_worker
        )

    def share_secret(self, secret, workers):
        shares = self.generate_shares(secret)
        shares_locations = self.distribute_shares(workers, shares)
        self.child = shares_locations

    def generate_shares(self, secret, number_of_shares=3):
        shares = []
        for _ in range(number_of_shares - 1):
            shares.append(random.randrange(self.ring_size))
        shares.append(secret - sum(shares) % self.ring_size)
        return shares

    @staticmethod
    def distribute_shares(workers, shares):
        shares_locations = {}
        assert len(workers) == len(shares)
        for i in range(len(shares)):
            pointer1 = shares[i].send(workers[i])
            pointer2 = shares[(i + 1) % len(shares)].send(workers[i])
            shares_locations[workers[i].id] = (pointer1, pointer2)
        return shares_locations

    def reconstruct(self, shares):
        return sum(shares) % self.ring_size
