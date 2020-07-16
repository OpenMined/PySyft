import torch
from syft.generic.abstract.tensor import AbstractTensor
import random
import syft


class ReplicatedSharingTensor(AbstractTensor):
    def __init__(
        self, shares=None, owner=None, id=None, tags=None, description=None,
    ):
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.child = shares
        self.ring_size = 2 ** 5

    def share_secret(self, secret, workers):
        number_of_shares = len(workers)
        shares = self.generate_shares(secret, number_of_shares)
        shares_locations = self.distribute_shares(workers, shares)
        self.child = shares_locations
        return self

    def generate_shares(self, secret, number_of_shares=3):
        shares = []
        for _ in range(number_of_shares - 1):
            shares.append(torch.tensor(random.randrange(self.ring_size)))
        shares.append(torch.tensor((secret - sum(shares)) % self.ring_size))
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

    def reconstruct(self):
        shares_locations = self.child
        shares = self.retrieve_shares(shares_locations)
        plain_text = self.sum_shares(shares)
        return plain_text

    @staticmethod
    def retrieve_shares(shares_locations):
        shares = []
        pointers = list(shares_locations.values())
        for pointer_double in pointers:
            share0 = pointer_double[0].get()
            shares.append(share0)
        return shares

    def sum_shares(self, shares):
        return sum(shares) % self.ring_size

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        type_name = type(self).__name__
        out = f"[" f"{type_name}]"
        if self.child is not None:
            for v in self.child.values():
                out += "\n\t-> " + str(v)
        return out

    def private_add(self, secret):
        if not self.verify_matching_players(secret):
            raise ValueError("Shares must be distributed among same parties")
        z = {}
        x, y = self.get_shares_pointers(self, secret)
        for player in x.keys():
            z[player] = (x[player][0] + y[player][0], x[player][1] + y[player][1])
        return ReplicatedSharingTensor(z)

    def verify_matching_players(self, *secrets):
        players_set_0 = self.get_players(self)
        for secret in secrets:
            players_set_i = self.get_players(secret)
            if players_set_i != players_set_0:
                return False
        return True

    def get_players(self, secret):
        return list(self.get_shares_pointers(secret).keys())

    @staticmethod
    def get_shares_pointers(*secrets):
        pointers_dict = [secret.child for secret in secrets]
        return pointers_dict if len(pointers_dict) > 1 else pointers_dict[0]
