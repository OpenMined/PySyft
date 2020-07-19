import torch
from syft.generic.abstract.tensor import AbstractTensor
import random
import syft
from operator import add, sub


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
        shares_locations = self.__distribute_shares(workers, shares)
        self.child = shares_locations
        return self

    def generate_shares(self, secret, number_of_shares=3):
        shares = []
        for _ in range(number_of_shares - 1):
            shares.append(torch.tensor(random.randrange(self.ring_size)))
        shares.append(torch.tensor((secret - sum(shares)) % self.ring_size))
        return shares

    @staticmethod
    def __distribute_shares(workers, shares):
        shares_locations = {}
        assert len(workers) == len(shares)
        for i in range(len(shares)):
            pointer1 = shares[i].send(workers[i])
            pointer2 = shares[(i + 1) % len(shares)].send(workers[i])
            shares_locations[workers[i]] = (pointer1, pointer2)
        return shares_locations

    def reconstruct(self):
        shares_locations = self.child
        shares = self.__retrieve_shares(shares_locations)
        plain_text_mod = self.__sum_shares(shares)
        plain_text = self.__map_modular_to_real(plain_text_mod)
        return plain_text

    @staticmethod
    def __retrieve_shares(shares_locations):
        shares = []
        pointers = list(shares_locations.values())
        for pointer_double in pointers:
            share0 = pointer_double[0].get()
            shares.append(share0)
        return shares

    def __sum_shares(self, shares):
        return sum(shares) % self.ring_size

    def __map_modular_to_real(self, mod_number):
        """In a modular ring, a number x is mapped to negative
         real number ]0,-∞[ iff x > ring_size/2 """
        if mod_number > self.ring_size // 2:
            real_number = mod_number - self.ring_size
        else:
            real_number = mod_number
        return real_number

    def add(self, y):
        if isinstance(y, (int, float, torch.Tensor)):
            return self.public_add(y)
        elif isinstance(y, syft.ReplicatedSharingTensor):
            return self.private_add(y)
        else:
            raise ValueError(
                "ReplicatedSharingTensor can only be added to"
                " int, float, torch tensor, or ReplicatedSharingTensor"
            )

    def public_add(self, plain_text):
        plain_text = torch.tensor(plain_text)
        players = self.get_players()
        y = syft.ReplicatedSharingTensor().share_secret(plain_text, players)
        z = self.private_add(y)
        return z

    def private_add(self, secret):
        return self.linear_operation(secret, add)

    def sub(self, y):
        if isinstance(y, (int, float, torch.Tensor)):
            return self.public_sub(y)
        elif isinstance(y, syft.ReplicatedSharingTensor):
            return self.private_sub(y)
        else:
            raise ValueError(
                "ReplicatedSharingTensor can only be added to"
                " int, float, torch tensor, or ReplicatedSharingTensor"
            )

    def public_sub(self, y):
        return self.add(-y)

    def private_sub(self, secret):
        return self.linear_operation(secret, sub)

    def linear_operation(self, secret, operator):
        if not self.verify_matching_players(secret):
            raise ValueError("Shares must be distributed among same parties")
        z = {}
        x, y = self.get_pointers_map(self, secret)
        for player in x.keys():
            z[player] = (operator(x[player][0], y[player][0]), operator(x[player][1], y[player][1]))
        return ReplicatedSharingTensor(z)

    def verify_matching_players(self, *secrets):
        players_set_0 = self.get_players()
        for secret in secrets:
            players_set_i = secret.get_players()
            if players_set_i != players_set_0:
                return False
        return True

    def get_players(self):
        return list(self.get_pointers_map(self).keys())

    @staticmethod
    def get_pointers_map(*secrets):
        """pointer_map: dict(worker i : (pointer_to_share i, pointer_to_share i+1)"""
        pointers_maps = [secret.child for secret in secrets]
        return pointers_maps if len(pointers_maps) > 1 else pointers_maps[0]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        type_name = type(self).__name__
        out = f"[" f"{type_name}]"
        if self.child is not None:
            for v in self.child.values():
                out += "\n\t-> " + str(v)
        return out
