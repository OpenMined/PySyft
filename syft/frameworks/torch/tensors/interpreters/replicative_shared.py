import torch
from syft.generic.abstract.tensor import AbstractTensor
import random
import syft
from operator import add, sub


class ReplicatedSharingTensor(AbstractTensor):
    def __init__(
        self, shares_map=None, owner=None, id=None, tags=None, description=None,
    ):
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.child = shares_map
        self.ring_size = 2 ** 5

    def share_secret(self, secret, workers):
        number_of_shares = len(workers)
        workers = self.__arrange_workers(workers)
        shares = self.generate_shares(secret, number_of_shares)
        shares_map = self.__distribute_shares(workers, shares)
        self.child = shares_map
        return self

    @staticmethod
    def __arrange_workers(workers):
        """ having local worker in index 0 saves one communication round"""
        if syft.hook.local_worker in workers:
            workers.pop(syft.hook.local_worker)
            workers = [syft.hook.local_worker] + workers
        return workers

    def generate_shares(self, plain_text, number_of_shares=3):
        shares = []
        for _ in range(number_of_shares - 1):
            shares.append(torch.tensor(random.randrange(self.ring_size)))
        shares.append(torch.tensor((plain_text - sum(shares)) % self.ring_size))
        return shares

    @staticmethod
    def __distribute_shares(workers, shares):
        shares_map = {}
        for i in range(len(shares)):
            pointer1 = shares[i].send(workers[i])
            pointer2 = shares[(i + 1) % len(shares)].send(workers[i])
            shares_map[workers[i]] = (pointer1, pointer2)
        return shares_map

    def reconstruct(self):
        shares_map = self.get_shares_map()
        shares = self.__retrieve_shares(shares_map)
        plain_text_mod = self.__sum_shares(shares)
        plain_text = self.__map_modular_to_real(plain_text_mod)
        return plain_text

    def __retrieve_shares(self, shares_map):
        pointers = self.__retrieve_pointers(shares_map)
        shares = []
        for pointer in pointers:
            shares.append(pointer.get())
        return shares

    @staticmethod
    def __retrieve_pointers(shares_map):
        players = list(shares_map.keys())
        pointers = list(shares_map[players[0]])
        pointers.append(shares_map[players[1]][1])
        return pointers

    def __sum_shares(self, shares):
        return sum(shares) % self.ring_size

    def __map_modular_to_real(self, mod_number):
        """In a modular ring, a number x is mapped to a negative
         real number ]0,-∞[ iff x > ring_size/2 """
        element_wise_comparison = mod_number > self.ring_size // 2
        real_number = (element_wise_comparison * -self.ring_size) + mod_number
        return real_number

    def add(self, value):
        return self.__switch_public_private(value, self.public_add, self.private_add)

    def public_add(self, plain_text):
        return self.public_linear_operation(plain_text, add)

    def private_add(self, secret):
        return self.private_linear_operation(secret, add)

    def sub(self, value):
        return self.__switch_public_private(value, self.public_sub, self.private_sub)

    def public_sub(self, plain_text):
        return self.public_linear_operation(plain_text, sub)

    def private_sub(self, secret):
        return self.private_linear_operation(secret, sub)

    @staticmethod
    def __switch_public_private(value, public_function, private_function):
        if isinstance(value, (int, float, torch.Tensor, syft.FixedPrecisionTensor)):
            return public_function(value)
        elif isinstance(value, syft.ReplicatedSharingTensor):
            return private_function(value)
        else:
            raise NotImplementedError(
                "ReplicatedSharingTensor can only be added to"
                " int, float, torch tensor, or ReplicatedSharingTensor"
            )

    def private_linear_operation(self, secret, operator):
        if not self.verify_matching_players(secret):
            raise ValueError("Shares must be distributed among same parties")
        z = {}
        x, y = self.get_shares_map(), secret.get_shares_map()
        for player in x.keys():
            z[player] = (operator(x[player][0], y[player][0]), operator(x[player][1], y[player][1]))
        return ReplicatedSharingTensor(z)

    def public_linear_operation(self, plain_text, operator):
        players = self.get_players()
        shares_map = self.get_shares_map()
        plain_text = torch.tensor(plain_text).send(players[0])
        shares_map[players[0]] = (
            operator(shares_map[players[0]][0], plain_text),
            shares_map[players[0]][1],
        )
        return syft.ReplicatedSharingTensor(shares_map)

    def verify_matching_players(self, *secrets):
        players_set_0 = self.get_players()
        for secret in secrets:
            players_set_i = secret.get_players()
            if players_set_i != players_set_0:
                return False
        return True

    def get_players(self):
        return list(self.get_shares_map().keys())

    def get_shares_map(self):
        """
        shares_map: dic(worker i : (share_pointer i, share_pointer i+1)
        """
        return self.child

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        type_name = type(self).__name__
        out = f"[" f"{type_name}]"
        if self.child is not None:
            for v in self.child.values():
                out += "\n\t-> " + str(v)
        return out
