from operator import add, sub, mul
import torch
import syft
from syft.generic.abstract.tensor import AbstractTensor
from syft.frameworks.torch.mpc.przs import PRZS, gen_alpha_3of3


class ReplicatedSharingTensor(AbstractTensor):
    def __init__(
        self, shares_map=None, owner=None, id=None, tags=None, description=None,
    ):
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.child = shares_map
        self.ring_size = 2 ** 5

    def share_secret(self, secret, workers):
        number_of_shares = len(workers)
        workers = self.__arrange_workers(list(workers))
        shares = self.generate_shares(secret, number_of_shares)
        shares_map = self.__distribute_shares(workers, shares)
        PRZS.setup(workers)
        self.child = shares_map
        return self

    @staticmethod
    def __arrange_workers(workers):
        """ having local worker in index 0 saves one communication round"""
        me = syft.hook.local_worker
        if me in workers:
            workers.remove(me)
            workers = [me] + workers
        return workers

    def generate_shares(self, plain_text, number_of_shares=3):
        shares = []
        for _ in range(number_of_shares - 1):
            shares.append(torch.randint(high=self.ring_size, size=plain_text.shape))
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
        shares = self.retrieve_shares()
        plain_text_mod = self.__sum_shares(shares)
        plain_text = self.__map_modular_to_real(plain_text_mod)
        return plain_text

    def retrieve_shares(self):
        pointers = self.retrieve_pointers()
        shares = []
        for pointer in pointers:
            shares.append(pointer.get())
        return shares

    def retrieve_pointers(self):
        shares_map = self.get_shares_map()
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

    __add__ = add

    def sub(self, value):
        return self.__switch_public_private(value, self.public_sub, self.private_sub)

    def public_sub(self, plain_text):
        return self.public_linear_operation(plain_text, sub)

    def private_sub(self, secret):
        return self.private_linear_operation(secret, sub)

    __sub__ = sub

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
        return ReplicatedSharingTensor(shares_map)

    def mul(self, value):
        return self.__switch_public_private(value, self.public_mul, self.private_mul)

    def public_mul(self, plain_text):
        return self.public_multiplication_operation(plain_text, mul)

    def private_mul(self, secret):
        return self.private_multiplication_operation(secret, mul)

    __mul__ = mul

    def matmul(self, value):
        return self.__switch_public_private(value, self.public_matmul, self.private_matmul)

    def public_matmul(self, plain_text):
        return self.public_multiplication_operation(plain_text, torch.matmul)

    def private_matmul(self, secret):
        return self.private_multiplication_operation(secret, torch.matmul)

    __matmul__ = matmul

    def public_multiplication_operation(self, plain_text, operator):
        players = self.get_players()
        plain_text_map = {player: torch.tensor(plain_text).send(player) for player in players}
        shares_map = self.get_shares_map()
        for player in players:
            shares_map[player] = (
                operator(shares_map[player][0], plain_text_map[player]),
                operator(shares_map[player][1], plain_text_map[player]),
            )
        return ReplicatedSharingTensor(shares_map)

    def private_multiplication_operation(self, secret, operator):
        x, y = self.get_shares_map(), secret.get_shares_map()
        players = self.get_players()
        z = [
            operator(x[player][0], y[player][0])
            + operator(x[player][1], y[player][0])
            + operator(x[player][0], y[player][1])
            for player in players
        ]
        z = self.__add_noise(z)
        z = self.__reshare(z, players)
        return ReplicatedSharingTensor(z)

    @staticmethod
    def __add_noise(shares):
        noisy_shares = [share + gen_alpha_3of3(share.location).wrap() for share in shares]
        return noisy_shares

    @staticmethod
    def __reshare(shares, workers):
        """convert 3-out-of-3 secret sharing: {player i : share i}
          to 2-out-of-3 sharing: {player i : (share i, share i+1)}  """
        shares_map = {}
        for i in range(len(shares)):
            pointer = shares[(i + 1) % len(shares)].copy().move(workers[i])
            shares_map[workers[i]] = (shares[i], pointer)
        return shares_map

    def conv2d(self, value):
        return self.__switch_public_private(value, self.public_conv2d, self.private_conv2d)

    def public_conv2d(self, plain_text):
        raise NotImplementedError()

    def private_conv2d(self, secret):
        raise NotImplementedError()

    @staticmethod
    def __switch_public_private(value, public_function, private_function, *args, **kwargs):
        if isinstance(value, (int, float, torch.Tensor, syft.FixedPrecisionTensor)):
            return public_function(value, *args, **kwargs)
        elif isinstance(value, syft.ReplicatedSharingTensor):
            return private_function(value, *args, **kwargs)
        else:
            raise ValueError(
                "expected int, float, torch tensor, or ReplicatedSharingTensor"
                "but got {}".format(type(value))
            )

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

    def apply_to_shares(self, function, *args, **kwargs):
        """
        function: a reference to a function
        e.g. torch.Tensor.view, torch.nn.F.fold
        """
        shares_map = self.get_shares_map()
        players = self.get_players()
        shares_map = {
            player: (
                function(shares_map[player][0], *args, **kwargs),
                function(shares_map[player][1], *args, **kwargs),
            )
            for player in players
        }
        return ReplicatedSharingTensor(shares_map)

    def apply_to_shares_(self, function, *args, **kwargs):
        """
        function: a reference to an in-place function
        e.g. sort, append
        """
        shares_map = self.get_shares_map()
        players = self.get_players()
        for player in players:
            function(shares_map[player][0], *args, **kwargs)
            function(shares_map[player][1], *args, **kwargs)

    @property
    def shape(self):
        return self.retrieve_pointers()[0].shape

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        type_name = type(self).__name__
        out = f"[" f"{type_name}]"
        if self.child is not None:
            for v in self.child.values():
                out += "\n\t-> " + str(v)
        return out
