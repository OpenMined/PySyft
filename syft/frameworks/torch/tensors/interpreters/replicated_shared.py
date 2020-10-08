from operator import add, sub, mul
import torch
import syft
from syft.generic.abstract.tensor import AbstractTensor
from syft.frameworks.torch.mpc.przs import PRZS, gen_alpha_3of3


class ReplicatedSharingTensor(AbstractTensor):
    def __init__(self, plain_text=None, players=None, ring_size=None, owner=None):
        super().__init__(owner=owner)
        self.ring_size = ring_size or 2 ** 32
        shares_map = self.__validate_input(plain_text, players)
        self.child = shares_map

    def __validate_input(self, plain_text, players):
        """
        shares_map: dict(worker i : (share_pointer i, share_pointer i+1)
        """
        if plain_text is not None and players:
            if isinstance(plain_text, torch.LongTensor):
                return self.__share_secret(plain_text, players)
            elif plain_text is ReplicatedSharingTensor:
                return plain_text.child
            else:
                dtype = plain_text.dtype if hasattr(plain_text, "dtype") else type(plain_text)
                raise ValueError(f"Expected torch.(int64/long) but got {dtype}")
        else:
            return None

    def __share_secret(self, plain_text, workers):
        number_of_shares = len(workers)
        workers = self.__arrange_workers(list(workers))
        shares = self.generate_shares(plain_text, number_of_shares)
        shares_map = self.__distribute_shares(workers, shares)
        PRZS.setup(workers)
        return shares_map

    @staticmethod
    def __arrange_workers(workers):
        """ having local worker in index 0 saves one communication round"""
        if len(workers) != 3:
            raise ValueError("you must provide 3 players")
        me = syft.hook.local_worker
        if me in workers:
            workers.remove(me)
            workers = [me] + workers
        return workers

    def generate_shares(self, plain_text, number_of_shares=3):
        shares = []
        plain_text = torch.tensor(plain_text, dtype=torch.long)
        for _ in range(number_of_shares - 1):
            shares.append(torch.randint(high=self.ring_size // 2, size=plain_text.shape))
        shares.append((plain_text - sum(shares)) % self.ring_size)
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
            shares.append(pointer.copy().get())
        return shares

    def retrieve_pointers(self):
        shares_map = self.__get_shares_map()
        players = list(shares_map.keys())
        pointers = list(shares_map[players[0]])
        pointers.append(shares_map[players[1]][1])
        return pointers

    def __sum_shares(self, shares):
        return sum(shares) % self.ring_size

    def __map_modular_to_real(self, mod_number):
        """In a modular ring, a number x is mapped to a negative
        real number ]0,-∞[ iff x > ring_size/2"""
        element_wise_comparison = mod_number > self.ring_size // 2
        real_number = (element_wise_comparison * -self.ring_size) + mod_number
        return real_number

    def add(self, value):
        return self.__switch_public_private(value, self.__public_add, self.__private_add)

    def __public_add(self, plain_text):
        return self.__public_linear_operation(plain_text, add)

    def __private_add(self, secret):
        return self.__private_linear_operation(secret, add)

    __add__ = add
    __radd__ = add

    def sub(self, value):
        return self.__switch_public_private(value, self.__public_sub, self.__private_sub)

    def __public_sub(self, plain_text):
        return self.__public_linear_operation(plain_text, sub)

    def __private_sub(self, secret):
        return self.__private_linear_operation(secret, sub)

    __sub__ = sub

    def rsub(self, value):
        return (self - value) * -1

    __rsub__ = rsub

    def mul(self, value):
        return self.__switch_public_private(value, self.__public_mul, self.__private_mul)

    def __public_mul(self, plain_text):
        return self.__public_multiplication_operation(plain_text, mul)

    def __private_mul(self, secret):
        return self.__private_multiplication_operation(secret, mul)

    __mul__ = mul
    __rmul__ = mul

    def matmul(self, value):
        return self.__switch_public_private(value, self.__public_matmul, self.__private_matmul)

    def __public_matmul(self, plain_text):
        return self.__public_multiplication_operation(plain_text, torch.matmul)

    def __private_matmul(self, secret):
        return self.__private_multiplication_operation(secret, torch.matmul)

    __matmul__ = matmul

    def view(self, *args, **kwargs):
        return self.__apply_to_shares(torch.Tensor.view, *args, *kwargs)

    def unfold(self, kernel_size, padding=0):
        image = self
        image = image.__apply_to_shares(torch.Tensor.double)
        image = image.__apply_to_shares(
            torch.nn.functional.unfold, kernel_size=kernel_size, padding=padding
        )
        image = image.__apply_to_shares(torch.Tensor.long)
        return image

    @staticmethod
    def __switch_public_private(value, public_function, private_function, *args, **kwargs):
        if isinstance(value, (int, float, torch.Tensor, syft.FixedPrecisionTensor)):
            return public_function(value, *args, **kwargs)
        elif isinstance(value, syft.ReplicatedSharingTensor):
            return private_function(value, *args, **kwargs)
        else:
            raise ValueError(
                "expected int, float, torch tensor, or ReplicatedSharingTensor "
                "but got {}".format(type(value))
            )

    def __public_linear_operation(self, plain_text, operator):
        players = self.__get_players()
        shares_map = self.__get_shares_map().copy()
        plain_text = torch.tensor(plain_text, dtype=torch.long)
        remote_plain_text = [plain_text.send(players[0]), plain_text.send(players[-1])]
        shares_map[players[0]] = (
            operator(shares_map[players[0]][0], remote_plain_text[0]),
            shares_map[players[0]][1],
        )
        shares_map[players[-1]] = (
            shares_map[players[-1]][0],
            operator(shares_map[players[-1]][1], remote_plain_text[-1]),
        )
        return ReplicatedSharingTensor().__set_shares_map(shares_map)

    def __private_linear_operation(self, secret, operator):
        x, y = self.__get_shares_map(), secret.__get_shares_map()
        players = self.__get_players()
        z = {
            player: (operator(x[player][0], y[player][0]), operator(x[player][1], y[player][1]))
            for player in players
        }
        return ReplicatedSharingTensor().__set_shares_map(z)

    def __public_multiplication_operation(self, plain_text, operator):
        players = self.__get_players()
        plain_text_map = {
            player: torch.tensor(plain_text, dtype=torch.long).send(player) for player in players
        }
        shares_map = self.__get_shares_map().copy()
        for player in players:
            shares_map[player] = (
                operator(shares_map[player][0], plain_text_map[player]),
                operator(shares_map[player][1], plain_text_map[player]),
            )
        return ReplicatedSharingTensor().__set_shares_map(shares_map)

    def __private_multiplication_operation(self, secret, operator):
        x, y = self.__get_shares_map(), secret.__get_shares_map()
        players = self.__get_players()
        z = [
            operator(x[player][0], y[player][0])
            + operator(x[player][1], y[player][0])
            + operator(x[player][0], y[player][1])
            for player in players
        ]
        z = self.__add_noise(z)
        z = self.__reshare(z, players)
        return ReplicatedSharingTensor().__set_shares_map(z)

    @staticmethod
    def __add_noise(shares):
        noisy_shares = [share + gen_alpha_3of3(share.location).wrap() for share in shares]
        return noisy_shares

    @staticmethod
    def __reshare(shares, workers):
        """convert 3-out-of-3 secret sharing: {player i : share i}
        to 2-out-of-3 sharing: {player i : (share i, share i+1)}"""
        shares_map = {}
        for i in range(len(shares)):
            pointer = shares[(i + 1) % len(shares)].copy().move(workers[i])
            shares_map[workers[i]] = (shares[i], pointer)
        return shares_map

    def __apply_to_shares(self, function, *args, **kwargs):
        """
        function: a reference to a function
        e.g. torch.Tensor.view, torch.nn.F.fold
        """
        shares_map = self.__get_shares_map()
        players = self.__get_players()
        shares_map = {
            player: (
                function(shares_map[player][0], *args, **kwargs),
                function(shares_map[player][1], *args, **kwargs),
            )
            for player in players
        }
        return ReplicatedSharingTensor().__set_shares_map(shares_map)

    def __get_players(self):
        return list(self.__get_shares_map().keys())

    def __get_shares_map(self):
        return self.child

    def __set_shares_map(self, shares_map):
        self.child = shares_map
        return self

    @property
    def shape(self):
        return self.retrieve_pointers()[0].shape

    @property
    def players(self):
        return self.__get_players()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        type_name = type(self).__name__
        out = f"[" f"{type_name}]"
        if self.child is not None:
            for v in self.child.values():
                out += "\n\t-> " + str(v)
        return out
