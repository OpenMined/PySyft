from operator import add, sub, mul
import torch
from syft.generic.frameworks.hook import hook_args
import syft
from syft.generic.abstract.tensor import AbstractTensor
from syft.workers.abstract import AbstractWorker
from syft.frameworks.torch.mpc.przs import PRZS, gen_alpha_3of3


class ReplicatedSharingTensor(AbstractTensor):
    def __init__(
        self,
        plain_text=None,
        players=None,
        ring_size=None,
        owner=None,
        id=None,
        tags=None,
        description=None,
        shares=None,
    ):
        super().__init__(owner=owner, id=id, tags=tags, description=description)
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
            pointer1 = shares[i].copy().send(workers[i])
            pointer2 = shares[(i + 1) % len(shares)].copy().send(workers[i])
            shares_map[workers[i]] = (pointer1, pointer2)
        return shares_map

    def reconstruct(self):
        shares = self.retrieve_shares()
        plain_text_mod = self.__sum_shares(shares)
        plain_text = self.__map_modular_to_real(plain_text_mod)
        return plain_text

    get = reconstruct

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

    def set_garbage_collect_data(self, value):
        shares = self.child

        for _, shares in shares.items():
            for share in shares:
                assert share.is_wrapper is True
                share.gc = value

    def get_garbage_collect_data(self):
        shares = self.child
        res = None

        """ Select the first share """
        ref_share = next(iter(shares.values()))[0]

        is_wrapper = ref_share.is_wrapper
        if is_wrapper:
            ref_share = ref_share.child

        gc_ref = ref_share.garbage_collect_data

        for worker, shares in shares.items():
            for share in shares:
                assert share.is_wrapper == is_wrapper

                if share.is_wrapper:
                    share = share.child

                """ Make sure the GC value is the same for all shares """
                assert share.garbage_collect_data == gc_ref

        return gc_ref

    @property
    def grad(self):
        """
        Gradient makes no sense for Replicated Shared Tensor
        We make it clear that if someone query .grad on a Replicated Shared Tensor it would
        not throw an error
        Return None such that it can not be set
        """
        return None

    def backward(self, *args, **kwargs):
        """Calling backward on Replicated Shared Tensor doesn't make sense, but sometimes a call
        can be propagated downward the chain to an RST (for example in create_grad_objects), so
        we just ignore the call."""
        pass

    @staticmethod
    def simplify(worker: AbstractWorker, tensor: "ReplicatedSharingTensor") -> tuple:
        """
        This function takes the attributes of a ReplicatedSharingTensor and saves them in a tuple
        Args:
            worker (AbstractWorker): the worker that does the serialization
            tensor (ReplicatedSharingTensor): a ReplicatedSharingTensor
        Returns:
            tuple: a tuple holding the unique attributes of the replicated shared tensor
        Examples:
            data = simplify(tensor)
        """
        _simplify = lambda x: syft.serde.msgpack.serde._simplify(worker, x)

        # Don't delete the remote values of the shares at simplification
        garbage_collect = tensor.get_garbage_collect_data()
        # We should always have wrappers
        prep_simplify = list(tensor.child.values())
        chain = _simplify(prep_simplify)

        tensor.set_garbage_collect_data(False)

        return (
            _simplify(tensor.id),
            _simplify(tensor.ring_size),
            chain,
            garbage_collect,
        )

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "ReplicatedSharingTensor":
        """
            This function reconstructs a ReplicatedSharingTensor given it's attributes in
        form of a tuple.
        Args:
            worker: the worker doing the deserialization
            tensor_tuple: a tuple holding the attributes of the ReplicatedSharingTensor
        Returns:
            ReplicatedSharingTensor: a ReplicatedSharingTensor
        Examples:
            shared_tensor = detail(data)
        """
        _detail = lambda x: syft.serde.msgpack.serde._detail(worker, x)

        tensor_id, ring_size, chain, garbage_collect = tensor_tuple

        tensor = ReplicatedSharingTensor(
            owner=worker,
            id=_detail(tensor_id),
            ring_size=_detail(ring_size),
        )

        chain = _detail(chain)
        tensor.child = {}
        for shares in chain:
            ref_loc = shares[0].location
            ref_owner = shares[0].owner

            for share in shares:
                assert share.location == ref_loc
                assert share.owner == ref_owner

            if ref_loc is not None:
                # Remote
                worker = ref_loc
            else:
                # Local
                worker = ref_owner.id

            tensor.child[worker] = shares

        tensor.set_garbage_collect_data(garbage_collect)
        return tensor

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        type_name = type(self).__name__
        out = f"[" f"{type_name}]"
        if self.child is not None:
            for v in self.child.values():
                out += "\n\t-> " + str(v)
        return out


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(ReplicatedSharingTensor)
