from collections import defaultdict
from typing import List, Tuple, Union

import torch as th
import syft as sy
from syft.workers.abstract import AbstractWorker


class PrimitiveStorage:
    """
    Used by normal workers to store crypto primitives
    Used by crypto providers to build crypto primitives
    """

    def __init__(self, owner):
        # The different kinds of primitives available
        self.fss_eq: list = []
        self.fss_comp: list = []
        self.beaver: list = []
        self.xor_add_couple: list = []  # couple of the same value shared via ^ or + op

        self._owner: AbstractWorker = owner
        self._builders: dict = {
            "fss_eq": self.build_fss_keys(type_op="eq"),
            "fss_comp": self.build_fss_keys(type_op="comp"),
            "beaver": self.build_triples,
            "xor_add_couple": self.build_xor_add_couple,
        }

    def provide_primitives(
        self,
        crypto_types: Union[str, List[str]],
        workers: List[AbstractWorker],
        n_instances: int = 10,
        **kwargs,
    ):
        """
        Build n_instances of crypto primitives of the different crypto_types given and send them to some workers.

        Args:
            crypto_types: type of primitive (fss_eq, etc)
            workers: recipients for those primitive
            n_instances: how many of them are needed
            **kwargs: any parameters needs for the primitive builder

        Returns:

        """
        if isinstance(crypto_types, str):
            crypto_types = [crypto_types]

        worker_types_primitives = defaultdict(dict)
        for crypto_type in crypto_types:
            builder = self._builders[crypto_type]

            primitives = []
            for i in range(n_instances):
                primitive_instance: Tuple[Tuple] = builder(n_party=len(workers), **kwargs)
                primitives.append(primitive_instance)

            for i, worker in enumerate(workers):
                worker_types_primitives[worker][crypto_type] = [
                    primitive[i] for primitive in primitives
                ]

        for i, worker in enumerate(workers):
            worker_message = self._owner.create_worker_command_message(
                "feed_crypto_primitive_store", None, worker_types_primitives[worker]
            )
            self._owner.send_msg(worker_message, worker)

    def add_primitives(self, types_primitives: dict):
        """
        Include primitives in the store

        Args:
            types_primitives: dict {crypto_type: str: primitives: list}
        """
        for crypto_type, primitives in types_primitives.items():
            assert hasattr(self, crypto_type), f"Unknown crypto primitives {crypto_type}"

            current_primitives = getattr(self, crypto_type)
            current_primitives.extend(primitives)

    def build_fss_keys(self, type_op):
        """
        The builder to generate functional keys for Function Secret Sharing (FSS)
        """
        if type_op == "eq":
            fss_class = sy.frameworks.torch.mpc.fss.DPF
        elif type_op == "comp":
            fss_class = sy.frameworks.torch.mpc.fss.DIF
        else:
            raise ValueError(f"type_op {type_op} not valid")

        n = sy.frameworks.torch.mpc.fss.n

        def build_separate_fss_keys(n_party):
            assert (
                n_party == 2
            ), f"The FSS protocol only works for 2 workers, {n_party} were provided."
            alpha, s_00, s_01, *CW = fss_class.keygen()
            # simulate sharing TODO clean this
            (mask,) = th.randint(0, 2 ** n, (1,))
            return [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

        return build_separate_fss_keys

    @staticmethod
    def build_xor_add_couple(n_party, nbits=1):
        assert (
            n_party == 2
        ), f"build_xor_add_couple is only implemented for 2 workers, {n_party} were provided."
        r = th.randint(2, size=(nbits,))
        mask1 = th.randint(2, size=(nbits,))
        mask2 = th.randint(2, size=(nbits,))

        return [(r ^ mask1, r - mask2), (mask1, mask2)]

    def build_triples(self, **kwargs):
        raise NotImplementedError
