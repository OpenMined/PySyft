from collections import defaultdict
from typing import List, Union

import torch as th
import syft as sy
from syft.exceptions import EmptyCryptoPrimitiveStoreError
from syft.workers.abstract import AbstractWorker


class PrimitiveStorage:
    """
    Used by normal workers to store crypto primitives
    Used by crypto providers to build crypto primitives
    """

    def __init__(self, owner):
        """
        Their are below different kinds of primitives available.
        Each primitive stack is a fixed length list corresponding to all the
        components for the primitive. For example, the beaver triple primitive would
        have 3 components. Each component is a high dimensional tensor whose
        last dimension is the same and corresponds to the number of instances available
        for this primitive. That's why get_keys uses a quite complicated dimension
        selector. This structure helps generating efficiently primitives using
        tensorized key generation algorithms.
        """
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

    def get_keys(self, type_op, n_instances=1, remove=True):
        """
        Return FSS keys primitives

        Args:
            type_op: fss_eq, fss_comp, or xor_add_couple
            n_instances: how many primitives to retrieve. Comparison is pointwise so this is
                convenient: for any matrice of size nxm I can unstack n*m elements for the
                comparison
            remove: if true, pop out the primitive. If false, only read it. Read mode is
                needed because we're working on virtual workers and they need to gather
                a some point and then re-access the keys.
        """
        primitive_stack = getattr(self, type_op)

        available_instances = len(primitive_stack[0]) if len(primitive_stack) > 0 else -1
        if available_instances >= n_instances:
            keys = []
            # We iterate on the different elements that constitute a given primitive, for
            # example of the beaver triples, you would have 3 elements.
            for i, prim in enumerate(primitive_stack):
                # We're selecting on the last dimension of the tensor because it's simpler for
                # generating those primitives in crypto protocols
                # th.narrow(dim, index_start, length)
                keys.append(th.narrow(prim, -1, 0, n_instances))
                if remove:
                    length = prim.shape[-1] - n_instances
                    primitive_stack[i] = th.narrow(prim, -1, n_instances, length)

            return keys
        else:
            raise EmptyCryptoPrimitiveStoreError(self, type_op, available_instances, n_instances)

    def provide_primitives(
        self,
        crypto_types: Union[str, List[str]],
        workers: List[AbstractWorker],
        n_instances: int = 10,
        **kwargs,
    ):
        """ Build n_instances of crypto primitives of the different crypto_types given and
        send them to some workers.

        Args:
            crypto_types: type of primitive (fss_eq, etc)
            workers: recipients for those primitive
            n_instances: how many of them are needed
            **kwargs: any parameters needs for the primitive builder
        """
        if isinstance(crypto_types, str):
            crypto_types = [crypto_types]

        worker_types_primitives = defaultdict(dict)
        for crypto_type in crypto_types:
            builder = self._builders[crypto_type]

            primitives = builder(n_party=len(workers), n_instances=n_instances, **kwargs)

            for worker_primitives, worker in zip(primitives, workers):
                worker_types_primitives[worker][crypto_type] = worker_primitives

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
            if len(current_primitives) == 0:
                setattr(self, crypto_type, list(primitives))
            else:
                for i, primitive in enumerate(primitives):
                    if len(current_primitives[i]) == 0:
                        current_primitives[i] = primitive
                    else:
                        current_primitives[i] = th.cat(
                            (current_primitives[i], primitive), dim=len(primitive.shape) - 1
                        )

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

        def build_separate_fss_keys(n_party, n_instances=100):
            assert (
                n_party == 2
            ), f"The FSS protocol only works for 2 workers, {n_party} were provided."
            alpha, s_00, s_01, *CW = fss_class.keygen(n_values=n_instances)
            # simulate sharing TODO clean this
            mask = th.randint(0, 2 ** n, alpha.shape)
            return [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

        return build_separate_fss_keys

    @staticmethod
    def build_xor_add_couple(n_party, n_instances=100):
        assert (
            n_party == 2
        ), f"build_xor_add_couple is only implemented for 2 workers, {n_party} were provided."
        r = th.randint(2, size=(n_instances,))
        mask1 = th.randint(2, size=(n_instances,))
        mask2 = th.randint(2, size=(n_instances,))

        return [(r ^ mask1, r - mask2), (mask1, mask2)]

    def build_triples(self, **kwargs):
        raise NotImplementedError
