from collections import defaultdict
from typing import List

import numpy as np
import syft as sy
from syft.exceptions import EmptyCryptoPrimitiveStoreError
from syft.workers.abstract import AbstractWorker


class PrimitiveStorage:
    """
    Used by normal workers to store crypto primitives
    Used by crypto providers to build crypto primitives
    """

    def __init__(self, owner: AbstractWorker):
        """
        Their are below different kinds of primitives available.
        Each primitive stack is a fixed length list corresponding to all the
        components for the primitive. For example, the beaver triple primitive would
        have 3 components. Each component is a high dimensional tensor whose
        last dimension is the same and corresponds to the number of instances available
        for this primitive. That's why get_keys uses a quite complicated dimension
        selector. Those structures help generating efficiently primitives using
        tensorized key generation algorithms.
        """
        self.fss_eq: list = []
        self.fss_comp: list = []

        self._owner: AbstractWorker = owner
        self._builders: dict = {
            "fss_eq": self.build_fss_keys(op="eq"),
            "fss_comp": self.build_fss_keys(op="comp"),
        }

        self.force_preprocessing = False

    def get_keys(self, op: str, n_instances: int = 1, remove: bool = True, **kwargs):
        """
        Return FSS keys primitives

        Args:
            op (str): primitive type, should be fss_eq, fss_comp, mul or matmul
            n_instances (int): how many primitives to retrieve. Comparison is pointwise so this is
                convenient: for any matrice of size nxm I can unstack n*m elements for the
                comparison
            remove (boolean): if true, pop out the primitive. If false, only read it. Read mode is
                needed because we're working on virtual workers and they need to gather
                a some point and then re-access the keys.
            kwargs (dict): further arguments to be used depending of the primitive
        """
        primitive_stack = getattr(self, op)

        if op in {"fss_eq", "fss_comp"}:
            available_instances = len(primitive_stack[0]) if len(primitive_stack) > 0 else -1
            if available_instances >= n_instances:
                keys = []
                # We iterate on the different elements that constitute a given primitive, for
                # example of the beaver triples, you would have 3 elements.
                for i, prim in enumerate(primitive_stack):
                    # We're selecting on the last dimension of the tensor because it's simpler for
                    # generating those primitives in crypto protocols
                    # [:] ~ [slice(None)]
                    # [:1] ~ [slice(1)]
                    # [1:] ~ [slice(1, None)]
                    # [:, :, :1] ~ [slice(None)] * 2 + [slice(1)]
                    if isinstance(prim, tuple):

                        ps = []
                        left_ps = []
                        for p in prim:
                            n_dim = len(p.shape)
                            get_slice = tuple([slice(None)] * (n_dim - 1) + [slice(n_instances)])
                            remaining_slice = tuple(
                                [slice(None)] * (n_dim - 1) + [slice(n_instances, None)]
                            )
                            ps.append(p[get_slice])
                            if remove:
                                left_ps.append(p[remaining_slice])

                        keys.append(tuple(ps))
                        if remove:
                            primitive_stack[i] = tuple(left_ps)
                    else:
                        n_dim = len(prim.shape)
                        get_slice = tuple([slice(None)] * (n_dim - 1) + [slice(n_instances)])
                        remaining_slice = tuple(
                            [slice(None)] * (n_dim - 1) + [slice(n_instances, None)]
                        )

                        keys.append(prim[get_slice])
                        if remove:
                            primitive_stack[i] = prim[remaining_slice]

                return keys
            else:
                if self._owner.verbose:
                    print(
                        f"Autogenerate: "
                        f'"{op}", '
                        f"[{', '.join(c.id for c in sy.local_worker.clients)}], "
                        f"n_instances={n_instances}"
                    )
                raise EmptyCryptoPrimitiveStoreError(
                    self, available_instances, n_instances=n_instances, op=op, **kwargs
                )

    def provide_primitives(
        self, op: str, workers: List[AbstractWorker], n_instances: int = 10, **kwargs,
    ):
        """Build n_instances of crypto primitives of the different crypto_types given and
        send them to some workers.

        Args:
            op (str): type of primitive (fss_eq, etc)
            workers (AbstractWorker): recipients for those primitive
            n_instances (int): how many of them are needed
            **kwargs: any parameters needed for the primitive builder
        """
        assert isinstance(op, str)

        worker_types_primitives = defaultdict(dict)

        builder = self._builders[op]

        primitives = builder(n_party=len(workers), n_instances=n_instances, **kwargs)

        for worker_primitives, worker in zip(primitives, workers):
            worker_types_primitives[worker][op] = worker_primitives

        for i, worker in enumerate(workers):
            worker_message = self._owner.create_worker_command_message(
                "feed_crypto_primitive_store", None, worker_types_primitives[worker]
            )
            self._owner.send_msg(worker_message, worker)

    def add_primitives(self, types_primitives: dict):
        """
        Include primitives in the store

        Args:
            types_primitives: dict {op: str: primitives: list}
        """
        for op, primitives in types_primitives.items():
            assert hasattr(self, op), f"Unknown crypto primitives {op}"

            current_primitives = getattr(self, op)
            if op in ("fss_eq", "fss_comp"):
                if len(current_primitives) == 0:
                    setattr(self, op, list(primitives))
                else:
                    for i, primitive in enumerate(primitives):
                        if len(current_primitives[i]) == 0:
                            current_primitives[i] = primitive
                        else:
                            if isinstance(current_primitives[i], tuple):
                                new_prims = []
                                for cur_prim, prim in zip(current_primitives[i], primitive):
                                    new_prims.append(
                                        np.concatenate((cur_prim, prim), axis=len(prim.shape) - 1)
                                    )
                                current_primitives[i] = tuple(new_prims)
                            else:
                                current_primitives[i] = np.concatenate(
                                    (current_primitives[i], primitive),
                                    axis=len(primitive.shape) - 1,
                                )
            else:
                raise TypeError(f"Can't resolve primitive {op} to a framework")

    def build_fss_keys(self, op: str):
        """
        The builder to generate functional keys for Function Secret Sharing (FSS)
        """
        if op == "eq":
            fss_class = sy.frameworks.torch.mpc.fss.DPF
        elif op == "comp":
            fss_class = sy.frameworks.torch.mpc.fss.DIF
        else:
            raise ValueError(f"type_op {op} not valid")

        n = sy.frameworks.torch.mpc.fss.n

        def build_separate_fss_keys(n_party: int, n_instances: int = 100):
            assert (
                n_party == 2
            ), f"The FSS protocol only works for 2 workers, {n_party} were provided."
            alpha, s_00, s_01, *CW = fss_class.keygen(n_values=n_instances)
            # simulate sharing TODO clean this
            mask = np.random.randint(0, 2 ** n, alpha.shape, dtype=alpha.dtype)
            return [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

        return build_separate_fss_keys
