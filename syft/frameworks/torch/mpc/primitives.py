from collections import defaultdict
from typing import List

import numpy as np
import torch as th
import syft as sy
from syft.exceptions import EmptyCryptoPrimitiveStoreError
from syft.frameworks.torch.mpc.beaver import build_triple
from syft.workers.abstract import AbstractWorker


class PrimitiveStorage:
    """
    Used by normal workers to store crypto primitives
    Used by crypto providers to build crypto primitives
    """

    _known_components = {}

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
        self.mul: dict = defaultdict(list)
        self.matmul: dict = defaultdict(list)

        self._owner: AbstractWorker = owner
        self._builders: dict = {
            "fss_eq": self.build_fss_keys(op="eq"),
            "fss_comp": self.build_fss_keys(op="comp"),
            "mul": self.build_triples(op="mul"),
            "matmul": self.build_triples(op="matmul"),
        }

        self.force_preprocessing = False
        for name, component in PrimitiveStorage._known_components.items():
            setattr(self, name, component())

    @staticmethod
    def register_component(name, cls):
        PrimitiveStorage._known_components[name] = cls

    def get_keys(self, op: str, n_instances: int = 1, remove: bool = True, **kwargs):
        """
        Return keys primitives

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

        if op in {"mul", "matmul"}:
            shapes = kwargs.get("shapes")
            dtype = kwargs.get("dtype")
            torch_dtype = kwargs.get("torch_dtype")
            field = kwargs.get("field")
            config = (shapes, dtype, torch_dtype, field)
            primitive_stack = primitive_stack[config]
            available_instances = len(primitive_stack[0]) if len(primitive_stack) > 0 else -1
            if available_instances >= n_instances:
                keys = []
                for i, prim in enumerate(primitive_stack):
                    if n_instances == 1:
                        keys.append(prim[0])
                        if remove:
                            primitive_stack[i] = prim[1:]
                    else:
                        keys.append(prim[:n_instances])
                        if remove:
                            primitive_stack[i] = prim[n_instances:]
                return keys
            else:
                if self._owner.verbose:
                    print(
                        f"Autogenerate: "
                        f'"{op}", '
                        f"[({str(tuple(shapes[0]))}, {str(tuple(shapes[1]))})], "
                        f"n_instances={n_instances}"
                    )
                raise EmptyCryptoPrimitiveStoreError(
                    self,
                    available_instances=available_instances,
                    n_instances=n_instances,
                    op=op,
                    **kwargs,
                )
        elif op in {"fss_eq", "fss_comp"}:
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
                    self,
                    available_instances=available_instances,
                    n_instances=n_instances,
                    op=op,
                    **kwargs,
                )

    def provide_primitives(
        self,
        op: str,
        workers: List[AbstractWorker],
        n_instances: int = 10,
        **kwargs,
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
            if op in {"mul", "matmul"}:
                for params, primitive_triple in primitives:
                    if params not in current_primitives or len(current_primitives[params]) == 0:
                        current_primitives[params] = primitive_triple
                    else:
                        for i, primitive in enumerate(primitive_triple):
                            current_primitives[params][i] = th.cat(
                                (current_primitives[params][i], primitive)
                            )
            elif op in {"fss_eq", "fss_comp"}:
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

        n = sy.frameworks.torch.mpc.fss.n

        def build_separate_fss_keys(n_party: int, n_instances: int = 100):
            assert (
                n_party == 2
            ), f"The FSS protocol only works for 2 workers, {n_party} were provided."
            alpha, s_00, s_01, *CW = sy.frameworks.torch.mpc.fss.keygen(n_values=n_instances, op=op)
            # simulate sharing TODO clean this
            mask = np.random.randint(0, 2 ** n, alpha.shape, dtype=alpha.dtype)
            return [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

        return build_separate_fss_keys

    def build_triples(self, op: str):
        """
        The builder to generate beaver triple for multiplication or matrix multiplication
        """

        def build_separate_triples(n_party: int, n_instances: int, **kwargs) -> list:
            assert n_party == 2, (
                "Only 2 workers supported for the moment. "
                "Please fill an issue if you have an urgent need."
            )
            shapes = kwargs["shapes"]  # should be a list of pairs of shapes
            if not isinstance(shapes, list):
                # if shapes was not given a list, we check that it is a pair of two shapes,
                # the one of x and y
                assert len(shapes) == 2
                shapes = [shapes]

            # get params and set default values
            dtype = kwargs.get("dtype", "long")
            torch_dtype = kwargs.get("torch_dtype", th.int64)
            field = kwargs.get("field", 2 ** 64)

            primitives_worker = [[] for _ in range(n_party)]
            for shape in shapes:
                shares_worker = build_triple(op, shape, n_party, n_instances, torch_dtype, field)
                config = (shape, dtype, torch_dtype, field)
                for primitives, shares in zip(primitives_worker, shares_worker):
                    primitives.append((config, shares))

            return primitives_worker

        return build_separate_triples
