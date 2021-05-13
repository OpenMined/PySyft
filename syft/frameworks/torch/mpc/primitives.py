from collections import defaultdict
from typing import List

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
        self.conv2d: dict = defaultdict(list)
        self.conv_transpose2d: dict = defaultdict(list)

        self._owner: AbstractWorker = owner
        self._builders: dict = {
            "fss_eq": self.build_fss_keys(op="eq"),
            "fss_comp": self.build_fss_keys(op="comp"),
            "mul": self.build_triples(op="mul"),
            "matmul": self.build_triples(op="matmul"),
            "conv2d": self.build_triples(op="conv2d"),
            "conv_transpose2d": self.build_triples(op="conv_transpose2d"),
        }

        self.force_preprocessing = False
        for name, component in PrimitiveStorage._known_components.items():
            setattr(self, name, component())

    @staticmethod
    def register_component(name, cls):
        PrimitiveStorage._known_components[name] = cls

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
        if op in {"mul", "matmul", "conv2d", "conv_transpose2d"}:
            assert n_instances == 1
            shapes = kwargs.get("shapes")
            dtype = kwargs.get("dtype")
            torch_dtype = str(kwargs.get("torch_dtype"))
            field = kwargs.get("field")
            kwargs_ = kwargs.get("kwargs_")
            hashable_kwargs = {k: v for k, v in kwargs_.items() if k != "bias"}
            hashable_kwargs_ = tuple(hashable_kwargs.keys()), tuple(hashable_kwargs.values())
            if op in {"conv2d", "conv_transpose2d"}:

                config = (shapes, dtype, torch_dtype, field, hashable_kwargs_)
            else:
                config = (shapes, dtype, torch_dtype, field)
            primitive_stack = primitive_stack[config]
            available_instances = len(primitive_stack[0]) if len(primitive_stack) > 0 else -1
            if available_instances > 0:
                keys = primitive_stack[0]
                if remove:
                    del primitive_stack[0]
                return keys
            else:
                if self._owner.verbose:
                    print(
                        f"Autogenerate: "
                        f'"{op}", '
                        f"[({str(tuple(shapes[0]))}, {str(tuple(shapes[1]))})], "
                        f"n_instances={n_instances}"
                    )

                if op in {"conv2d", "conv_transpose2d"}:
                    sy.preprocessed_material[op].append(
                        (tuple(shapes[0]), tuple(shapes[1]), hashable_kwargs_)
                    )
                else:
                    sy.preprocessed_material[op].append((tuple(shapes[0]), tuple(shapes[1])))

                raise EmptyCryptoPrimitiveStoreError(
                    self, available_instances, n_instances=n_instances, op=op, **kwargs
                )
        elif op in {"fss_eq", "fss_comp"}:
            if th.cuda.is_available():
                # print('opening store...')
                available_instances = len(primitive_stack[0][0]) if len(primitive_stack) > 0 else -1
                # print('available_instances', available_instances)
                # print(primitive_stack)
            else:
                # The primitive stack is a list of keys arrays (2d numpy u8 arrays).
                available_instances = len(primitive_stack[0]) if len(primitive_stack) > 0 else -1

            if available_instances >= n_instances:
                if th.cuda.is_available():
                    assert available_instances == n_instances
                    keys = primitive_stack[0]
                else:
                    keys = primitive_stack[0][0:n_instances]
                if remove:
                    # We throw the whole key array away, not just the keys we used
                    del primitive_stack[0]
                return keys
            else:
                if self._owner.verbose:
                    print(
                        f"Autogenerate: "
                        f'"{op}", '
                        f"[{', '.join(c.id for c in sy.local_worker.clients)}], "
                        f"n_instances={n_instances}"
                    )

                sy.preprocessed_material[op].append(n_instances)

                raise EmptyCryptoPrimitiveStoreError(
                    self, available_instances, n_instances=n_instances, op=op, **kwargs
                )

    def provide_primitives(
        self,
        op: str,
        kwargs_: dict,
        workers: List[AbstractWorker],
        n_instances: int = 10,
        **kwargs,
    ):
        """Build n_instances of crypto primitives of the different crypto_types given and
        send them to some workers.

        Args:
            op (str): type of primitive (fss_eq, etc)
            workers (AbstractWorker): recipients for those primitives
            n_instances (int): how many of them are needed
            **kwargs: any parameters needed for the primitive builder
        """
        if not isinstance(op, str):
            raise TypeError("op should be a string")

        worker_types_primitives = defaultdict(dict)

        builder = self._builders[op]

        primitives = builder(
            kwargs_=kwargs_, n_party=len(workers), n_instances=n_instances, **kwargs
        )

        for worker_primitives, worker in zip(primitives, workers):
            worker_types_primitives[worker][op] = worker_primitives

        for i, worker in enumerate(workers):
            worker_message = self._owner.create_worker_command_message(
                "feed_crypto_primitive_store", None, worker_types_primitives[worker]
            )
            self._owner.send_msg_arrow(worker_message, worker)

    def add_primitives(self, types_primitives: dict):
        """
        Include primitives in the store

        Args:
            types_primitives: dict {op: str: primitives: list}
        """
        for op, primitives in types_primitives.items():
            if not hasattr(self, op):
                raise ValueError(f"Unknown crypto primitives {op}")

            current_primitives = getattr(self, op)
            if op in {"mul", "matmul", "conv2d", "conv_transpose2d"}:
                for params, primitive_triple in primitives:
                    if th.cuda.is_available():
                        primitive_triple = [p.cuda() for p in primitive_triple]
                    if params not in current_primitives or len(current_primitives[params]) == 0:
                        current_primitives[params] = [primitive_triple]
                    else:
                        current_primitives[params].append(primitive_triple)
            elif op in {"fss_eq", "fss_comp"}:
                if th.cuda.is_available():
                    primitives = [
                        p.cuda() if not isinstance(p, tuple) else tuple(pi.cuda() for pi in p)
                        for p in primitives
                    ]
                if len(current_primitives) == 0 or len(current_primitives[0]) == 0:
                    setattr(self, op, [primitives])
                else:
                    # This branch never happens with on-the-fly primitives
                    current_primitives.append(primitives)
            else:
                raise TypeError(f"Can't resolve primitive {op} to a framework")

    def build_fss_keys(self, op: str):
        """
        The builder to generate functional keys for Function Secret Sharing (FSS)
        """
        if op == "eq":
            if th.cuda.is_available():
                fss_class = sy.frameworks.torch.mpc.cuda.fss.DPF
            else:
                fss_class = sy.frameworks.torch.mpc.fss.DPF
        elif op == "comp":
            if th.cuda.is_available():
                fss_class = sy.frameworks.torch.mpc.cuda.fss.DIF
            else:
                fss_class = sy.frameworks.torch.mpc.fss.DIF
        else:
            raise ValueError(f"type_op {op} not valid")

        n = sy.frameworks.torch.mpc.fss.n

        def build_separate_fss_keys(kwargs_: dict, n_party: int, n_instances: int = 100):
            if n_party != 2:
                raise AttributeError(
                    f"The FSS protocol only works for 2 workers, " f"{n_party} were provided."
                )
            if th.cuda.is_available():
                alpha, s_00, s_01, *CW = sy.frameworks.torch.mpc.cuda.fss.keygen(
                    n_values=n_instances, op=op
                )
                # simulate sharing TODO clean this
                mask = th.randint(0, 2 ** n, alpha.shape, device="cuda")
                return [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]
            else:
                keys_a, keys_b = fss_class.keygen(n_values=n_instances)
                return [keys_a, keys_b]

        return build_separate_fss_keys

    def build_triples(self, op: str):
        """
        The builder to generate beaver triple for multiplication or matrix multiplication
        """

        def build_separate_triples(kwargs_: dict, n_party: int, n_instances: int, **kwargs) -> list:
            assert n_instances == 1, "For Beaver, only n_instances == 1 is allowed."
            if n_party != 2:
                raise NotImplementedError(
                    "Only 2 workers supported for the moment. "
                    "Please fill an issue if you have an urgent need."
                )
            shapes = kwargs["shapes"]  # should be a list of pairs of shapes
            if not isinstance(shapes, list):
                # if shapes was not given a list, we check that it is a pair of two shapes,
                # the one of x and y
                if len(shapes) != 2:
                    raise ValueError(
                        "if shapes was not given a list, we check that it is a pair of two shapes, "
                        "the one of x and y"
                    )
                shapes = [shapes]

            # get params and set default values
            dtype = kwargs.get("dtype", "long")
            torch_dtype = kwargs.get("torch_dtype", th.int64)
            field = kwargs.get("field", 2 ** 64)

            if op in {"conv2d", "conv_transpose2d"}:
                hashable_kwargs = {k: v for k, v in kwargs_.items() if k != "bias"}
                hashable_kwargs_ = tuple(hashable_kwargs.keys()), tuple(hashable_kwargs.values())

            primitives_worker = [[] for _ in range(n_party)]
            for shape in shapes:
                shares_worker = build_triple(op, kwargs_, shape, n_party, torch_dtype, field)
                shape = (tuple(shape[0]), tuple(shape[1]))
                torch_dtype = str(torch_dtype)

                if op in {"conv2d", "conv_transpose2d"}:
                    config = (shape, dtype, torch_dtype, field, hashable_kwargs_)
                else:
                    config = (shape, dtype, torch_dtype, field)

                for primitives, shares in zip(primitives_worker, shares_worker):
                    primitives.append((config, shares))

            return primitives_worker

        return build_separate_triples
