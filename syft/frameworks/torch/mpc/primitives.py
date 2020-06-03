from collections import defaultdict
from typing import List, Tuple, Union
import math

import numpy as np
import torch as th
import syft as sy
from syft.exceptions import EmptyCryptoPrimitiveStoreError
from syft.workers.abstract import AbstractWorker

import os.path


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
        selector. This structures helps generating efficiently primitives using
        tensorized key generation algorithms.
        """
        self.fss_eq: list = []
        self.fss_comp: list = []
        self.beaver: list = defaultdict(list)

        self._owner: AbstractWorker = owner
        self._builders: dict = {
            "fss_eq": self.build_fss_keys(type_op="eq"),
            "fss_comp": self.build_fss_keys(type_op="comp"),
            "beaver": self.build_triples,
        }

        self.force_preprocessing = False

    def get_keys(self, type_op, n_instances=1, remove=True, **kwargs):
        """
        Return FSS keys primitives #TODO

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

        if type_op == "beaver":
            op = kwargs.get("op")
            shapes = kwargs.get("shapes")
            op_shapes = (op, *shapes)
            primitive_stack = primitive_stack[op_shapes]
            available_instances = len(primitive_stack[0]) if len(primitive_stack) > 0 else -1
            # print('requires:', n_instances, '\tavailable:', available_instances)
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
                if not self.force_preprocessing:
                    # print(
                    #     f"Autogenerate: "
                    #     f'["{type_op}"], '
                    #     # f"[{', '.join(c.id for c in sy.local_worker.clients)}], "
                    #     f"n_instances={n_instances}, "
                    #     'beaver={"op_shape": ['
                    #     f'("{op}", {str(tuple(shapes[0]))}, {str(tuple(shapes[1]))})'
                    #     "]}"
                    # )
                    # print(f"\t\t\t " f'("{op}", {str(tuple(shapes[0]))}, {str(tuple(shapes[1]))}),')
                    sy.local_worker.crypto_store.provide_primitives(
                        [type_op],
                        sy.local_worker.clients,
                        n_instances=n_instances,
                        beaver={"op_shapes": [op_shapes]},
                    )
                    return self.get_keys(type_op, n_instances=n_instances, remove=remove, **kwargs)
                else:
                    raise EmptyCryptoPrimitiveStoreError(
                        self, type_op, available_instances, n_instances, **kwargs
                    )
        else:
            available_instances = len(primitive_stack[0]) if len(primitive_stack) > 0 else -1
            # print('check enoought prim?', available_instances, 'neeeded', n_instances)
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
                if not self.force_preprocessing:
                    print(
                        f"Autogenerate: "
                        f'["{type_op}"], '
                        f"[{', '.join(c.id for c in sy.local_worker.clients)}], "
                        f"n_instances={n_instances}"
                    )
                    sy.local_worker.crypto_store.provide_primitives(
                        [type_op], sy.local_worker.clients, n_instances=n_instances
                    )
                    return self.get_keys(type_op, n_instances=n_instances, remove=remove, **kwargs)
                else:
                    raise EmptyCryptoPrimitiveStoreError(
                        self, type_op, available_instances, n_instances
                    )

    def provide_primitives(
        self, crypto_type: str, workers: List[AbstractWorker], n_instances: int = 10, **kwargs,
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
        if isinstance(crypto_type, list):
            crypto_type = crypto_type[0]

        while n_instances > 0:
            n_instances_batch = min(500_000, n_instances)
            if n_instances_batch > 10_000:
                #     n_instances_batch = 500_000
                n_instances_batch = math.ceil(n_instances_batch / 100_000) * 100_000
                # print('| n_instances_batch', n_instances_batch)
            worker_types_primitives = defaultdict(dict)

            path = "/Users/tryffel/code/PySyft/data/primitives"

            def filename(worker):
                if "beaver" in crypto_type:
                    op, *shapes = kwargs["beaver"]["op_shapes"][0]
                    shapes = [",".join([str(s) for s in shape]) for shape in shapes]
                    return f"{path}/{crypto_type}-{op}-({shapes[0]})-({shapes[1]})-{n_instances_batch}-{worker.id}.data"
                else:
                    return f"{path}/{crypto_type}-{n_instances_batch}-{worker.id}.data"

            if os.path.isfile(filename(workers[0]) + ".npy") and "beaver" not in crypto_type:
                # if "comp" in crypto_type:
                #     print(f"{n_instances_batch} from file")
                for i, worker in enumerate(workers):
                    worker_message = self._owner.create_worker_command_message(
                        "load_crypto_primitive", None, crypto_type, filename(worker)
                    )
                    self._owner.send_msg(worker_message, worker)
            else:
                if "comp" in crypto_type:
                    print(f"{n_instances_batch} building")
                builder = self._builders[crypto_type]

                primitives = builder(n_party=len(workers), n_instances=n_instances_batch, **kwargs)

                for worker_primitives, worker in zip(primitives, workers):
                    if "beaver" not in crypto_type:
                        np.save(filename(worker), worker_primitives)
                    # print('saved', filename(worker))
                    worker_types_primitives[worker][crypto_type] = worker_primitives

                for i, worker in enumerate(workers):
                    worker_message = self._owner.create_worker_command_message(
                        "feed_crypto_primitive_store", None, worker_types_primitives[worker]
                    )
                    self._owner.send_msg(worker_message, worker)

            n_instances -= n_instances_batch

    def add_primitives(self, types_primitives: dict):
        """
        Include primitives in the store

        Args:
            types_primitives: dict {crypto_type: str: primitives: list}
        """
        for crypto_type, primitives in types_primitives.items():
            assert hasattr(self, crypto_type), f"Unknown crypto primitives {crypto_type}"

            current_primitives = getattr(self, crypto_type)
            if crypto_type == "beaver":
                for op_shapes, primitive_triple in primitives:
                    if (
                        op_shapes not in current_primitives
                        or len(current_primitives[op_shapes]) == 0
                    ):
                        current_primitives[op_shapes] = primitive_triple
                    else:
                        for i, primitive in enumerate(primitive_triple):
                            current_primitives[op_shapes][i] = th.cat(
                                (current_primitives[op_shapes][i], primitive)
                            )
            elif crypto_type in ("fss_eq", "fss_comp"):
                if len(current_primitives) == 0:
                    setattr(self, crypto_type, list(primitives))
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
                raise TypeError(f"Can't resolve primitive {crypto_type} to a framework")

    def load_primitives(self, crypto_type, filename):
        """
        Load primitives in the store from file

        Args:
            types_primitives: dict {crypto_type: str: primitives: list}
        """
        primitives = np.load(filename + ".npy", allow_pickle=True)
        if len(primitives.shape) > 0:
            primitives = primitives.tolist()
        else:
            primitives = primitives.item()

        types_primitives = {crypto_type: primitives}

        self.add_primitives(types_primitives)

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
            mask = np.random.randint(0, 2 ** n, alpha.shape, dtype=alpha.dtype)
            return [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

        return build_separate_fss_keys

    def build_triples(self, n_party, n_instances, **kwargs):
        # TODO n -> field
        assert n_party == 2, f"Only 2 workers supported for the moment"
        n = sy.frameworks.torch.mpc.fss.n
        op_shapes = kwargs["beaver"]["op_shapes"]
        primitives_worker = [[], []]
        for op, a_shape, b_shape in op_shapes:
            cmd = getattr(th, op)
            a = th.randint(0, 2 ** n, (n_instances, *a_shape))
            b = th.randint(0, 2 ** n, (n_instances, *b_shape))
            # a = th.ones(*(n_instances, *a_shape)).long()
            # b = th.ones(*(n_instances, *b_shape)).long()

            if op == "mul" and b.numel() == a.numel():
                # examples:
                #   torch.tensor([3]) * torch.tensor(3) = tensor([9])
                #   torch.tensor([3]) * torch.tensor([[3]]) = tensor([[9]])
                if len(a.shape) == len(b.shape):
                    c = cmd(a, b)
                elif len(a.shape) > len(b.shape):
                    shape = b.shape
                    b = b.reshape_as(a)
                    c = cmd(a, b)
                    b = b.reshape(*shape)
                else:  # len(a.shape) < len(b.shape):
                    shape = a.shape
                    a = a.reshape_as(b)
                    c = cmd(a, b)
                    a = a.reshape(*shape)
            else:
                c = cmd(a, b)

            masks_0 = []
            masks_1 = []
            for i, tensor in enumerate([a, b, c]):
                mask = th.randint(0, 2 ** n, tensor.shape)
                masks_0.append(tensor - mask)
                masks_1.append(mask)

            primitives_worker[0].append(((op, a_shape, b_shape), masks_0))
            primitives_worker[1].append(((op, a_shape, b_shape), masks_1))

        return primitives_worker
