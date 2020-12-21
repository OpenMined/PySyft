"""
This is an implementation of Function Secret Sharing

Useful papers are:
- Function Secret Sharing- Improvements and Extensions, Boyle 2017
  Link: https://eprint.iacr.org/2018/707.pdf
- Secure Computation with Preprocessing via Function Secret Sharing, Boyle 2019
  Link: https://eprint.iacr.org/2019/1095

Note that the protocols are quite different in aspect from those papers
"""
import math
import numpy as np
import multiprocessing
import asyncio
import sycret

import torch as th
import syft as sy
from syft.exceptions import EmptyCryptoPrimitiveStoreError
from syft.workers.websocket_client import WebsocketClientWorker
from syft.generic.utils import allow_command
from syft.generic.utils import remote


λ = 127  # security parameter
n = 32  # bit precision
N = 4  # byte precision
λs = math.ceil(λ / 64)  # how many int64 are needed to store λ, here 2
if λs != 2:
    raise ValueError("Check the value of security parameter")

no_wrap = {"no_wrap": True}


def full_name(f):
    return f"syft.frameworks.torch.mpc.fss.{f.__name__}"


# internal codes
EQ = 0
COMP = 1

# number of processes
N_CORES = multiprocessing.cpu_count()
dpf = sycret.EqFactory(n_threads=N_CORES)
dif = sycret.LeFactory(n_threads=N_CORES)


def keygen(n_values, op):
    """
    Run FSS keygen in parallel to accelerate the offline part of the protocol

    Args:
        n_values (int): number of primitives to generate
        op (str): eq or comp <=> DPF or DIF
    """
    if op == "eq":
        return DPF.keygen(n_values=n_values)
    if op == "comp":
        return DIF.keygen(n_values=n_values)

    raise ValueError(f"{op} is an unsupported operation.")


def fss_op(x1, x2, op="eq"):
    """
    Define the workflow for a binary operation using Function Secret Sharing

    Currently supported operand are = & <=, respectively corresponding to
    op = 'eq' and 'comp'

    Args:
        x1: first AST
        x2: second AST
        op: type of operation to perform, should be 'eq' or 'comp'

    Returns:
        shares of the comparison
    """

    assert not th.cuda.is_available()

    if isinstance(x1, sy.AdditiveSharingTensor):
        locations = x1.locations
        class_attributes = x1.get_class_attributes()
    else:
        locations = x2.locations
        class_attributes = x2.get_class_attributes()

    dtype = class_attributes.get("dtype")
    asynchronous = isinstance(locations[0], WebsocketClientWorker)

    workers_args = [
        (
            x1.child[location.id]
            if isinstance(x1, sy.AdditiveSharingTensor)
            else (x1 if i == 0 else 0),
            x2.child[location.id]
            if isinstance(x2, sy.AdditiveSharingTensor)
            else (x2 if i == 0 else 0),
            op,
        )
        for i, location in enumerate(locations)
    ]

    try:
        shares = []
        for i, location in enumerate(locations):
            share = remote(mask_builder, location=location)(*workers_args[i], return_value=True)
            shares.append(share)
    except EmptyCryptoPrimitiveStoreError as e:
        if sy.local_worker.crypto_store.force_preprocessing:
            raise
        sy.local_worker.crypto_store.provide_primitives(workers=locations, kwargs_={}, **e.kwargs_)
        return fss_op(x1, x2, op)

    mask_value = sum(shares) % 2 ** n

    for location, share in zip(locations, shares):
        location.de_register_obj(share)
        del share

    workers_args = [
        (
            th.IntTensor([i]).cuda() if th.cuda.is_available() else th.IntTensor([i]),
            mask_value,
            op,
            dtype,
        )
        for i in range(2)
    ]
    if not asynchronous:
        shares = []
        for i, location in enumerate(locations):
            share = remote(evaluate, location=location)(*workers_args[i], return_value=False)
            shares.append(share)
    else:
        # print("async")
        shares = asyncio.run(
            sy.local_worker.async_dispatch(
                workers=locations,
                commands=[(full_name(evaluate), None, workers_args[i], {}) for i in [0, 1]],
            )
        )

    shares = {loc.id: share for loc, share in zip(locations, shares)}

    response = sy.AdditiveSharingTensor(shares, **class_attributes)

    return response


# share level
@allow_command
def mask_builder(x1, x2, op):
    if not isinstance(x1, int):
        worker = x1.owner
        numel = x1.numel()
    else:
        worker = x2.owner
        numel = x2.numel()
    x = x1 - x2

    keys = worker.crypto_store.get_keys(f"fss_{op}", n_instances=numel, remove=False)
    alpha = np.frombuffer(np.ascontiguousarray(keys[:, 0:N]), dtype=np.uint32)
    r = x + th.tensor(alpha.astype(np.int64)).reshape(x.shape)
    return r


# share level
@allow_command
def evaluate(b, x_masked, op, dtype):
    if op == "eq":
        return eq_evaluate(b, x_masked)
    elif op == "comp":
        return comp_evaluate(b, x_masked, dtype=dtype)
    else:
        raise ValueError


# process level
def eq_evaluate(b, x_masked):
    keys = x_masked.owner.crypto_store.get_keys(
        op="fss_eq", n_instances=x_masked.numel(), remove=True
    )
    result_share = DPF.eval(b.numpy().item(), x_masked.numpy(), keys)

    return th.tensor(result_share)


# process level
def comp_evaluate(b, x_masked, owner_id=None, core_id=None, burn_offset=0, dtype=None):
    if owner_id is not None:
        x_masked.owner = x_masked.owner.get_worker(owner_id)

    if burn_offset > 0:
        _ = x_masked.owner.crypto_store.get_keys(
            op="fss_comp", n_instances=burn_offset, remove=True
        )

    keys = x_masked.owner.crypto_store.get_keys(
        op="fss_comp", n_instances=x_masked.numel(), remove=True
    )

    result_share = DIF.eval(b.numpy().item(), x_masked.numpy(), keys)

    dtype_options = {None: th.long, "int": th.int32, "long": th.long}
    result = th.tensor(result_share, dtype=dtype_options[dtype])

    if core_id is None:
        return result
    else:
        return core_id, result


def eq(x1, x2):
    return fss_op(x1, x2, "eq")


def le(x1, x2):
    return fss_op(x1, x2, "comp")


class DPF:
    """Distributed Point Function - used for equality"""

    @staticmethod
    def keygen(n_values=1):
        return dpf.keygen(n_values=n_values)

    @staticmethod
    def eval(b, x, k_b):
        original_shape = x.shape
        x = x.reshape(-1)
        flat_result = dpf.eval(b, x, k_b)
        return flat_result.astype(np.int32).astype(np.int64).reshape(original_shape)


class DIF:
    """Distributed Interval Function - used for comparison"""

    @staticmethod
    def keygen(n_values=1):
        return dif.keygen(n_values=n_values)

    @staticmethod
    def eval(b, x, k_b):
        # x = x.astype(np.uint64)
        original_shape = x.shape
        x = x.reshape(-1)
        flat_result = dif.eval(b, x, k_b)
        return flat_result.astype(np.int32).astype(np.int64).reshape(original_shape)
