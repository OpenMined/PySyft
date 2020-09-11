import asyncio
import math
import multiprocessing
import torch as th

import syft as sy
from syft.exceptions import EmptyCryptoPrimitiveStoreError
from syft.generic.utils import allow_command
from syft.generic.utils import remote

from syft.frameworks.torch.mpc.fss import N_CORES

no_wrap = {"no_wrap": True}


def full_name(f):
    return f"syft.frameworks.torch.mpc.spdz.{f.__name__}"


# share level
@allow_command
def spdz_mask(x, y, op: str, dtype: str, torch_dtype: th.dtype, field: int):
    """
    Build the shares of delta and epsilon in the SPDZ protocol
    Args:
        x (Tensor): share of x, where the global computation is z = x ° y
        y (Tensor): share of y
        op (str): type of operation ('mul' or 'matmul')
        dtype (str): type of sahres ('int' or 'long')
        torch_dtype (th.dtype): corresponding torch dtype
        field (int): the field of the corresponding AdditiveSharingTensor

    Returns:
        The shares of delta and epsilon
    """
    a, b, c = x.owner.crypto_store.get_keys(
        op=op,
        shapes=(x.shape, y.shape),
        n_instances=1,
        remove=False,
        dtype=dtype,
        torch_dtype=torch_dtype,
        field=field,
    )
    return x - a, y - b


def slice(x, j, slice_size):
    x_slice = x[j * slice_size : (j + 1) * slice_size]
    x_slice.owner = x.owner
    return x_slice


def triple_mat_mul(core_id, delta, epsilon, a, b):
    cmd = th.matmul
    delta_b = cmd(delta, b)
    a_epsilon = cmd(a, epsilon)
    delta_epsilon = cmd(delta, epsilon)
    return core_id, delta_b, a_epsilon, delta_epsilon


# share level
@allow_command
def spdz_compute(j: int, delta, epsilon, op: str, dtype: str, torch_dtype: th.dtype, field: int):
    """
    Compute the mul or matmul part of the SPDZ protocol, once delta and epsilon
    have been made public
    Args:
        j (int): the rank of the worker, from 0 to n_worker - 1
        delta (Tensor): delta in the SPDZ protocol
        epsilon (Tensor): epsilon in the SPDZ protocol
        op (str): type of operation ('mul' or 'matmul')
        dtype (str): type of sahres ('int' or 'long')
        torch_dtype (th.dtype): corresponding torch dtype
        field (int): the field of the corresponding AdditiveSharingTensor

    Returns:
        The shares of the result of the multiplication
    """
    a, b, c = delta.owner.crypto_store.get_keys(
        op=op,
        shapes=(delta.shape, epsilon.shape),
        n_instances=1,
        remove=True,
        dtype=dtype,
        torch_dtype=torch_dtype,
        field=field,
    )

    if op == "matmul":

        batch_size = delta.shape[0]

        multiprocessing_args = []
        slice_size = math.ceil(batch_size / N_CORES)
        for core_id in range(N_CORES):
            process_args = (
                core_id,
                slice(delta, core_id, slice_size),
                epsilon,
                slice(a, core_id, slice_size),
                b,
            )
            multiprocessing_args.append(process_args)
        p = multiprocessing.Pool()
        partitions = p.starmap(triple_mat_mul, multiprocessing_args)
        p.close()
        partitions = sorted(partitions, key=lambda k: k[0])
        delta_b = th.cat([partition[1] for partition in partitions])
        a_epsilon = th.cat([partition[2] for partition in partitions])
        delta_epsilon = th.cat([partition[3] for partition in partitions])
    else:
        cmd = getattr(th, op)

        delta_b = cmd(delta, b)
        a_epsilon = cmd(a, epsilon)
        delta_epsilon = cmd(delta, epsilon)

    if j == 0:
        return delta_epsilon + delta_b + a_epsilon + c
    else:
        return delta_b + a_epsilon + c


def spdz_mul(cmd, x, y, crypto_provider, dtype, torch_dtype, field):
    """Abstractly multiplies two tensors (mul or matmul)
    Args:
        cmd: a callable of the equation to be computed (mul or matmul)
        x (AdditiveSharingTensor): the left part of the operation
        y (AdditiveSharingTensor): the right part of the operation
        crypto_provider (AbstractWorker): an AbstractWorker which is used
            to generate triples
        dtype (str): denotes the dtype of the shares, should be 'long' (default),
            'int' or 'custom'
        torch_dtype (torch.dtype): the real type of the shares, should be th.int64
            (default) or th.int32
        field (int): an integer denoting the size of the field, default is 2**64
    Return:
        an AdditiveSharingTensor
    """

    op = cmd
    locations = x.locations
    # Experimental results don't show real improvements with asynchronous = True
    asynchronous = False  # isinstance(locations[0], WebsocketClientWorker)

    try:
        shares_delta, shares_epsilon = [], []
        for location in locations:
            args = (x.child[location.id], y.child[location.id], op, dtype, torch_dtype, field)
            share_delta, share_epsilon = remote(spdz_mask, location=location)(
                *args, return_value=True, return_arity=2
            )
            shares_delta.append(share_delta)
            shares_epsilon.append(share_epsilon)
    except EmptyCryptoPrimitiveStoreError as e:
        if sy.local_worker.crypto_store.force_preprocessing:
            raise
        sy.local_worker.crypto_store.provide_primitives(workers=locations, **e.kwargs_)
        return spdz_mul(cmd, x, y, crypto_provider, dtype, torch_dtype, field)

    delta = sum(shares_delta)
    epsilon = sum(shares_epsilon)

    for location, share_delta, share_epsilon in zip(locations, shares_delta, shares_epsilon):
        location.de_register_obj(share_delta)
        location.de_register_obj(share_epsilon)
        del share_delta
        del share_epsilon

    if not asynchronous:
        shares = []
        for i, location in enumerate(locations):
            args = (th.LongTensor([i]), delta, epsilon, op, dtype, torch_dtype, field)
            share = remote(spdz_compute, location=location)(*args, return_value=False)
            shares.append(share)
    else:
        shares = asyncio.run(
            sy.local_worker.async_dispatch(
                workers=locations,
                commands=[
                    (
                        full_name(spdz_compute),
                        None,
                        (th.LongTensor([i]), delta, epsilon, op),
                        {},
                    )
                    for i in [0, 1]
                ],
                return_value=False,
            )
        )

    shares = {loc.id: share for loc, share in zip(locations, shares)}

    response = sy.AdditiveSharingTensor(shares, **x.get_class_attributes())
    return response
