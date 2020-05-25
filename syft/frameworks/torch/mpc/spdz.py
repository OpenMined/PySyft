from typing import Callable

import torch

import syft as sy
from syft.frameworks.torch.mpc.beaver import request_triple
from syft.workers.abstract import AbstractWorker

no_wrap = {"no_wrap": True}


def spdz_mul(cmd: Callable, x_sh, y_sh, crypto_provider: AbstractWorker, field: int, dtype: str):
    """Abstractly multiplies two tensors (mul or matmul)

    Args:
        cmd: a callable of the equation to be computed (mul or matmul)
        x_sh (AdditiveSharingTensor): the left part of the operation
        y_sh (AdditiveSharingTensor): the right part of the operation
        crypto_provider (AbstractWorker): an AbstractWorker which is used to generate triples
        field (int): an integer denoting the size of the field
        dtype (str): denotes the dtype of shares

    Return:
        an AdditiveSharingTensor
    """
    assert isinstance(x_sh, sy.AdditiveSharingTensor)
    assert isinstance(y_sh, sy.AdditiveSharingTensor)

    locations = x_sh.locations
    torch_dtype = x_sh.torch_dtype

    # Get triples
    a, b, a_mul_b = request_triple(
        crypto_provider, cmd, field, dtype, x_sh.shape, y_sh.shape, locations
    )

    delta = x_sh - a
    epsilon = y_sh - b
    # Reconstruct and send to all workers
    delta = delta.reconstruct()
    epsilon = epsilon.reconstruct()

    delta_epsilon = cmd(delta, epsilon)

    # Trick to keep only one child in the MultiPointerTensor (like in SNN)
    j1 = torch.ones(delta_epsilon.shape).type(torch_dtype).send(locations[0], **no_wrap)
    j0 = torch.zeros(delta_epsilon.shape).type(torch_dtype).send(*locations[1:], **no_wrap)
    if len(locations) == 2:
        j = sy.MultiPointerTensor(children=[j1, j0])
    else:
        j = sy.MultiPointerTensor(children=[j1] + list(j0.child.values()))

    delta_b = cmd(delta, b)
    a_epsilon = cmd(a, epsilon)
    res = delta_epsilon * j + delta_b + a_epsilon + a_mul_b
    res = res.type(torch_dtype)
    return res
