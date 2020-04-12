from typing import Callable

import torch

import syft as sy
from syft.frameworks.torch.mpc.beaver import request_triple
from syft.workers.abstract import AbstractWorker

no_wrap = {"no_wrap": True}


def spdz_mul(cmd: Callable, x_sh, y_sh, crypto_provider: AbstractWorker, field: int):
    """Abstractly multiplies two tensors (mul or matmul)

    Args:
        cmd: a callable of the equation to be computed (mul or matmul)
        x_sh (AdditiveSharingTensor): the left part of the operation
        y_sh (AdditiveSharingTensor): the right part of the operation
        crypto_provider (AbstractWorker): an AbstractWorker which is used to generate triples
        field (int): an integer denoting the size of the field

    Return:
        an AdditiveSharingTensor
    """
    assert isinstance(x_sh, sy.AdditiveSharingTensor)
    assert isinstance(y_sh, sy.AdditiveSharingTensor)

    locations = x_sh.locations

    # Get triples
    a, b, a_mul_b = request_triple(crypto_provider, cmd, field, x_sh.shape, y_sh.shape, locations)

    delta = x_sh - a  # Not gced coz the pointer to other loc doesnt exist
    epsilon = y_sh - b
    # Reconstruct and send to all workers
    delta_recon = delta.reconstruct()
    epsilon_recon = epsilon.reconstruct()

    delta_epsilon = cmd(delta_recon, epsilon_recon)

    # Trick to keep only one child in the MultiPointerTensor (like in SNN)
    j1 = torch.ones(delta_epsilon.shape).long().send(locations[0], **no_wrap)
    j0 = torch.zeros(delta_epsilon.shape).long().send(*locations[1:], **no_wrap)
    if len(locations) == 2:
        j = sy.MultiPointerTensor(children=[j1, j0])
    else:
        j = sy.MultiPointerTensor(children=[j1] + list(j0.child.values()))

    delta_b = cmd(delta_recon, b)
    a_epsilon = cmd(a, epsilon_recon)

    # delta.get()
    # epsilon.get()
    # for loc in locations:
    #     loc.de_register_obj(delta)
    #     loc.de_register_obj(epsilon)
    # try:
    #     # delta.locations[0].de_register_obj(delta)
    #     # delta.locations[0].rm_obj(delta.child[delta.locations[0].id])
    #     del delta.child[delta.locations[0].id]
    #     # epsilon.locations[0].de_register_obj(epsilon)
    #     # epsilon.locations[0].rm_obj(epsilon.child[epsilon.locations[0].id])
    #     del epsilon.child[epsilon.locations[0].id]
    # except Exception as e:
    #     print(e)

    return delta_epsilon * j + delta_b + a_epsilon + a_mul_b
