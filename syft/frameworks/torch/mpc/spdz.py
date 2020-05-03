from typing import Callable

import torch as th

import syft as sy
from syft.frameworks.torch.mpc.beaver import request_triple
from syft.workers.abstract import AbstractWorker
from syft.frameworks.torch.mpc.fss import remote_exec, full_name

n = 62  # TODO; sy.frameworks.torch.mpc.securenn.Q_BITS

no_wrap = {"no_wrap": True}

NAMESPACE = "syft.frameworks.torch.mpc.spdz"
authorized2 = set(f"{NAMESPACE}.{name}" for name in ["spdz_mask", "spdz_compute"])


def full_name(f):
    return f"{NAMESPACE}.{f.__name__}"


# share level
def spdz_mask(x, y, type_op):
    a, b, c = x.owner.crypto_store.get_keys(
        "beaver", op=type_op, shapes=(x.shape, y.shape), n_instances=1, remove=False
    )
    return x - a, y - b


# share level
def spdz_compute(j, delta, epsilon, type_op):
    a, b, c = delta.owner.crypto_store.get_keys(
        "beaver", op=type_op, shapes=(delta.shape, epsilon.shape), n_instances=1, remove=True
    )

    cmd = getattr(th, type_op)

    delta_b = cmd(delta.float(), b.float()).long()
    a_epsilon = cmd(a.float(), epsilon.float()).long()
    delta_epsilon = cmd(delta.float(), epsilon.float()).long()

    if j:
        return delta_epsilon + delta_b + a_epsilon + c
    else:
        return delta_b + a_epsilon + c


def spdz_mul(cmd, x, y, crypto_provider, field, dtype):
    """
    Define the workflow for a binary operation using Function Secret Sharing

    Currently supported operand are = & <=, respectively corresponding to
    type_op = 'eq' and 'comp'

    Args:
        x1: first AST
        x2: second AST
        type_op: type of operation to perform, should be 'eq' or 'comp'

    Returns:
        shares of the comparison
    """

    # TODO field
    type_op = cmd
    locations = x.locations

    shares_delta, shares_epsilon = [], []
    for location in locations:
        args = (x.child[location.id], y.child[location.id], type_op)
        share_delta, share_epsilon = remote_exec(
            full_name(spdz_mask), location, args=args, return_value=True
        )
        shares_delta.append(share_delta)
        shares_epsilon.append(share_epsilon)

    delta = sum(shares_delta) % 2 ** n
    epsilon = sum(shares_epsilon) % 2 ** n

    shares = []
    for i, location in enumerate(locations):
        args = (th.LongTensor([i]), delta, epsilon, type_op)
        share = remote_exec(full_name(spdz_compute), location, args=args, return_value=False)
        shares.append(share)

    shares = {loc.id: share for loc, share in zip(locations, shares)}

    response = sy.AdditiveSharingTensor(shares, **x.get_class_attributes())
    return response
