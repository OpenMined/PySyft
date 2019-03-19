import torch
from typing import Callable
from syft.workers.abstract import AbstractWorker


def spdz_mul(
    cmd: Callable, shares: dict, other_shares, crypto_provider: AbstractWorker, field: int, **kwargs
):
    """Abstractly Multiplies two tensors

    Args:
        cmd: a callable of the equation to be commuted
        shares: a dictionary <location_id -> PointerTensor) of shares corresponding to
            self. Equivalent to calling self.child.
        other_shares: a dictionary <location_id -> PointerTensor) of shares corresponding
            to the tensor being multiplied by self.
        cypto_provider: an AbstractWorker which is used to generate triples
        field: an interger denoting the size of the field
        """
    locations = list(shares.keys())
    shares_shape = shares[locations[0]].shape
    other_shape = other_shares[locations[0]].shape
    triple = crypto_provider.generate_triple(cmd, field, shares_shape, other_shape, locations)
    a, b, c = triple
    d = {}
    e = {}
    for location in locations:
        d[location] = shares[location] - a[location]
        e[location] = other_shares[location] - b[location]
    delta = torch.zeros(shares_shape, dtype=torch.long)
    epsilon = torch.zeros(other_shape, dtype=torch.long)

    for location in locations:
        d_temp = d[location].get()
        e_temp = e[location].get()
        delta = delta + d_temp
        epsilon = epsilon + e_temp

    delta_epsilon = cmd(delta, epsilon)

    delta_ptrs = {}
    epsilon_ptrs = {}
    a_epsilon = {}
    delta_b = {}
    z = {}
    for location in locations:
        delta_ptrs[location] = delta.send(location)
        epsilon_ptrs[location] = epsilon.send(location)
        a_epsilon[location] = cmd(a[location], epsilon_ptrs[location])
        delta_b[location] = cmd(delta_ptrs[location], b[location])
        z[location] = a_epsilon[location] + delta_b[location] + c[location]
    delta_epsilon_pointer = delta_epsilon.send(locations[0])
    z[locations[0]] = z[locations[0]] + delta_epsilon_pointer
    return z
