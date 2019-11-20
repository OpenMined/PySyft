import torch
from typing import Callable
from syft.workers.abstract import AbstractWorker


def request_triple(
    crypto_provider: AbstractWorker,
    cmd: Callable,
    field: int,
    a_size: tuple,
    b_size: tuple,
    locations: list,
):
    """Generates a multiplication triple and sends it to all locations.

    Args:
        crypto_provider: worker you would like to request the triple from
        cmd: An equation in einsum notation.
        field: An integer representing the field size.
        a_size: A tuple which is the size that a should be.
        b_size: A tuple which is the size that b should be.
        locations: A list of workers where the triple should be shared between.

    Returns:
        A triple of AdditiveSharedTensors such that c_shared = cmd(a_shared, b_shared).
    """
    a = torch.randint(field, a_size)
    b = torch.randint(field, b_size)
    c = cmd(a, b)
    a_shared = a.share(*locations, field=field, crypto_provider=crypto_provider).child
    b_shared = b.share(*locations, field=field, crypto_provider=crypto_provider).child
    c_shared = c.share(*locations, field=field, crypto_provider=crypto_provider).child
    return a_shared, b_shared, c_shared
