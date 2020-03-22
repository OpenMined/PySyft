from typing import Callable
import torch

from syft.workers.abstract import AbstractWorker


def request_triple(
    crypto_provider: AbstractWorker,
    cmd: Callable,
    field: int,
    dtype: str,
    a_size: tuple,
    b_size: tuple,
    locations: list,
):
    """Generates a multiplication triple and sends it to all locations.

    Args:
        crypto_provider: worker you would like to request the triple from
        cmd: An equation in einsum notation.
        field: An integer representing the field size.
        dtype: represents the dtype of shares
        a_size: A tuple which is the size that a should be or
                a torch.Size instance
        b_size: A tuple which is the size that b should be or
                a torch.Size intance
        locations: A list of workers where the triple should be shared between.

    Returns:
        A triple of AdditiveSharedTensors such that c_shared = cmd(a_shared, b_shared).
    """
    a = crypto_provider.remote.torch.randint(-(field // 2), (field - 1) // 2, a_size)
    b = crypto_provider.remote.torch.randint(-(field // 2), (field - 1) // 2, b_size)
    c = cmd(a, b)

    res = torch.cat((a.view(-1), b.view(-1), c.view(-1)))

    shares = (
        res.share(*locations, field=field, dtype=dtype, crypto_provider=crypto_provider).get().child
    )
    a_shared = shares[: a.numel()].reshape(a_size)
    b_shared = shares[a.numel() : -c.numel()].reshape(b_size)
    c_shared = shares[-c.numel() :].reshape(c.shape)

    return a_shared, b_shared, c_shared
