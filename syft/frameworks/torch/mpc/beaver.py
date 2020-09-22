import torch as th
from typing import Tuple

import syft as sy
from .cuda import CUDALongTensor

import torchcsprng as csprng
generator = csprng.create_random_device_generator('/dev/urandom')

def build_triple(
    op: str,
    kwargs_, 
    shape: Tuple[th.Size, th.Size],
    n_workers: int,
    n_instances: int,
    torch_dtype: th.dtype,
    field: int,
):
    """
    Generates and shares a multiplication triple (a, b, c)

    Args:
        op (str): 'mul' or 'matmul': the op ° which ensures a ° b = c
        shape (Tuple[th.Size, th.Size]): the shapes of a and b
        n_workers (int): number of workers
        n_instances (int): the number of tuples (works only for mul: there is a
            shape issue for matmul which could be addressed)
        torch_dtype (th.dtype): the type of the shares
        field (int): the field for the randomness

    Returns:
        a triple of shares (a_sh, b_sh, c_sh) per worker where a_sh is a share of a
    """
    left_shape, right_shape = shape
    cmd = getattr(th, op)
    low_bound, high_bound = -(field // 2), (field - 1) // 2
    a = th.empty(
        *left_shape, dtype=torch_dtype, device='cuda'
    ).random_(low_bound, high_bound, generator=generator)
    b = th.empty(
        *right_shape, dtype=torch_dtype, device='cuda'
    ).random_(low_bound, high_bound, generator=generator)

    if op == "mul":
        if b.numel() == a.numel():
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
    elif op in {"matmul", "conv2d"}:
        cmd = getattr(CUDALongTensor, op)
        a_ = CUDALongTensor(a)
        b_ = CUDALongTensor(b)
        c = cmd(a_, b_, **kwargs_)
        c = c._tensor
    else:
        raise ValueError

    helper = sy.AdditiveSharingTensor(field=field)

    shares_worker = [[0, 0, 0] for _ in range(n_workers)]
    for i, tensor in enumerate([a, b, c]):
        shares = helper.generate_shares(secret=tensor, n_workers=n_workers, random_type=torch_dtype)
        for w_id in range(n_workers):
            shares_worker[w_id][i] = shares[w_id].unsqueeze(0) #TODO change primitives to not expect batches
    return shares_worker
