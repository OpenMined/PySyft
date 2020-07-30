import torch as th
from typing import Tuple


def build_triple(
    op: str,
    shape: Tuple[th.Size],
    n_worker: int,
    n_instances: int,
    torch_dtype: th.dtype,
    field: int,
):
    """
    Generates and shares a multiplication triple (a, b, c)

    Args:
        op (str): 'mul' or 'matmul': the op ° which ensures a ° b = c
        shape (Tuple[th.Size]): the shapes of a and b
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
    a = th.randint(low_bound, high_bound, (n_instances, *left_shape), dtype=torch_dtype)
    b = th.randint(low_bound, high_bound, (n_instances, *right_shape), dtype=torch_dtype)

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

    shares_worker = [[0, 0, 0] for _ in range(n_worker)]
    for i, tensor in enumerate([a, b, c]):
        shares_worker[0][i] = tensor
        for w_id in range(n_worker - 1):
            mask = th.randint(low_bound, high_bound, tensor.shape, dtype=torch_dtype)
            shares_worker[w_id][i] -= mask
            shares_worker[w_id + 1][i] = mask

    return shares_worker
