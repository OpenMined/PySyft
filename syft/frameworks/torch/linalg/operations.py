import torch as th
import syft as sy


def inv_sym(t):
    """
    This function performs the inversion of a symmetric matrix (2-dim tensor) in MPC.
    It uses LDLt decomposition, which is better than Cholensky decomposition in our case
    since it doesn't use square root.
    Algorithm reference: https://arxiv.org/abs/1111.4144 - Section IV

    Args:
        t: symmetric 2-dim tensor

    Returns:
        t_inv: inverse of t as 2-dim tensor

    """

    n = t.shape[0]
    l, d, inv_d = _ldl(t)
    l_t = l.t()
    t_inv = th.zeros_like(t)
    for j in range(n - 1, -1, -1):
        for i in range(j, -1, -1):
            if i == j:
                t_inv[i, j] = inv_d[i] - (l_t[i, i + 1 : n] * t_inv[i + 1 : n, j]).sum()
            else:
                t_inv[j, i] = t_inv[i, j] = -(l_t[i, i + 1 : n] * t_inv[i + 1 : n, j]).sum()

    return t_inv


def _ldl(t):
    """
    This function performs the LDLt decomposition of a symmetric matrix (2-dim tensor)

    Args:
        t: symmetric 2-dim tensor

    Returns:
        l: lower triangular matrix as a 2-dim tensor with same type as t
        d: 1-dim tensor which represents the diagonal in the LDLt decomposition
        inv_d: 1-dim tensor which represents the inverse of the diagonal d. It is usefull
               when computing inverse of a symmetric matrix, by caching it we avoid repeated
               computations with division, which is very slow in MPC
    """
    n = t.shape[0]
    l = th.zeros_like(t)
    d = th.diag(l).copy()
    inv_d = d.copy()

    for i in range(n):
        d[i] = t[i, i] - (l[i, :i] ** 2 * d[:i]).sum()
        inv_d[i] = (0 * d[i] + 1) / d[i]  # Needed to compute inv of a number in MPC
        for j in range(i, n):
            # The diagonal of L in LDLt decomposition is 1
            if j == i:
                l[j, i] += 1
            else:
                l[j, i] = (t[j, i] - (l[j, :i] * l[i, :i] * d[:i]).sum()) * inv_d[i]

    return l, d, inv_d


def qr(t):
    # TODO
    pass
