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
    """
    This function performs the QR decomposition of a matrix (2-dim tensor). The
    decomposition is performed using Householder Reflection.
    Please note that this function only supports local or pointer tensors, it
    does not support instances of FixedPrecisionTensor or AdditiveSharedTensor

    Args:
        t: symmetric 2-dim tensor. It should be whether a local tensor or a pointer
            to a remote tensor

    Returns:
        q: orthogonal matrix as a 2-dim tensor with same type as t
        r: lower triangular matrix as a 2-dim tensor with same type as t
    """
    n_cols = t.shape[1]

    # Initiate R matrix from t
    R = t.copy()

    # Initiate identity matrix with same size and in the same worker as t
    I = th.diag(th.diag(th.ones_like(t)))

    # Iteration via Household Reflection
    for i in range(n_cols):
        # Identity for this iteration, it has size (n_cols-i, n_cols-i)
        I_i = I[i:, i:]

        # Init 1st vector of the canonical base in the same worker as t
        e = th.zeros_like(t)[i:, 0].view(-1, 1)
        e[0, 0] += 1.0

        # Current vector in R to perform reflection
        x = R[i:, i].view(-1, 1)
        x_norm = th.sqrt(x.t() @ x).squeeze()

        # Compute Householder transform
        numerator = x @ x.t() - x_norm * (e @ x.t() + x @ e.t()) + (x.t() @ x) * (e @ e.t())
        denominator = x.t() @ x - x_norm * x[0, 0]
        H = I_i - numerator / denominator

        # If it's the 1st iteration, init Q_transpose
        if i == 0:
            Q_t = H
        else:
            # Expand matrix H with Identity at diagonal and zero elsewhere
            down_zeros = th.zeros_like(t)[: n_cols - i, :i]
            up_zeros = th.zeros_like(t)[:i, : n_cols - i]
            left_cat = th.cat((I[:i, :i], down_zeros), dim=0)
            right_cat = th.cat((up_zeros, H), dim=0)
            H = th.cat((left_cat, right_cat), dim=1)
            # Update Q_transpose
            Q_t = H @ Q_t
        # Update R
        R = H @ R

    # Get Q from Q_transpose
    Q = Q_t.t()

    return Q, R
