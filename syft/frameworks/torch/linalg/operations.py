import torch

from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor


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
    t_inv = torch.zeros_like(t)
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
    l = torch.zeros_like(t)
    d = torch.diag(l).copy()
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


def qr(t, mode="reduced", norm_factor=None):  # noqa: C901
    """
    This function performs the QR decomposition of a matrix (2-dim tensor). The
    decomposition is performed using Householder Reflection.

    Args:
        t: 2-dim tensor, shape(M, N). It should be whether a local tensor, a
            pointer to a remote tensor or an AdditiveSharedTensor

        mode: {'reduced', 'complete', 'r'}. If K = min(M, N), then
            - 'reduced' : returns q, r with dimensions (M, K), (K, N) (default)
            - 'complete' : returns q, r with dimensions (M, M), (M, N)
            - 'r' : returns r only with dimensions (K, N)

        norm_factor: float. The normalization factor used to avoid overflow when
            performing QR decomposition on an AdditiveSharedTensor. For example in
            the case of the DASH algorithm, this norm_factor should be of the
            order of the square root of number of entries in the original matrix
            used to perform the compression phase assuming the entries are standardized.

    Returns:
        q: orthogonal matrix as a 2-dim tensor with same type as t
        r: lower triangular matrix as a 2-dim tensor with same type as t
    """

    # Check if t is a PointerTensor or an AST
    if t.has_child() and isinstance(t.child, PointerTensor):
        t_type = "pointer"
        location = t.child.location
    elif t.has_child() and isinstance(t.child.child, AdditiveSharingTensor):
        t_type = "ast"
        if norm_factor is None:
            raise ValueError(
                "You are trying to perform QR decompostion on an ",
                "AdditiveSharingTensor, please provide a value for the ",
                "norm_factor argument.",
            )
        workers = t.child.child.locations
        crypto_prov = t.child.child.crypto_provider
        prec_frac = t.child.precision_fractional
        protocol = t.child.child.protocol
    elif isinstance(t, torch.Tensor):
        t_type = "local"
    else:
        raise TypeError(
            "The provided matrix should be a local torch.Tensor, a PointerTensor, ",
            "or an AdditiveSharedTensor",
        )

    # Check if t is 2-dim
    assert len(t.shape) == 2

    if mode not in ["reduced", "complete", "r"]:
        raise ValueError(
            "mode should have one of the values in the list:" + str(["reduced", "complete", "r"])
        )

    ######## QR decomposition via Householder Reflection #########
    n_rows, n_cols = t.shape

    # Initiate R matrix from t
    R = t.copy()

    # Initiate identity matrix with size (n_rows, n_rows)
    I = torch.diag(torch.Tensor([1.0] * n_rows))

    # Send it to remote worker if t is pointer, secret share it if it's an AST
    if t_type == "pointer":
        I = I.send(location)
    if t_type == "ast":
        I = I.fix_prec(precision_fractional=prec_frac).share(
            *workers, crypto_provider=crypto_prov, protocol=protocol
        )

    if not mode == "r":
        # Initiate Q_transpose
        Q_t = I.copy()

    # Iteration via Household Reflection
    for i in range(min(n_rows, n_cols)):

        # Identity for this iteration, it has size (n_cols-i, n_cols-i)
        I_i = I[i:, i:]

        # Init 1st vector of the canonical base in the same worker as t
        e = torch.zeros_like(t)[i:, 0].view(-1, 1)
        e[0, 0] += 1

        # Current vector in R to perform reflection
        x = R[i:, i].view(-1, 1)

        # Compute norm in MPC if it's an AST
        x_norm = _norm_mpc(x, norm_factor) if t_type == "ast" else torch.sqrt(x.t() @ x).squeeze()

        # Compute Householder transform
        numerator = x @ x.t() - x_norm * (e @ x.t() + x @ e.t()) + (x.t() @ x) * (e @ e.t())
        denominator = x.t() @ x - x_norm * x[0, 0]
        # Need the line below to perform inverse of a number in MPC
        inv_denominator = (0 * denominator + 1) / denominator
        H = I_i - numerator * inv_denominator

        # If it is not the 1st iteration
        # expand matrix H with Identity at diagonal and zero elsewhere
        if i > 0:
            down_zeros = torch.zeros([n_rows - i, i])
            up_zeros = torch.zeros([i, n_rows - i])

            # Send them to remote worker if t is pointer, secret share it if its an AST
            if t_type == "pointer":
                down_zeros = down_zeros.send(location)
                up_zeros = up_zeros.send(location)
            if t_type == "ast":
                down_zeros = down_zeros.fix_prec(precision_fractional=prec_frac).share(
                    *workers, crypto_provider=crypto_prov, protocol=protocol
                )
                up_zeros = up_zeros.fix_prec(precision_fractional=prec_frac).share(
                    *workers, crypto_provider=crypto_prov, protocol=protocol
                )
            left_cat = torch.cat((I[:i, :i], down_zeros), dim=0)
            right_cat = torch.cat((up_zeros, H), dim=0)
            H = torch.cat((left_cat, right_cat), dim=1)

        # Update R
        R = H @ R
        if not mode == "r":
            # Update Q_transpose
            Q_t = H @ Q_t

    if mode == "reduced":
        R = R[:n_cols, :]
        Q_t = Q_t[:n_cols, :]

    if mode == "r":
        R = R[:n_cols, :]
        return R

    return Q_t.t(), R


def _norm_mpc(t, norm_factor):
    """
    Computation of a norm of a vector in MPC. The vector should be an AdditiveSharedTensor.

    It performs the norm calculation by masking the tensor with a multiplication
    by a big random number drawn from a uniform distribution, and computing the
    square root of the squared norm of the masked tensor, which is computed
    beforehand with a dot product in MPC.

    In order to maintain stability and avoid overflow, this functions uses a
    norm_factor that scales down the tensor for MPC computations and rescale it at the end.
    For example in the case of the DASH algorithm, this norm_factor should be of
    the order of the square root of number of entries in the original matrix
    used to perform the compression phase assuming the entries are standardized.

    Args:
        t: 1-dim AdditiveSharedTensor, representing a vector.
        norm_factor: float. The normalization factor used to avoid overflow

    Returns:
        the norm of the vector as an AdditiveSharedTensor

    """

    workers = t.child.child.locations
    crypto_prov = t.child.child.crypto_provider
    prec_frac = t.child.precision_fractional
    protocol = t.child.child.protocol
    field = t.child.child.field
    norm_factor = int(norm_factor)
    t_normalized = t / norm_factor
    Q = int(field ** (1 / 2) / 10 ** (prec_frac / 2))

    norm_sq = (t_normalized ** 2).sum().squeeze()

    # Random big number
    r = (
        torch.LongTensor([0])
        .fix_precision(precision_fractional=prec_frac)
        .share(*workers, crypto_provider=crypto_prov, protocol=protocol)
        .random_(0, Q)
    )

    # Compute masked norm
    masked_norm_sq = r ** 2 * norm_sq

    # Get compute square root
    masked_norm_sq = masked_norm_sq.send(crypto_prov).remote_get().float_precision()
    masked_norm = torch.sqrt(masked_norm_sq)

    # Secret share and compute unmasked norm in MPC
    masked_norm = (
        masked_norm.fix_precision(precision_fractional=prec_frac)
        .share(*workers, crypto_provider=crypto_prov, protocol=protocol)
        .get()
    )
    norm = masked_norm / r * norm_factor

    return norm.squeeze()
