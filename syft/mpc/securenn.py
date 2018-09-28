# An implementation of the SecureNN protocols from Wagh et al.

from syft.spdz.spdz import (spdz_add, spdz_mul,
                           generate_zero_shares_communication,
                           Q_BITS, field)
from syft.core.frameworks.torch.tensor import _GeneralizedPointerTensor, _SPDZTensor
import torch

L = field
p = field  # 67 in original # TODO: extend to ops over multiple rings


def decompose(tensor):
    """
    decompose a tensor into its binary representation
    """
    powers = torch.arange(Q_BITS)
    for i in range(len(tensor.shape)):
        powers = powers.unsqueeze(0)
    tensor = tensor.unsqueeze(-1)
    moduli = 2 ** powers
    tensor = ((tensor + 2 ** (Q_BITS)) / moduli.type_as(tensor)) % 2
    return tensor


def select_shares(alpha, x, y, workers):
    """
    alpha is a shared binary tensor
    x and y are private tensors to choose elements or slices from
        oi8u7y6t(following broadcasting rules)

    all of type _GeneralizedPointerTensor

    Computes z = (1 - alpha) * x + alpha * y
    """
    # FIXME: generate_zero_shares_communication should be updated with new pysyft API
    u = generate_zero_shares_communication(*workers, *x.get_shape())
    z = x + alpha * (y - x)
    return z + u


def private_compare(x, r, beta, workers):
    """
    computes beta XOR (x > r)

    x is private shared bit tensor (shared output of decompose)
    r is public input for comparison
    beta is public random bit tensor

    all of type _GeneralizedPointerTensor
    """
    dimlen = len(x.size())
    while len(beta.size()) < dimlen:
        beta = beta.unsqueeze(-1)
    beta = beta.expand_as(x)

    t = (r + 1) % (2 ** Q_BITS)

    r_bits = decompose(r)
    t_bits = decompose(t)

    zeros = beta == 0
    ones = beta == 1
    others = (r == (2 ** Q_BITS - 1)).unsqueeze(-1).expand_as(ones)
    ones = ones & (others - 1).long().abs().byte()

    c_zeros = _pc_beta0(x[zeros], r_bits[zeros])
    c_ones = _pc_beta1(x[ones], t_bits[ones])
    c_other = _pc_else(workers, *x.size())

    c = torch.zeros(*x_bits.shape).long()
    c[zeros] = c_zeros
    c[ones] = c_ones
    c[others] = c_other

    s = random_as(c, mod=p)
    permute = torch.randperm(c.size(-1))
    d = s * c[..., permute]
    d.get()
    return (d == 0).max()


def msb(x, workers):
    """
    computes the most significant bit of a shared input tensor
    uses the fact that msb(x) = lsb(2x) in an odd ring,
    so coerces the shares to an odd ring if needed (FIXME)
    """
    if L % 2 != 1:
        x = share_convert(x)
    return lsb(2 * x, workers)


def lsb(y, workers):
    """
    computes the least significant bit of a shared input tensor
    """
    # happens on crypto producer
    xcrypt = random_as(y)
    xbits = decompose(xcrypt)
    xlsb = xbits[..., 0]
    beta = random_as(y, mod=2)
    # share x, xlsb, and xbits here, send to the 2 parties


    r = y + x
    r.get()
    rbits = decompose(r)
    rlsb = rbits[..., 0]
    r = r.share(*workers)  # TODO: remove this line when public-private addition works
    beta_prime = private_compare(xbits, r, beta)

    gamma = xor(beta, beta_prime)
    delta = xor(xlsb, rlsb)
    alpha = xor(gamma, delta)
    u = generate_zero_shares_communication(*workers, *alpha.get_shape())
    return alpha + u


def relu(x, workers):
    return x * nonnegative(x, workers)


def nonnegative(a, workers):
    c = 2 * a
    alpha = msb(c)
    gamma = nope(alpha)
    u = generate_zero_shares_communication(*workers, *gamma.get_shape())
    return gamma + u


def random_as(tensor, mod=L):
    r = torch.LongTensor(tensor.shape).random_(mod)
    return r.type_as(tensor)


def xor(x, y):
    return x + y - 2 * x * y


def nope(x):
    return 1 - x


def get_wrap(x, y, mod=L):
    # FIXME: the conditional on the right should include a plaintext add of x and y,
    # not an MPC add since the > test needs to be done without a modulus
    # needed for share_convert
    return x + y, x + y > mod


def share_convert(x):
    # FIXME: implement share convert protocol
    raise NotImplementedError("Share convert protocol unfinished -- Ring modulus needs to be odd.")


def _pc_beta0(x, r):
    # note x and r are both binary tensors,
    # and dim -1 contains their bits
    # x should be shared, r should be public
    w = xor(x, r)
    z = r  - (x - 1)
    w_sum = torch.zeros(*w.get_shape()).type_as(w)
    for i in range(Q_BITS - 2, -1, -1):
        w_sum[..., i] = w[..., (i + 1):].sum(dim=-1, keepdim=False)[0]
    c = z + w_sum
    return c


def _pc_beta1(x, t):
    w = xor(x, t)
    z = (x + 1) - t
    w_sum = torch.zeros(*w.get_shape()).type_as(w)
    for i in range(Q_BITS - 2, -1, -1):
        w_sum[..., i] = w[..., (i + 1):].sum(dim=-1, keepdim=False)[0]
    c = z + w_sum
    return c


def _pc_else(workers, *sizes):
    u = generate_zero_shares_communication(*workers, *sizes)
    print('u', type(u), u)
    (w0, u0), (w1, u1) = u.shares.pointer_tensor_dict.items()
    for i in range(Q_BITS - 2, -1, -1):
        if i == 0:
            c0[..., i] = -u0
            c1[..., i] = u1
        c0[..., i] = u0 + 1
        c1[..., i] = -u1
    ptr_dict = {w0:c0, w1:c1}
    c_gp = _GeneralizedPointerTensor(ptr_dict, torch_type='syft.LongTensor').wrap(True)
    c = _SPDZTensor(x_gp, torch_type='syft.LongTensor').wrap(True)
    return c
