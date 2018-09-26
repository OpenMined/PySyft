# An implementation of the SecureNN protocols from Wagh et al.

from syft.mpc.spdz import (spdz_add, spdz_mul,
                           generate_zero_shares_communication,
                           Q_BITS, field)
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
    tensor = (tensor / moduli.type_as(tensor)) % 2
    return tensor


def select_shares(alpha, x, y, workers, mod=L):
    """
    alpha is a shared binary tensor
    x and y are private tensors to choose elements or slices from
        oi8u7y6t(following broadcasting rules)

    all of type _GeneralizedPointerTensor

    Computes z = (1 - alpha) * x + alpha * y
    """
    u = generate_zero_shares_communication()
    z = x + alpha * (y - x)
    return z + u


def private_compare(x, r, beta):
    """
    computes beta XOR (x > r)

    x is private input
    r is public input for comparison
    beta is public random bit tensor

    all of type _GeneralizedPointerTensor
    """
    t = (r + 1) % (2 ** Q_BITS)

    x_bits = decompose(x)
    r_bits = decompose(r)
    t_bits = decompose(t)

    zeros = beta == 0
    ones = beta == 1
    others = r == (2 ** Q_BITS - 1)
    ones = ones & (others - 1).abs()

    c_zeros = _pc_beta0(x_bits, r_bits)
    c_ones = _pc_beta1(x_bits, t_bits)
    c_other = _pc_else()

    # TODO: recombine c properly here
    torch.zeros()
    c = torch.cat([c_zeros, c_ones, c_other], -1)

    s = random_as(c, mod=p)
    permute = torch.randperm(c.size(-1))
    d = s * c[..., permute]
    d.get()
    return (d == 0).max()



def msb(x):
    """
    computes the most significant bit of a shared input tensor
    uses the fact that msb(x) = lsb(2x) in an odd ring,
    so coerces the shares to an odd ring if needed (FIXME)
    """
    if L % 2 != 1:
        x = share_convert(x)
    return lsb(2 * x)


def lsb(y):
    """
    computes the least significant bit of a shared input tensor
    """
    # happens on crypto producer
    x = random_as(y)
    xbits = decompose(x)
    xlsb = xbits[..., 0]
    beta = random_as(y, mod=2)
    # share x, xlsb, and xbits here, send to the 2 parties
    r = y + x
    # r.get()  # reconstruct r
    rbits = decompose(r)
    rlsb = rbits[..., 0]
    beta_prime = private_compare(xbits, r, beta)

    gamma = xor(beta, beta_prime)
    delta = xor(xlsb, rlsb)
    alpha = xor(gamma, delta)
    u = generate_zero_shares_communication()
    return alpha + u


def share_convert(x):
    # FIXME: implement share convert protocol
    raise NotImplementedError("Share convert protocol unfinished -- Ring modulus needs to be odd.")


def random_as(tensor, mod=L):
    r = torch.LongTensor(tensor.shape).random_(mod)
    return r.type_as(tensor)


def get_wrap(x, y, mod=L):
    # FIXME: the conditional on the right should include a plaintext add of x and y,
    # not an MPC add since the > test needs to be done without a modulus
    # needed to complete share_convert
    return x + y, x + y > mod


def xor(x, y):
    return x + y - 2 * x * y


def nope(x):
    return 1 - x


def _pc_beta0(x, r):
    # note x and r are both binary tensors,
    # and dim -1 contains their bits
    # x will be shared, r will be public
    w = xor(x, r)
    z = r  - (x - 1)
    w_sum = torch.zeros(w.shape).type_as(w)
    for i in range(bits - 1, -1, -1):
        # FIXME: double check if keepdim should be True/False here
        w_sum[..., i] = w[..., (i + 1):].sum(dim=-1, keepdim=True)
    c = z + w_sum
    return c



def _pc_beta1(x, t):
    pass


def _pc_else():
    pass
