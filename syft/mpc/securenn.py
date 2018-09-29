# An implementation of the SecureNN protocols from Wagh et al.

from syft.spdz.spdz import (spdz_add, spdz_mul,
                           generate_zero_shares_communication,
                           get_ptrdict,
                           Q_BITS, field)
from syft.core.frameworks.torch.tensor import _GeneralizedPointerTensor, _SPDZTensor
from syft.core.frameworks.torch.utils import chain_print
import torch

L = field
p = field  # 67 in original # TODO: extend to ops over multiple rings


# spdz_params.append((params[remote_index][param_i].data+0).fix_precision().share(bob, alice).get())

def select_shares(alpha, x, y, workers):
    """
    alpha is a shared binary tensor
    x and y are private tensors to choose elements or slices from
        (following broadcasting rules)

    all of type _GeneralizedPointerTensor

    Computes z = (1 - alpha) * x + alpha * y
    """
    # FIXME: generate_zero_shares_communication should be updated with new pysyft API
    u = generate_zero_shares_communication(*workers, *x.get_shape())
    z = x + alpha * (y - x)
    return z + u


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


def flip(x, dim):
    indices = torch.arange(x.get_shape()[dim] - 1, -1, -1).long()

    if hasattr(x.child, 'pointer_tensor_dict'):
        indices = indices.send(*list(x.child.pointer_tensor_dict.keys()))

    return x.index_select(dim, indices)


def private_compare(x, r, BETA, alice, bob):
    l = Q_BITS

    workers = (alice, bob)

    t = (r + 1) % 2 ** l

    R_MASK = r == ((2 ** l) - 1)

    x = decompose(torch.LongTensor([x])).share(bob, alice).child.child
    r = decompose(torch.LongTensor([r])).send(bob, alice)
    t = decompose(torch.LongTensor([t])).send(bob, alice)
    BETA = torch.LongTensor([BETA]).unsqueeze(1).expand(r.get_shape()).send(bob, alice)
    R_MASK = torch.LongTensor([R_MASK]).unsqueeze(1).expand(r.get_shape()).send(bob, alice)
    u = (torch.rand(x.get_shape()) > 0.5).long().send(bob, alice)
    l1_mask = torch.zeros(x.get_shape()).long()
    l1_mask[:, -1] = 1
    l1_mask = l1_mask.send(bob, alice)

    j0 = torch.zeros(x.get_shape()).long().send(bob).child
    j1 = (torch.ones(x.get_shape())).long().send(alice).child
    j = _GeneralizedPointerTensor({bob: j0, alice: j1}, torch_type='syft.LongTensor').wrap(True)

    # if BETA == 0
    w = (j * r) + x - (2 * x * r)

    wf = flip(w, 1)
    wfc = wf.cumsum(1) - wf
    wfcf = flip(wfc, 1)

    c_beta0 = ((j * r) - x + j + wfcf)

    # elif BETA == 1 AND r != 2**Q_BITS - 1
    w = x + (j * t) - (2 * t * x)
    c_beta1 = (-j * t) + x + j + wfcf

    # else
    c_igt1 = (1 - j) * (u + 1) - (j * u)
    c_ie1 = (j * -2) + 1
    c_21l = (l1_mask * c_ie1) + ((1 - l1_mask) * c_igt1)

    c = (BETA * c_beta0) + (1 - BETA) * c_beta1
    c = (c * (1 - R_MASK)) + (c_21l * R_MASK)

    cmpc = _SPDZTensor(c).wrap(True).get()  # /2
    result = (cmpc == 0).sum(1)
    return result


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
    print('pcelse sz',*sizes)
    u = generate_zero_shares_communication(*workers, *sizes)
    print('beepbpp')
    chain_print(u)
    u_ptrdict = get_ptrdict(u)
    (w0, u0), (w1, u1) = u_ptrdict.items()
    print('BOOOP')
    chain_print(u0)
    c0 = torch.zeros(*u.get_shape()).long() # c0 = u0 * 0
    c1 = torch.zeros(*u.get_shape()).long()
    for i in range(Q_BITS - 2, -1, -1):
        if i == 0:
            c0[..., i] = -1 * u0
            c1[..., i] = u1
        c0[..., i] = u0 + 1
        c1[..., i] = -1 * u1
    ptr_dict = {w0:c0, w1:c1}
    c_gp = _GeneralizedPointerTensor(ptr_dict, torch_type='syft.LongTensor').wrap(True)
    c = _SPDZTensor(x_gp, torch_type='syft.LongTensor').wrap(True)
    return c
