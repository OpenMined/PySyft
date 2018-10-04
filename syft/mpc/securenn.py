# An implementation of the SecureNN protocols from Wagh et al.

from syft.spdz.spdz import (Q_BITS, field)
import syft
import torch

L = field
p = field

def decompose(tensor):
    """
    decompose a tensor into its binary representation
    """
    powers = torch.arange(Q_BITS)
    if hasattr(tensor.child, 'pointer_tensor_dict'):
        powers.send(*list(tensor.child.pointer_tensor_dict.keys()))
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


def private_compare(x, r, BETA, j, alice, bob):

    l = Q_BITS

    t = (r + 1) % 2 ** l

    R_MASK = (r == ((2 ** l) - 1)).long()

    x = x.child.child
    r = decompose(r)
    t = decompose(t)
    BETA = BETA.unsqueeze(1).expand(list(r.get_shape()))
    R_MASK = R_MASK.unsqueeze(1).expand(list(r.get_shape()))
    u = (torch.rand(x.get_shape()) > 0.5).long().send(bob, alice)
    l1_mask = torch.zeros(x.get_shape()).long()
    l1_mask[:, -1:] = 1
    l1_mask = l1_mask.send(bob, alice)

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

    cmpc = syft._SNNTensor(c).wrap(True).get()  # /2
    result = (cmpc == 0).sum(1)
    return result


def msb(a_sh, alice, bob):

    input_shape = a_sh.get_shape()
    a_sh = a_sh.view(-1)

    # the commented out numbers below correspond to the
    # line numbers in Table 5 of the SecureNN paper
    # https://eprint.iacr.org/2018/442.pdf

    # 1)

    x = torch.LongTensor(a_sh.get_shape()).random_(L - 1)
    x_bit = decompose(x)
    x_sh = x.share(bob, alice)
    x_bit_0 = x_bit[..., -1:]  # pretty sure decompose is backwards...
    x_bit_sh_0 = x_bit_0.share(bob, alice).child.child  # least -> greatest from left -> right
    x_bit_sh = x_bit.share(bob, alice)

    # 2)
    y_sh = 2 * a_sh
    r_sh = y_sh + x_sh

    # 3)
    r = r_sh.get()  # .send(bob, alice) #TODO: make this secure by exchanging shares remotely
    r_0 = decompose(r)[..., -1].send(bob, alice)
    r = r.send(bob, alice)

    j0 = torch.zeros(x_bit_sh.get_shape()).long().send(bob).child
    j1 = (torch.ones(x_bit_sh.get_shape())).long().send(alice).child
    j = syft._GeneralizedPointerTensor({bob: j0, alice: j1}, torch_type='syft.LongTensor').wrap(True)
    j_0 = j[..., -1]

    # 4)
    BETA = (torch.rand(a_sh.get_shape()) > 0.5).long().send(bob, alice)
    BETA_prime = private_compare(x_bit_sh,
                                 r,
                                 BETA=BETA,
                                 j=j,
                                 alice=alice,
                                 bob=bob).long()
    # 5)
    BETA_prime_sh = BETA_prime.share(bob, alice).child.child

    # 7)
    _lambda = syft._SNNTensor(BETA_prime_sh + (j_0 * BETA) - (2 * BETA * BETA_prime_sh)).wrap(True)

    # 8)
    _delta = syft._SNNTensor(x_bit_sh_0.squeeze(-1) + (j_0 * r_0) - (2 * r_0 * x_bit_sh_0.squeeze(-1))).wrap(True)

    # 9)
    theta = _lambda * _delta

    # 10)
    u = torch.zeros(list(theta.get_shape())).long().share(alice, bob)
    a = _lambda + _delta - (2 * theta) + u

    return a.view(*list(input_shape))

def relu_deriv(a_sh):
    return msb(a_sh, *list(a_sh.child.shares.child.pointer_tensor_dict.keys()))

def relu(a):
    return a * relu_deriv(a)

