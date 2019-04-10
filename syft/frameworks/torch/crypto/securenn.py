"""
This is an implementation of the SecureNN paper
https://eprint.iacr.org/2018/442.pdf
"""

import torch

import syft as sy

# Q field
Q_BITS = 31


def decompose(tensor):
    """decompose a tensor into its binary representation."""
    n_bits = Q_BITS
    powers = torch.arange(n_bits)
    if hasattr(tensor, "child") and isinstance(tensor.child, dict):
        powers = powers.send(*list(tensor.child.keys())).child
    for i in range(len(tensor.shape)):
        powers = powers.unsqueeze(0)
    tensor = tensor.unsqueeze(-1)
    moduli = 2 ** powers
    tensor = torch.fmod(((tensor + 2 ** n_bits) / moduli.type_as(tensor)), 2)
    return tensor


def flip(x, dim):
    indices = torch.arange(x.shape[dim] - 1, -1, -1).long()

    if hasattr(x, "child") and isinstance(x.child, dict):
        indices = indices.send(*list(x.child.keys())).child

    return x.index_select(dim, indices)


def private_compare(x, r, BETA, j, alice, bob, crypto_provider):
    """
    Perform privately x > r

    args:
        x (AdditiveSharedTensor): the private tensor
        r (MultiPointerTensor): the threshold commonly held by alice and bob
        BETA (MultiPointerTensor): a boolean commonly held by alice and bob to
            hide the result of computation for the crypto provider
        j (MultiPointerTensor): a tensor with 0 and 1 to to the loop in a
            single pass
        alice (AbstractWorker): 1st worker holding a private share
        bob (AbstractWorker): 2nd worker holding a private share
        crypto_provider (AbstractWorker): the crypto_provider

    return:
        β′ = β ⊕ (x > r).
    """
    # the commented out numbers below correspond to the
    # line numbers in Algorithm 3 of the SecureNN paper
    # https://eprint.iacr.org/2018/442.pdf

    field = x.field

    # 1)
    t = torch.fmod((r + 1), field)

    # Mask for the case r == 2^l −1
    R_MASK = (r == (field - 1)).long()

    r = decompose(r)
    t = decompose(t)
    # Mask for beta
    BETA = BETA.unsqueeze(1).expand(list(r.shape))
    R_MASK = R_MASK.unsqueeze(1).expand(list(r.shape))

    u = (torch.rand(x.shape) > 0.5).long().send(bob, alice).child
    # Mask for condition i̸=1 in 11)
    l1_mask = torch.zeros(x.shape).long()
    l1_mask[:, -1:] = 1
    l1_mask = l1_mask.send(bob, alice).child

    # if BETA == 0
    # 5)
    w = (j * r) + x - (2 * x * r)

    # 6)
    wf = flip(w, 1)
    wfc = wf.cumsum(1) - wf
    wfcf = flip(wfc, 1)
    c_beta0 = (j * r) - x + j + wfcf

    # elif BETA == 1 AND r != 2^l- 1
    # 8)
    w = x + (j * t) - (2 * t * x)  # FIXME: unused
    # 9)
    c_beta1 = (-j * t) + x + j + wfcf

    # else
    # 11)
    c_igt1 = (1 - j) * (u + 1) - (j * u)
    c_ie1 = (j * -2) + 1
    c_21l = (l1_mask * c_ie1) + ((1 - l1_mask) * c_igt1)

    # Mask combination to execute the if / else statements of 4), 7), 10)
    c = (BETA * c_beta0) + (1 - BETA) * c_beta1
    c = (c * (1 - R_MASK)) + (c_21l * R_MASK)

    # 14)
    # Hide c values
    s = torch.randint(1, field, c.shape).send(alice, bob).child
    mask = s * c
    # Permute the mask
    perm = torch.randperm(c.shape[1]).send(alice, bob).child
    permuted_mask = mask[:, perm]
    # Send it to another worker
    remote_mask = permuted_mask.wrap().send(crypto_provider)
    # transform remotely the AdditiveSharingTensor back to a torch tensor
    d_ptr = remote_mask.remote_get()
    result_ptr = (d_ptr == 0).sum(1)
    # Get result back
    return result_ptr.get()


def msb(a_sh, alice, bob):
    """
    Compute the most significant bit in a_sh

    args:
        a_sh (AdditiveSharingTensor): the tensor of study
        alice (AbstractWorker): 1st worker holding a private share of a_sh
        bob (AbstractWorker): 2nd worker holding a private share

    return:
        the most significant bit
    """

    crypto_provider = a_sh.crypto_provider
    L = a_sh.field

    input_shape = a_sh.shape
    a_sh = a_sh.view(-1)

    # the commented out numbers below correspond to the
    # line numbers in Table 5 of the SecureNN paper
    # https://eprint.iacr.org/2018/442.pdf

    # 1)
    x = torch.LongTensor(a_sh.shape).random_(L - 1)
    x_bit = decompose(x)
    x_sh = x.share(bob, alice, crypto_provider=crypto_provider).child
    # Get last column / value as decompose reverts bits: first one is in last position
    x_bit_0 = x_bit[..., -1:]
    x_bit_sh_0 = x_bit_0.share(
        bob, alice, crypto_provider=crypto_provider
    ).child  # least -> greatest from left -> right
    x_bit_sh = x_bit.share(bob, alice, crypto_provider=crypto_provider).child

    # 2)
    y_sh = 2 * a_sh

    r_sh = y_sh + x_sh

    # 3)
    r = r_sh.get()  # .send(bob, alice) #TODO: make this secure by exchanging shares remotely
    r_0 = decompose(r)[..., -1].send(bob, alice).child
    r = r.send(bob, alice).child

    assert isinstance(r, sy.MultiPointerTensor)

    j0 = torch.zeros(x_bit_sh.shape).long().send(bob)
    j1 = torch.ones(x_bit_sh.shape).long().send(alice)
    j = sy.MultiPointerTensor(children=[j0, j1])
    j_0 = j[..., -1]

    assert isinstance(j, sy.MultiPointerTensor)
    assert isinstance(j_0, sy.MultiPointerTensor)

    # 4)
    BETA = (torch.rand(a_sh.shape) > 0.5).long().send(bob, alice).child
    BETA_prime = private_compare(
        x_bit_sh, r, BETA=BETA, j=j, alice=alice, bob=bob, crypto_provider=crypto_provider
    ).long()

    # 5)
    BETA_prime_sh = BETA_prime.share(bob, alice, crypto_provider=crypto_provider).child

    # 7)
    _lambda = BETA_prime_sh + (j_0 * BETA) - (2 * BETA * BETA_prime_sh)

    # 8)
    _delta = x_bit_sh_0.squeeze(-1) + (j_0 * r_0) - (2 * r_0 * x_bit_sh_0.squeeze(-1))

    # 9)
    theta = _lambda * _delta

    # 10)
    u = (
        torch.zeros(list(theta.shape))
        .long()
        .share(alice, bob, crypto_provider=crypto_provider)
        .child
    )
    a = _lambda + _delta - (2 * theta) + u

    return a.view(*list(input_shape))


def relu_deriv(a_sh):
    assert isinstance(a_sh, sy.AdditiveSharingTensor)

    workers = [sy.hook.local_worker.get_worker(w_name) for w_name in list(a_sh.child.keys())]
    return msb(a_sh, *workers)


def relu(a):
    return a * relu_deriv(a)
