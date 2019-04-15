"""
This is an implementation of the SecureNN paper
https://eprint.iacr.org/2018/442.pdf

Note that there is a difference here in that our shares can be
negative numbers while they are always positive in the paper
"""

import torch

import syft as sy


# p is introduced in the SecureNN paper https://eprint.iacr.org/2018/442.pdf
# it is a small field for efficient additive sharing
p = 67

# Q field
Q_BITS = 62


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
    """
    Reverse the order of the elements in a tensor
    """
    indices = torch.arange(x.shape[dim] - 1, -1, -1).long()

    if hasattr(x, "child") and isinstance(x.child, dict):
        indices = indices.send(*list(x.child.keys())).child

    return x.index_select(dim, indices)


def _random_common_bit(*workers):
    """
    Return a bit chosen by a worker and sent to all workers,
    in the form of a MultiPointerTensor
    """
    pointer = torch.LongTensor([1]).send(workers[0]).random_(2)
    pointers = [pointer]
    for worker in workers[1:]:
        pointers.append(pointer.copy().move(worker))
    bit = sy.MultiPointerTensor(children=pointers)

    return bit


def _random_common_value(max_value, *workers):
    """
    Return n in [0, max_value-1] chosen by a worker and sent to all workers,
    in the form of a MultiPointerTensor
    """
    pointer = torch.LongTensor([1]).send(workers[0]).random_(max_value)
    pointers = [pointer]
    for worker in workers[1:]:
        pointers.append(pointer.copy().move(worker))
    common_value = sy.MultiPointerTensor(children=pointers)

    return common_value


def private_compare(x, r, BETA):
    """
    Perform privately x > r

    args:
        x (AdditiveSharedTensor): the private tensor
        r (MultiPointerTensor): the threshold commonly held by alice and bob
        BETA (MultiPointerTensor): a boolean commonly held by alice and bob to
            hide the result of computation for the crypto provider

    return:
        β′ = β ⊕ (x > r).
    """
    assert isinstance(x, sy.AdditiveSharingTensor)
    assert isinstance(r, sy.MultiPointerTensor)
    assert isinstance(BETA, sy.MultiPointerTensor)

    alice, bob = x.locations
    crypto_provider = x.crypto_provider
    p = x.field
    L = 2 ** Q_BITS  # 2**l

    # the commented out numbers below correspond to the
    # line numbers in Algorithm 3 of the SecureNN paper
    # https://eprint.iacr.org/2018/442.pdf

    # 1)
    t = (r + 1) % L

    # Mask for the case r == 2^l −1
    R_MASK = (r == (L - 1)).long()

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
    j0 = torch.zeros(x.shape).long().send(bob)
    j1 = torch.ones(x.shape).long().send(alice)
    j = sy.MultiPointerTensor(children=[j0, j1])
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
    c = (1 - BETA) * c_beta0 + BETA * c_beta1
    c = (c * (1 - R_MASK)) + (c_21l * R_MASK)

    # 14)
    # Hide c values
    s = torch.randint(1, p, c.shape).send(alice, bob).child
    mask = s * c
    # Permute the mask
    perm = torch.randperm(c.shape[1]).send(alice, bob).child
    permuted_mask = mask[:, perm]
    # Send it to another worker
    remote_mask = permuted_mask.wrap().send(crypto_provider)

    # 15)
    # transform remotely the AdditiveSharingTensor back to a torch tensor
    d_ptr = remote_mask.remote_get()
    beta_prime = (d_ptr == 0).sum(1)
    # Get result back
    res = beta_prime.get()
    return res


def msb(a_sh):
    """
    Compute the most significant bit in a_sh, this is an implementation of the
    SecureNN paper https://eprint.iacr.org/2018/442.pdf

    Args:
        a_sh (AdditiveSharingTensor): the tensor of study
    Return:
        the most significant bit
    """

    alice, bob = a_sh.locations
    crypto_provider = a_sh.crypto_provider
    L = a_sh.field + 1  # field of a is L - 1

    input_shape = a_sh.shape
    a_sh = a_sh.view(-1)

    # the commented out numbers below correspond to the
    # line numbers in Table 5 of the SecureNN paper
    # https://eprint.iacr.org/2018/442.pdf

    # Common Randomness
    BETA = _random_common_bit(alice, bob)
    u = torch.zeros(1).long().share(alice, bob, field=L, crypto_provider=crypto_provider).child

    # 1)
    x = torch.LongTensor(a_sh.shape).random_(L - 1)
    x_bit = decompose(x)
    x_sh = x.share(bob, alice, field=L - 1, crypto_provider=crypto_provider).child
    x_bit_0 = x_bit[
        ..., -1
    ]  # Get last value as decompose reverts bits: 1st one is in last position
    x_bit_sh_0 = x_bit_0.share(
        bob, alice, field=L, crypto_provider=crypto_provider
    ).child  # least -> greatest from left -> right
    x_bit_sh = x_bit.share(bob, alice, field=p, crypto_provider=crypto_provider).child

    # 2)
    y_sh = 2 * a_sh
    r_sh = y_sh + x_sh

    # 3)
    r = r_sh.reconstruct()  # convert an additive sharing in multi pointer Tensor
    r_0 = decompose(r)[..., -1]

    # 4)
    BETA_prime = private_compare(x_bit_sh, r, BETA=BETA)

    # 5)
    BETA_prime_sh = BETA_prime.share(bob, alice, field=L, crypto_provider=crypto_provider).child

    # 7)
    j = sy.MultiPointerTensor(children=[torch.tensor([0]).send(alice), torch.tensor([1]).send(bob)])
    gamma = BETA_prime_sh + (j * BETA) - (2 * BETA * BETA_prime_sh)

    # 8)
    delta = x_bit_sh_0 + (j * r_0) - (2 * r_0 * x_bit_sh_0)

    # 9)
    theta = gamma * delta

    # 10)
    a = gamma + delta - (2 * theta) + u

    if len(input_shape):
        return a.view(*list(input_shape))
    else:
        return a


def share_convert(a_sh):
    """
    Convert shares of a in field L to shares of a in field L - 1

    Args:
        a_sh (AdditiveSharingTensor): the additive sharing tensor who owns
            the shares in field L to convert

    Return:
        An additive sharing tensor with shares in field L-1
    """
    assert isinstance(a_sh, sy.AdditiveSharingTensor)

    workers = a_sh.locations
    crypto_provider = a_sh.crypto_provider
    L = a_sh.field

    # Common randomness
    eta_pp = _random_common_bit(*workers)
    r = _random_common_value(L, *workers)

    # Share remotely r
    r_sh = (
        (r * 1)
        .child[workers[0].id]
        .share(*workers, field=L, crypto_provider=crypto_provider)
        .get()
        .child
    )
    r_shares = r_sh.child
    alpha = (
        ((r_shares[workers[0].id] + (r_shares[workers[1].id] * 1).move(workers[0])).get() >= L)
        .long()
        .send(*workers)
        .child
    )  # FIXME security issue: the local worker learns alpha while this should be avoided
    u_sh = (
        torch.zeros(1)
        .long()
        .send(workers[0])
        .share(*workers, field=L - 1, crypto_provider=crypto_provider)
        .get()
        .child
    )

    # 2)
    a_tilde_sh = a_sh + r_sh
    a_shares = a_sh.child
    ptr0 = a_shares[workers[0].id] + r_shares[workers[0].id]
    beta0 = ((a_shares[workers[0].id] + r_shares[workers[0].id]) >= L).long() - (
        (a_shares[workers[0].id] + r_shares[workers[0].id]) < 0
    ).long()
    ptr1 = a_shares[workers[1].id] + r_shares[workers[1].id]
    beta1 = ((a_shares[workers[1].id] + r_shares[workers[1].id]) >= L).long() - (
        (a_shares[workers[1].id] + r_shares[workers[1].id]) < 0
    ).long()
    beta = sy.MultiPointerTensor(children=[beta0.long(), beta1.long()])

    # 4)
    a_tilde_shares = a_tilde_sh.child
    delta = (
        ((a_tilde_shares[workers[0].id] * 1).get() + (a_tilde_shares[workers[1].id] * 1).get()) >= L
    ).long()
    x = a_tilde_sh.get()

    # 5)
    x_bit = decompose(x)
    x_bit_sh = x_bit.share(*workers, field=p, crypto_provider=crypto_provider).child
    delta_sh = delta.share(*workers, field=L - 1, crypto_provider=crypto_provider).child

    # 6)
    eta_p = private_compare(x_bit_sh, r, eta_pp)

    # 7)
    eta_p_sh = eta_p.share(*workers, field=L - 1, crypto_provider=crypto_provider).child

    # 9)
    j = sy.MultiPointerTensor(
        children=[torch.tensor([0]).send(workers[0]), torch.tensor([1]).send(workers[1])]
    )
    eta_sh = eta_p_sh + (1 - j) * eta_pp - 2 * eta_pp * eta_p_sh

    # 10)
    theta_sh = beta - (1 - j) * (alpha + 1) + delta_sh + eta_sh

    # 11)
    y_sh = a_sh - theta_sh + u_sh
    y_sh.field = L - 1
    return y_sh


def relu_deriv(a_sh):
    """
    Compute the derivative of Relu

    Args:
        a_sh (AdditiveSharingTensor): the private tensor on which the op applies

    Returns:
        0 if Dec(a_sh) < 0
        1 if Dec(a_sh) > 0
        encrypted in an AdditiveSharingTensor
    """

    alice, bob = a_sh.locations
    crypto_provider = a_sh.crypto_provider
    L = a_sh.field

    # Common randomness
    u = (
        torch.zeros(1)
        .long()
        .send(alice)
        .share(alice, bob, field=L, crypto_provider=crypto_provider)
        .get()
        .child
    )

    # 1)
    y_sh = 2 * a_sh

    # 2) Not applicable with algebraic shares
    # y_sh = share_convert(y_sh)
    y_sh.field = L - 1

    # 3)
    alpha_sh = msb(y_sh)
    assert alpha_sh.field == L

    # 4)
    j = sy.MultiPointerTensor(children=[torch.tensor([0]).send(alice), torch.tensor([1]).send(bob)])
    gamma_sh = j - alpha_sh + u
    assert gamma_sh.field == L
    return gamma_sh


def relu(a_sh):
    """
    Compute Relu

    Args:
        a_sh (AdditiveSharingTensor): the private tensor on which the op applies

    Returns:
        Dec(a_sh) > 0
        encrypted in an AdditiveSharingTensor
    """

    alice, bob = a_sh.locations
    crypto_provider = a_sh.crypto_provider
    L = a_sh.field

    # Common Randomness
    u = torch.zeros(1).long().share(alice, bob, field=L, crypto_provider=crypto_provider).child

    return a_sh * relu_deriv(a_sh) + u
