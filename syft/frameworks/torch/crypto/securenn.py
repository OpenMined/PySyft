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

no_wrap = {"no_wrap": True}


def decompose(tensor):
    """decompose a tensor into its binary representation."""
    n_bits = Q_BITS
    powers = torch.arange(n_bits)
    if hasattr(tensor, "child") and isinstance(tensor.child, dict):
        powers = powers.send(*list(tensor.child.keys()), **no_wrap)
    for i in range(len(tensor.shape)):
        powers = powers.unsqueeze(0)
    tensor = tensor.unsqueeze(-1)
    moduli = 2 ** powers
    tensor = torch.fmod((tensor / moduli.type_as(tensor)), 2)
    return tensor


def flip(x, dim):
    """
    Reverse the order of the elements in a tensor
    """
    indices = torch.arange(x.shape[dim] - 1, -1, -1).long()

    if hasattr(x, "child") and isinstance(x.child, dict):
        indices = indices.send(*list(x.child.keys()), **no_wrap)

    return x.index_select(dim, indices)


def _random_common_bit(*workers):
    """
    Return a bit chosen by a worker and sent to all workers,
    in the form of a MultiPointerTensor
    """
    pointer = torch.LongTensor([1]).send(workers[0], **no_wrap).random_(2)
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
    pointer = torch.LongTensor([1]).send(workers[0], **no_wrap).random_(max_value)
    pointers = [pointer]
    for worker in workers[1:]:
        pointers.append(pointer.copy().move(worker))
    common_value = sy.MultiPointerTensor(children=pointers)

    return common_value


def _shares_of_zero(size, field, crypto_provider, *workers):
    """
    Return n in [0, max_value-1] chosen by a worker and sent to all workers,
    in the form of a MultiPointerTensor
    """
    u = (
        torch.zeros(size)
        .long()
        .send(workers[0])
        .share(*workers, field=field, crypto_provider=crypto_provider, **no_wrap)
        .get()
        .child
    )

    return u


def select_share(alpha_sh, x_sh, y_sh):
    """ Performs select share protocol
    If the bit alpha_sh is 0, x_sh is returned
    If the bit alpha_sh is 1, y_sh is returned

    Args:
        x_sh (AdditiveSharingTensor): the first share to select
        y_sh (AdditiveSharingTensor): the second share to select
        alpha_sh (AdditiveSharingTensor): the bit to choose between x_sh and y_sh

    Return:
        z_sh = (1 - alpha_sh) * x_sh + alpha_sh * y_sh
    """
    alice, bob = alpha_sh.locations
    crypto_provider = alpha_sh.crypto_provider
    L = alpha_sh.field

    u_sh = _shares_of_zero(1, L, crypto_provider, alice, bob)

    # 1)
    w_sh = y_sh - x_sh

    # 2)
    c_sh = alpha_sh * w_sh

    # 3)
    z_sh = x_sh + c_sh + u_sh

    return z_sh


def private_compare(x_bit_sh, r, beta):
    """
    Perform privately x > r

    args:
        x (AdditiveSharedTensor): the private tensor
        r (MultiPointerTensor): the threshold commonly held by alice and bob
        beta (MultiPointerTensor): a boolean commonly held by alice and bob to
            hide the result of computation for the crypto provider

    return:
        β′ = β ⊕ (x > r).
    """
    assert isinstance(x_bit_sh, sy.AdditiveSharingTensor)
    assert isinstance(r, sy.MultiPointerTensor)
    assert isinstance(beta, sy.MultiPointerTensor)
    # Would it be safer to have a different r/beta for each value in the tensor?

    alice, bob = x_bit_sh.locations
    crypto_provider = x_bit_sh.crypto_provider
    p = x_bit_sh.field
    L = 2 ** Q_BITS  # 2**l

    # the commented out numbers below correspond to the
    # line numbers in Algorithm 3 of the SecureNN paper
    # https://eprint.iacr.org/2018/442.pdf

    # Common randomess
    s = torch.randint(1, p, x_bit_sh.shape).send(alice, bob, **no_wrap)
    u = torch.randint(1, p, x_bit_sh.shape).send(alice, bob, **no_wrap)
    perm = torch.randperm(x_bit_sh.shape[-1]).send(alice, bob, **no_wrap)

    j = sy.MultiPointerTensor(
        children=[torch.tensor([0]).send(alice, **no_wrap), torch.tensor([1]).send(bob, **no_wrap)]
    )

    # 1)
    t = (r + 1) % L
    t_bit = decompose(t)
    r_bit = decompose(r)

    # if beta == 0
    # 5)
    w = x_bit_sh + (j * r_bit) - (2 * r_bit * x_bit_sh)
    # 6)
    wc = w.flip(-1).cumsum(-1).flip(-1) - w
    c_beta0 = -x_bit_sh + (j * r_bit) + j + wc

    # elif beta == 1 AND r != 2^l- 1
    # 8)
    w = x_bit_sh + (j * t_bit) - (2 * t_bit * x_bit_sh)
    # 9)
    wc = w.flip(-1).cumsum(-1).flip(-1) - w
    c_beta1 = x_bit_sh + (-j * t_bit) + j + wc

    # else
    # 11)
    c_igt1 = (1 - j) * (u + 1) - (j * u)
    c_ie1 = (1 - 2 * j) * u

    l1_mask = torch.zeros(x_bit_sh.shape).long()
    l1_mask[..., 0] = 1
    l1_mask = l1_mask.send(alice, bob, **no_wrap)
    # c_else = if i == 1 c_ie1 else c_igt1
    c_else = (l1_mask * c_ie1) + ((1 - l1_mask) * c_igt1)

    # Mask for the case r == 2^l −1
    r_mask = (r == (L - 1)).long()
    r_mask = r_mask.unsqueeze(-1)

    # Mask combination to execute the if / else statements of 4), 7), 10)
    c = (1 - beta) * c_beta0 + (beta * (1 - r_mask)) * c_beta1 + (beta * r_mask) * c_else

    # 14)
    # Hide c values
    mask = s * c

    # Permute the mask
    # I have to create idx because Ellipsis are still not supported
    # (I would like to do permuted_mask = mask[..., perm])
    idx = [slice(None)] * (len(x_bit_sh.shape) - 1) + [perm]
    permuted_mask = mask[idx]
    # Send it to another worker
    # We do this because we can't allow the local worker to get and see permuted_mask
    # because otherwise it can inverse the permutation and remove s to get c.
    # So opening the permuted_mask should be made by a worker which doesn't have access to the randomness
    remote_mask = permuted_mask.wrap().send(crypto_provider, **no_wrap)

    # 15)
    d_ptr = remote_mask.remote_get()
    beta_prime = (d_ptr == 0).sum(-1)

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
    beta = _random_common_bit(alice, bob)
    u = _shares_of_zero(1, L, crypto_provider, alice, bob)

    # 1)
    x = torch.LongTensor(a_sh.shape).random_(L - 1)
    x_bit = decompose(x)
    x_sh = x.share(bob, alice, field=L - 1, crypto_provider=crypto_provider, **no_wrap)
    x_bit_0 = x_bit[..., 0]
    x_bit_sh_0 = x_bit_0.share(bob, alice, field=L, crypto_provider=crypto_provider, **no_wrap)
    x_bit_sh = x_bit.share(bob, alice, field=p, crypto_provider=crypto_provider, **no_wrap)

    # 2)
    y_sh = a_sh * 2
    r_sh = y_sh + x_sh

    # 3)
    r = r_sh.reconstruct() % (L - 1)  # convert an additive sharing in multi pointer Tensor
    r_0 = decompose(r)[..., 0]

    # 4)
    beta_prime = private_compare(x_bit_sh, r, beta=beta)

    # 5)
    beta_prime_sh = beta_prime.share(
        bob, alice, field=L, crypto_provider=crypto_provider, **no_wrap
    )

    # 7)
    j = sy.MultiPointerTensor(
        children=[torch.tensor([0]).send(alice, **no_wrap), torch.tensor([1]).send(bob, **no_wrap)]
    )
    gamma = beta_prime_sh + (j * beta) - (2 * beta * beta_prime_sh)

    # 8)
    delta = x_bit_sh_0 + (j * r_0) - (2 * r_0 * x_bit_sh_0)

    # 9)
    theta = gamma * delta

    # 10)
    a = gamma + delta - (theta * 2) + u

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

    alpha0 = (
        (r_shares[workers[0].id] + r_shares[workers[1].id].copy().move(workers[0])) >= L
    ).long()
    alpha1 = alpha0.copy().move(workers[1])
    alpha = sy.MultiPointerTensor(children=[alpha0, alpha1])

    u_sh = _shares_of_zero(1, L - 1, crypto_provider, *workers)

    # 2)
    a_tilde_sh = a_sh + r_sh
    a_shares = a_sh.child
    beta0 = ((a_shares[workers[0].id] + r_shares[workers[0].id]) >= L).long()
    beta1 = ((a_shares[workers[1].id] + r_shares[workers[1].id]) >= L).long()
    beta = sy.MultiPointerTensor(children=[beta0, beta1])

    # 4)
    a_tilde_shares = a_tilde_sh.child
    delta = (
        (a_tilde_shares[workers[0].id].copy().get() + a_tilde_shares[workers[1].id].copy().get())
        >= L
    ).long()
    x = a_tilde_sh.get() % L

    # 5)
    x_bit = decompose(x)
    x_bit_sh = x_bit.share(*workers, field=p, crypto_provider=crypto_provider, **no_wrap)
    delta_sh = delta.share(*workers, field=L - 1, crypto_provider=crypto_provider, **no_wrap)

    # 6)
    eta_p = private_compare(x_bit_sh, r - 1, eta_pp)
    # 7)
    eta_p_sh = eta_p.share(*workers, field=L - 1, crypto_provider=crypto_provider, **no_wrap)

    # 9)
    j = sy.MultiPointerTensor(
        children=[
            torch.tensor([0]).send(workers[0], **no_wrap),
            torch.tensor([1]).send(workers[1], **no_wrap),
        ]
    )
    eta_sh = eta_p_sh + (1 - j) * eta_pp - 2 * eta_pp * eta_p_sh

    # 10)
    theta_sh = beta - (1 - j) * (alpha + 1) + delta_sh + eta_sh

    # 11)
    y_sh = -theta_sh + a_sh + u_sh
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
    u = _shares_of_zero(1, L, crypto_provider, alice, bob)

    # 1)
    y_sh = a_sh * 2

    # 2) Not applicable with algebraic shares
    y_sh = share_convert(y_sh)
    # y_sh.field = L - 1

    # 3)
    alpha_sh = msb(y_sh)
    assert alpha_sh.field == L

    # 4)
    j = sy.MultiPointerTensor(
        children=[torch.tensor([0]).send(alice, **no_wrap), torch.tensor([1]).send(bob, **no_wrap)]
    )
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
    u = _shares_of_zero(1, L, crypto_provider, alice, bob)

    return a_sh * relu_deriv(a_sh) + u


def division(x_sh, y_sh, bit_len_max=Q_BITS):
    """ Performs division of encrypted numbers
    
    Args:
        x_sh, y_sh (AdditiveSharingTensor): the private tensors on which the op applies
        bit_len_max: the number of bits needed to represent the highest value in the tensors
            we may want to avoid giving this value so Q_BITS is the default value

    Returns:
        element-wise integer division of x_sh by y_sh
    """
    alice, bob = x_sh.locations
    crypto_provider = x_sh.crypto_provider
    L = x_sh.field

    x_shape = x_sh.shape
    y_shape = y_sh.shape
    assert x_shape == y_shape

    x_sh = x_sh.view(-1)
    y_sh = y_sh.view(-1)

    # Common Randomness
    w_sh = _shares_of_zero(bit_len_max, L, crypto_provider, alice, bob)
    s_sh = _shares_of_zero(1, L, crypto_provider, alice, bob)
    u_sh = _shares_of_zero(1, L, crypto_provider, alice, bob)

    ks = []
    for i in range(bit_len_max - 1, -1, -1):
        # 3)
        z_sh = x_sh - u_sh - 2 ** i * y_sh + w_sh[i]

        # 4)
        beta_sh = relu_deriv(z_sh)

        # 5)
        v_sh = beta_sh * (2 ** i * y_sh)

        # 6)
        k_sh = beta_sh * 2 ** i
        ks.append(k_sh)

        # 7)
        u_sh = u_sh + v_sh

    # 9)
    q = sum(ks) + s_sh

    if len(x_shape):
        return q.view(*x_shape)
    else:
        return q


def maxpool(x_sh):
    """ Compute MaxPool: returns fresh shares of the max value in the input tensor
    and the index of this value in the flattened tensor
    
    Args:
        x_sh (AdditiveSharingTensor): the private tensor on which the op applies
        
    Returns:
        maximum value as an AdditiveSharingTensor
        index of this value in the flattened tensor as an AdditiveSharingTensor
    """
    alice, bob = x_sh.locations
    crypto_provider = x_sh.crypto_provider
    L = x_sh.field

    input_shape = x_sh.shape
    x_sh = x_sh.view(-1)

    # Common Randomness
    u_sh = _shares_of_zero(1, L, crypto_provider, alice, bob)
    v_sh = _shares_of_zero(1, L, crypto_provider, alice, bob)

    # 1)
    max_sh = x_sh[0]
    ind_sh = torch.tensor([0]).share(
        alice, bob, crypto_provider=crypto_provider, **no_wrap
    )  # I did not manage to create an AST with 0 and 0 as shares

    for i in range(1, len(x_sh)):
        # 3)
        w_sh = x_sh[i] - max_sh

        # 4)
        beta_sh = relu_deriv(w_sh)

        # 5)
        max_sh = select_share(beta_sh, max_sh, x_sh[i])

        # 6)
        k = torch.tensor([i]).share(
            alice, bob, crypto_provider=crypto_provider, **no_wrap
        )  # I did not manage to create an AST with 0 and i as shares

        # 7)
        ind_sh = select_share(beta_sh, ind_sh, k)

    return max_sh + u_sh, ind_sh + v_sh


def maxpool_deriv(x_sh):
    """ Compute derivative of MaxPool

    Args:
        x_sh (AdditiveSharingTensor): the private tensor on which the op applies

    Returns:
        an AdditiveSharingTensor of the same shape as x_sh full of zeros except for
        a 1 at the position of the max value
    """
    alice, bob = x_sh.locations
    crypto_provider = x_sh.crypto_provider
    L = x_sh.field

    n1, n2 = x_sh.shape
    n = n1 * n2
    x_sh = x_sh.view(-1)

    # Common Randomness
    U_sh = _shares_of_zero(n, L, crypto_provider, alice, bob)
    r = _random_common_value(L, alice, bob)

    # 1)
    _, ind_max_sh = maxpool(x_sh)

    # 2)
    j = sy.MultiPointerTensor(
        children=[torch.tensor([1]).send(alice, **no_wrap), torch.tensor([0]).send(bob, **no_wrap)]
    )
    k_sh = ind_max_sh + j * r

    # 3)
    t = k_sh.get()
    k = t % n
    E_k = torch.zeros(n)
    E_k[k] = 1
    E_sh = E_k.share(alice, bob, **no_wrap)

    # 4)
    g = r % n
    D_sh = torch.roll(E_sh, -g)

    maxpool_d_sh = D_sh + U_sh
    return maxpool_d_sh.view(n1, n2)
