import torch
import syft as sy

BASE = 2
KAPPA = 3  # ~29 bits

# TODO set these intelligently
PRECISION_INTEGRAL = 2
PRECISION_FRACTIONAL = 0
PRECISION = PRECISION_INTEGRAL + PRECISION_FRACTIONAL
BOUND = BASE ** PRECISION

# Q field
Q_BITS = 31#32#62
field = (2 ** Q_BITS) - 1  # < 63 bits
Q_MAXDEGREE = 1
torch_max_value = torch.LongTensor([round(field/2)])
torch_field = torch.LongTensor([field])


def encode(rational, precision_fractional=PRECISION_FRACTIONAL, mod=field):
    upscaled = (rational * BASE ** precision_fractional).long()
    field_element = torch.fmod(upscaled, mod)
    return field_element


def decode(field_element, precision_fractional=PRECISION_FRACTIONAL, mod=field):
    neg_values = field_element.gt(mod)
    # pos_values = field_element.le(field)
    # upscaled = field_element*(neg_valuese+pos_values)
    field_element[neg_values] = mod - field_element[neg_values]
    rational = field_element.float() / BASE ** precision_fractional
    return rational

# I think decode() above may be wrong... and the correct one is below
# TODO: explore this

# def decode(field_element, precision_fractional=PRECISION_FRACTIONAL, mod=field):
#     value = field_element % field
#     gate = (value > torch_max_value).long()
#     neg_nums = (value - spdz.torch_field) * gate
#     pos_nums = value * (1 - gate)
#     result = (neg_nums + pos_nums).float() / (BASE ** precision_fractional)
#     return result


def share(secret, n_workers, mod=field, random_type=torch.LongTensor):

    random_shares = [random_type(secret.get_shape()) for i in range(n_workers - 1)]

    for share in random_shares:
        share.random_(mod)

    shares = []
    for i in range(n_workers):
        if i == 0:
            share = random_shares[i]
        elif i < n_workers - 1:
            share = random_shares[i] - random_shares[i-1]
        else:
            share = secret - random_shares[i-1]
        shares.append(share)

    return shares


def reconstruct(shares, mod=field):
    return sum(shares) % mod


def swap_shares(shares):
    ptd = shares.child.pointer_tensor_dict
    alice, bob = list(ptd.keys())
    new_alice = (ptd[alice]+0)
    new_bob = (ptd[bob]+0)
    new_alice.send(bob)
    new_bob.send(alice)

    return sy._GeneralizedPointerTensor({alice: new_bob,bob: new_alice}).on(sy.LongTensor([]))


def truncate(x, interface, amount=PRECISION_FRACTIONAL, mod=field):
    print("truncating")
    if (interface.get_party() == 0):
        return torch.fmod((x / BASE ** amount), mod)
    return torch.fmod((mod - ((mod - x) / BASE ** amount)), mod)


def public_add(x, y, interface):
    if (interface.get_party() == 0):
        return (x + y)
    elif (interface.get_party() == 1):
        return x


def spdz_add(a, b, mod=field):
    c = a + b
    return torch.fmod(c, mod)


def spdz_neg(a, mod=field):
    return torch.fmod((mod - a), mod)


def spdz_mul(x, y, workers, mod=field):
    if x.get_shape() != y.get_shape():
        raise ValueError("Shapes must be identical in order to multiply them")
    shape = x.get_shape()
    triple = generate_mul_triple_communication(shape, workers)
    a, b, c = triple

    d = torch.fmod((x - a), mod)
    e = torch.fmod((y - b), mod)

    delta = torch.fmod(d.child.sum_get(), mod)
    epsilon = torch.fmod(e.child.sum_get(), mod)

    epsilon_delta = epsilon * delta

    delta = delta.broadcast(workers)
    epsilon = epsilon.broadcast(workers)

    z = torch.fmod((c
                    + torch.fmod((delta * b), mod)
                    + torch.fmod((epsilon * a), mod)
                    ), mod)

    z.child.public_add_(epsilon_delta)

    return z


def spdz_matmul(x, y, workers, mod=field):
    shapes = [x.get_shape(), y.get_shape()]
    if len(x.get_shape()) != 1:
        x_width = x.get_shape()[1]
    else:
        x_width = 1

    y_height = y.get_shape()[0]

    assert x_width == y_height, 'dimension mismatch: %r != %r' % (
        x_width, y_height,
    )
    a, b, c = generate_matmul_triple_communication(shapes, workers)

    r = torch.fmod((x - a), mod)
    s = torch.fmod((y - b), mod)

    # Communication
    rho = torch.fmod(r.child.sum_get(), mod)
    sigma = torch.fmod(s.child.sum_get(), mod)
    rho_sigma = torch.fmod(torch.mm(rho, sigma), mod)

    rho = rho.broadcast(workers)
    sigma = sigma.broadcast(workers)

    a_sigma = torch.fmod(torch.mm(a, sigma), mod)
    rho_b = torch.fmod(torch.mm(rho, b), mod)

    z = torch.fmod((a_sigma + rho_b + c), mod)
    z.child.public_add_(rho_sigma)

    return z

    # # we assume we need to mask the result for a third party crypto provider
    # u = generate_zero_shares_communication(alice, bob, *share.shape)
    # return spdz_add(share, u)


def spdz_sigmoid(x, interface):
    W0, W1, W3, W5 = generate_sigmoid_shares_communication(x, interface)
    x2 = spdz_mul(x, x, interface)
    x3 = spdz_mul(x, x2, interface)
    x5 = spdz_mul(x3, x2, interface)
    temp5 = spdz_mul(x5, W5, interface)
    temp3 = spdz_mul(x3, W3, interface)
    temp1 = spdz_mul(x, W1, interface)
    temp53 = spdz_add(temp5, temp3)
    temp531 = spdz_add(temp53, temp1)
    return spdz_add(W0, temp531)


def get_ptrdict(mpct):
    child = mpct
    while not isinstance(child, sy.core.frameworks.torch._GeneralizedPointerTensor) and child is not None:
        child = child.child
    if child is None:
        raise TypeError("Expected child tree to contain a GeneralizedPointerTensor")
    return child.pointer_tensor_dict


def get_workers(mpct):
    child = mpct
    while not isinstance(child, sy.core.frameworks.torch._GeneralizedPointerTensor) and child is not None:
        child = child.child
    if child is None:
        raise TypeError("Expected child tree to contain a GeneralizedPointerTensor")
    return list(child.pointer_tensor_dict.keys())


def get_shape(mpct):
    child = mpct
    while not isinstance(child, sy.core.frameworks.torch._GeneralizedPointerTensor) and child is not None:
        child = child.child
    if child is None:
        raise TypeError("Expected child tree to contain a GeneralizedPointerTensor")
    return child.shape


def generate_mul_triple(shape, mod=field):
    r = torch.LongTensor(shape).random_(mod)
    s = torch.LongTensor(shape).random_(mod)
    t = r * s
    return r, s, t


def generate_mul_triple_communication(shape, workers):
    r, s, t = generate_mul_triple(shape)

    n_workers = len(workers)
    r_shares = share(r, n_workers)
    s_shares = share(s, n_workers)
    t_shares = share(t, n_workers)

    # For r, s, t as a shared var, send each share to its worker
    for var_shares in [r_shares, s_shares, t_shares]:
        for var_share, worker in zip(var_shares, workers):
            var_share.send(worker)

    # Build the pointer dict for r, s, t. Note that we remove the head of the pointer (via .child)
    gp_r = sy._GeneralizedPointerTensor({
        share.location: share.child for share in r_shares
    }).on(r)
    gp_s = sy._GeneralizedPointerTensor({
        share.location: share.child for share in s_shares
    }).on(s)
    gp_t = sy._GeneralizedPointerTensor({
        share.location: share.child for share in t_shares
    }).on(t)
    triple = [gp_r, gp_s, gp_t]
    return triple


def generate_zero_shares_communication(alice, bob, *sizes):
    return torch.zeros(*sizes).fix_precision().share(alice, bob)

def generate_one_shares_communication(alice, bob, sizes):
    return torch.ones(sizes).long().share(alice, bob)


def generate_matmul_triple(shapes, mod=field):
    r = torch.LongTensor(shapes[0]).random_(mod)
    s = torch.LongTensor(shapes[1]).random_(mod)
    t = torch.mm(r, s)
    assert t.shape == (shapes[0][0], shapes[1][1]), (t.shape, (shapes[0][0], shapes[1][1]), 'mismatch')
    return r, s, t


def generate_matmul_triple_communication(shapes, workers):
    r, s, t = generate_matmul_triple(shapes)

    n_workers = len(workers)
    r_shares = share(r, n_workers)
    s_shares = share(s, n_workers)
    t_shares = share(t, n_workers)

    # For r, s, t as a shared var, send each share to its worker
    for var_shares in [r_shares, s_shares, t_shares]:
        for var_share, worker in zip(var_shares, workers):
            var_share.send(worker)

    # Build the pointer dict for r, s, t. Note that we remove the head of the pointer (via .child)
    gp_r = sy._GeneralizedPointerTensor({
        share.location: share.child for share in r_shares
    }).on(r)
    gp_s = sy._GeneralizedPointerTensor({
        share.location: share.child for share in s_shares
    }).on(s)
    gp_t = sy._GeneralizedPointerTensor({
        share.location: share.child for share in t_shares
    }).on(t)
    triple = [gp_r, gp_s, gp_t]
    return triple


def generate_sigmoid_shares_communication(x, interface):
    if (interface.get_party() == 0):
        W0 = encode(torch.FloatTensor(x.shape).one_() * 1 / 2)
        W1 = encode(torch.FloatTensor(x.shape).one_() * 1 / 4)
        W3 = encode(torch.FloatTensor(x.shape).one_() * -1 / 48)
        W5 = encode(torch.FloatTensor(x.shape).one_() * 1 / 480)

        W0_alice, W0_bob = share(W0)
        W1_alice, W1_bob = share(W1)
        W3_alice, W3_bob = share(W3)
        W5_alice, W5_bob = share(W5)

        swap_shares(W0_bob, interface)
        swap_shares(W1_bob, interface)
        swap_shares(W3_bob, interface)
        swap_shares(W5_bob, interface)

        quad_alice = [W0_alice, W1_alice, W3_alice, W5_alice]
        return quad_alice
    elif (interface.get_party() == 1):
        W0_bob = swap_shares(
            torch.LongTensor(
                x.shape,
            ).zero_(), interface,
        )
        W1_bob = swap_shares(
            torch.LongTensor(
                x.shape,
            ).zero_(), interface,
        )
        W3_bob = swap_shares(
            torch.LongTensor(
                x.shape,
            ).zero_(), interface,
        )
        W5_bob = swap_shares(
            torch.LongTensor(
                x.shape,
            ).zero_(), interface,
        )
        quad_bob = [W0_bob, W1_bob, W3_bob, W5_bob]
        return quad_bob
