import torch

BASE = 10
KAPPA = 3  # ~29 bits

# TODO set these intelligently
PRECISION_INTEGRAL = 2
PRECISION_FRACTIONAL = 0
PRECISION = PRECISION_INTEGRAL + PRECISION_FRACTIONAL
BOUND = BASE ** PRECISION

# Q field
field = 2 ** 31 - 1  # < 64 bits
# Q = 2147483648
Q_MAXDEGREE = 1


def encode(rational, precision_fractional=PRECISION_FRACTIONAL):
    upscaled = (rational * BASE ** precision_fractional).long()
    field_element = upscaled % field
    return field_element


def decode(field_element, precision_fractional=PRECISION_FRACTIONAL):
    field_element = field_element
    neg_values = field_element.gt(field)
    # pos_values = field_element.le(field)
    # upscaled = field_element*(neg_valuese+pos_values)
    field_element[neg_values] = field - field_element[neg_values]
    rational = field_element.float() / BASE ** precision_fractional
    return rational


def share(secret):
    first = torch.LongTensor(secret.shape).random_(field)
    second = (secret - first) % field
    return [first, second]


def reconstruct(shares):
    return sum(shares) % field


def swap_shares(share, interface):
    share_other = torch.LongTensor(share.shape).zero_()
    if (interface.get_party() == 0):
        interface.send(share)
        share_other = interface.recv(share_other)
    elif (interface.get_party() == 1):
        share_other = interface.recv(share_other)
        interface.send(share)
    return share_other


def truncate(x, interface, amount=PRECISION_FRACTIONAL):
    if (interface.get_party() == 0):
        return (x / BASE ** amount) % field
    return (field - ((field - x) / BASE ** amount)) % field


def public_add(x, y, interface):
    if (interface.get_party() == 0):
        return (x + y)
    elif (interface.get_party() == 1):
        return x


def spdz_add(a, b):
    return ((a + b) % field)


def spdz_neg(a):
    return (field - a) % field


def generate_mul_triple(m, n):
    r = torch.LongTensor(m, n).random_(field)
    s = torch.LongTensor(m, n).random_(field)
    t = r * s
    return r, s, t


def generate_mul_triple_communication(m, n, interface):
    if (interface.get_party() == 0):
        r, s, t = generate_mul_triple(m, n)

        r_alice, r_bob = share(r)
        s_alice, s_bob = share(s)
        t_alice, t_bob = share(t)

        swap_shares(r_bob, interface)
        swap_shares(s_bob, interface)
        swap_shares(t_bob, interface)

        triple_alice = [r_alice, s_alice, t_alice]
        return triple_alice
    elif (interface.get_party() == 1):
        r_bob = swap_shares(torch.LongTensor(m, n).zero_(), interface)
        s_bob = swap_shares(torch.LongTensor(m, n).zero_(), interface)
        t_bob = swap_shares(torch.LongTensor(m, n).zero_(), interface)
        triple_bob = [r_bob, s_bob, t_bob]
        return triple_bob


def spdz_mul(x, y, interface):
    if x.shape != y.shape:
        raise ValueError()
    m, n = x.shape
    triple = generate_mul_triple_communication(m, n, interface)
    a, b, c = triple
    d = (x - a) % field
    e = (y - b) % field

    d_other = swap_shares(d, interface)
    e_other = swap_shares(e, interface)
    delta = (d + d_other) % field
    epsilon = (e + e_other) % field
    r = delta * epsilon
    s = a * epsilon
    t = b * delta
    share = s + t + c
    share = public_add(share, r, interface)
    share = truncate(share, interface)
    return share


def generate_matmul_triple(m, n, k):
    r = torch.LongTensor(m, k).random_(field)
    s = torch.LongTensor(k, n).random_(field)
    t = r * s
    return r, s, t


def generate_matmul_triple_communication(m, n, k, interface):
    if (interface.get_party() == 0):
        r, s, t = generate_matmul_triple(m, n, k)
        r_alice, r_bob = share(r)
        s_alice, s_bob = share(s)
        t_alice, t_bob = share(t)

        swap_shares(r_bob, interface)
        swap_shares(s_bob, interface)
        swap_shares(t_bob, interface)

        triple_alice = [r_alice, s_alice, t_alice]
        return triple_alice
    elif (interface.get_party() == 1):
        r_bob = swap_shares(torch.LongTensor(m, k).zero_(), interface)
        s_bob = swap_shares(torch.LongTensor(k, n).zero_(), interface)
        t_bob = swap_shares(torch.LongTensor(m, n).zero_(), interface)
        triple_bob = [r_bob, s_bob, t_bob]
        return triple_bob


def spdz_matmul(x, y, interface):
    x_height = x.shape[0]
    if len(x.shape) != 1:
        x_width = x.shape[1]
    else:
        x_width = 1

    y_height = y.shape[0]
    if len(y.shape) != 1:
        y_width = y.shape[1]
    else:
        y_width = 1

    assert x_width == y_height, 'dimension mismatch: %r != %r' % (x_width, y_height)

    r, s, t = generate_matmul_triple_communication(
        x_height, y_width, x_width, interface)

    rho_local = (x - r) % field
    sigma_local = (y - s) % field

    # Communication
    rho_other = swap_shares(rho_local, interface)
    sigma_other = swap_shares(sigma_local, interface)

    # They both add up the shares locally
    rho = reconstruct([rho_local, rho_other])
    sigma = reconstruct([sigma_local, sigma_other])

    r_sigma = r * sigma
    rho_s = rho * s

    share = r_sigma + rho_s + t

    rs = rho * sigma

    share = public_add(share, rs, interface)
    share = truncate(share, interface)
    return share


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
        W0_bob = swap_shares(torch.LongTensor(
            x.shape).zero_(), interface)
        W1_bob = swap_shares(torch.LongTensor(
            x.shape).zero_(), interface)
        W3_bob = swap_shares(torch.LongTensor(
            x.shape).zero_(), interface)
        W5_bob = swap_shares(torch.LongTensor(
            x.shape).zero_(), interface)
        quad_bob = [W0_bob, W1_bob, W3_bob, W5_bob]
        return quad_bob


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
