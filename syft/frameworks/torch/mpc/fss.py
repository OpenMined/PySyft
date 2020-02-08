"""
This is an implementation of Function Secret Sharing

Useful papers are:
- Function Secret Sharing- Improvements and Extensions, Boyle 2017 https://eprint.iacr.org/2018/707.pdf
- Secure Computation with Preprocessing via Function Secret Sharing, Boyle 2019 https://eprint.iacr.org/2019/1095

Note that the protocols are quite different in aspect from those papers
"""
import torch as th
import syft as sy
from syft.messaging.plan import func2plan

λ = 63  # 6  # 63  # security parameter
n = 32  # 8  # 32  # bit precision

no_wrap = {"no_wrap": True}


def manual_init_store(worker):
    keygen = func2plan()(DPF.keygen)
    keygen.build()

    alpha, s_00, s_01, *CW = keygen()
    k = [(s_00, *CW), (s_01, *CW)]
    evaluate = func2plan()(DPF.eval)
    evaluate.build(th.IntTensor([0]), alpha, *k[0])

    keygen.owner = worker
    keygen.tag("#fss-keygen")
    evaluate.owner = worker
    evaluate.tag("#fss-eval")
    worker.register_obj(keygen)
    worker.register_obj(evaluate)

    keygen.forward = None
    evaluate.forward = None


def eq(x1, x2):

    # x1==x2 ~ x1-x2==0
    x_sh = x1 - x2

    locations = x_sh.locations
    crypto_provider = x_sh.crypto_provider

    # Retrieve keygen & eval plans
    keygen_ptr, = crypto_provider.search("#fss-keygen")
    evaluate_ptr, = crypto_provider.search("#fss-eval")

    # key gen
    alpha, s_00, s_01, *CW = keygen_ptr()

    # send mask
    alpha_sh = alpha.share(*locations, crypto_provider=crypto_provider, **no_wrap).get().child

    # reveal masked values
    x_masked = (x_sh + alpha_sh).get().send(*x_sh.locations, **no_wrap)

    alice, bob = locations

    k = [sy.MultiPointerTensor(children=[s_00.move(alice), s_01.move(bob)])] + [
        c.get().send(*locations, **no_wrap) for c in CW
    ]

    # Eval
    b = sy.MultiPointerTensor(
        children=[
            th.IntTensor([i]).send(location, **no_wrap) for i, location in enumerate(locations)
        ]
    )

    evaluate_ptr = evaluate_ptr.get().send(*locations)
    int_shares = evaluate_ptr(b, x_masked, *k)

    long_shares = {k: v.long() for k, v in int_shares.items()}
    response = sy.AdditiveSharingTensor(long_shares, **x_sh.get_class_attributes())

    return response


class DPF:
    def __init__(self):
        pass

    @staticmethod
    def keygen():
        beta = th.tensor([2], dtype=th.int32)
        alpha, = th.randint(0, 2 ** n, (1,))

        i = sy.ID_PROVIDER
        α = bit_decomposition(alpha)
        s, t, CW = Array(n + 1, 2, λ), Array(n + 1, 2), Array(n, 2 * (λ + 1))
        s[0] = randbit(size=(2, λ))
        t[0] = th.tensor([0, 1], dtype=th.uint8)
        for i in range(0, n):
            # Re-use useless randomness
            sL_0, _, sR_0, _ = split(G(s[i, 0]), [λ, 1, λ, 1])
            sL_1, _, sR_1, _ = split(G(s[i, 1]), [λ, 1, λ, 1])
            s_rand = (sL_0 ^ sL_1) * α[i] + (sR_0 ^ sR_1) * (1 - α[i])

            cw_i = TruthTableDPF(s_rand, α[i])
            CW[i] = cw_i ^ G(s[i, 0]) ^ G(s[i, 1])

            for b in (0, 1):
                τ = G(s[i, b]) ^ (t[i, b] * CW[i])
                τ = τ.reshape(2, λ + 1)
                s[i + 1, b], t[i + 1, b] = split(τ[𝛼[i]], [λ, 1])

        CW_n = (-1) ** t[n, 1] * (beta.to(th.uint8) - Convert(s[n, 0]) + Convert(s[n, 1]))

        return (alpha,) + s[0].unbind() + (CW, CW_n)

    @staticmethod
    def eval(b, x, *k_b):
        x = bit_decomposition(x)
        s, t = Array(n + 1, λ), Array(n + 1, 1)
        s[0] = k_b[0]
        # here k[1:] is (CW, CW_n)
        CW = k_b[1].unbind() + (k_b[2],)
        t[0] = b
        for i in range(0, n):
            τ = G(s[i]) ^ (t[i] * CW[i])
            τ = τ.reshape(2, λ + 1)
            s[i + 1], t[i + 1] = split(τ[x[i]], [λ, 1])
        return (-1) ** b * (Convert(s[n]) + t[n] * CW[n])


# PRG
def G(seed):
    assert len(seed) == λ
    th.manual_seed(Convert(seed))
    return th.randint(2, size=(2 * (λ + 1),), dtype=th.uint8)


# bit_pow_lambda = th.flip(2 ** th.arange(λ), (0,)).to(th.uint8)
def Convert(bits):
    bit_pow_lambda = th.flip(2 ** th.arange(λ), (0,)).to(th.uint8)
    return bits.dot(bit_pow_lambda)


def Array(*shape):
    return th.empty(shape, dtype=th.uint8)


# bit_pow_n = th.flip(2 ** th.arange(n), (0,))
def bit_decomposition(x):
    bit_pow_n = th.flip(2 ** th.arange(n), (0,))
    return ((x & bit_pow_n) > 0).to(th.int8)


def randbit(size):
    return th.randint(2, size=size)


def concat(*args, **kwargs):
    return th.cat(args, **kwargs)


def split(x, idx):
    return th.split(x, idx)


# one = th.tensor([1], dtype=th.uint8)
def TruthTableDPF(s, α_i):
    one = th.tensor([1], dtype=th.uint8)
    Table = th.zeros((2, λ + 1), dtype=th.uint8)
    Table[α_i] = concat(s, one)
    return Table.flatten()
