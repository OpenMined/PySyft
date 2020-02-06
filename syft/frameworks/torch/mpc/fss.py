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

λ = 63  # security parameter
n = 32  # bit precision

no_wrap = {"no_wrap": True}


def eq(x1, x2):

    # TODO put this outside
    keygen = func2plan()(DPF.keygen)
    beta = th.IntTensor([2])
    alpha, = th.randint(0, 2 ** n, (1,))
    keygen.build(alpha, beta)
    s_00, s_01, *CW = keygen(alpha, beta)
    k = [(s_00, *CW), (s_01, *CW)]
    evaluate = func2plan()(DPF.eval)
    evaluate.build(th.IntTensor([0]), alpha, *k[0])


    # x1==x2 : x1-x2==0
    x_sh = x1 - x2

    locations = x_sh.locations
    crypto_provider = x_sh.crypto_provider

    # build mask
    alpha, = th.randint(0, 2 ** n, (1,))
    alpha_sh = alpha.share(*locations, crypto_provider=crypto_provider, **no_wrap)

    # reveal masked values
    x_masked = (x_sh + alpha_sh).get().send(*x_sh.locations, **no_wrap)

    # key gen
    alpha = alpha.send(crypto_provider)
    beta = th.IntTensor([1]).send(crypto_provider)
    keygen_ptr = keygen.send(crypto_provider)
    s_00, s_01, *CW = keygen_ptr(alpha, beta)

    alice, bob = locations
    k = [
        sy.MultiPointerTensor(children=[s_00.move(alice), s_01.move(bob)])
    ] + [
        c.send(*locations, **no_wrap) for c in CW
    ]

    # k = [
    #     (s_00.move(alice), *[c.copy().move(alice) for c in CW]),
    #     (s_01.move(bob), *[c.copy().move(bob) for c in CW])
    # ]

    # Eval
    b = sy.MultiPointerTensor(children=[
            th.IntTensor([i]).send(location **no_wrap)
            for i, location in enumerate(locations)
    ])

    eval = DPF.eval.send(*locations)

    response = eval(b, x_masked, *k)

    # eval_ptr_alice(indices[0], public_x[0], *k[0]).get() + \
    # eval_ptr_bob(indices[1], public_x[1], *k[1]).get()



class DPF:
    def __init__(self):
        pass

    @staticmethod
    def keygen(alpha, beta):
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

        return s[0].unbind() + CW.unbind() + (CW_n,)

    @staticmethod
    def eval(b, x, *k_b):
        x = bit_decomposition(x)
        s, t = Array(n+1, λ), Array(n+1, 1)
        s[0] = k_b[0]
        CW = k_b[1:]
        t[0] = b
        for i in range(0, n):
            τ = G(s[i]) ^ (t[i]*CW[i])
            τ = τ.reshape(2, λ+1)
            s[i+1], t[i+1] = split(τ[x[i]], [λ, 1])
        return (-1)**b * (Convert(s[n]) + t[n]*CW[n])


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