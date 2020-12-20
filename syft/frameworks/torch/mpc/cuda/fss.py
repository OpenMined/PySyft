"""
This is an implementation of Function Secret Sharing

Useful papers are:
- Function Secret Sharing- Improvements and Extensions, Boyle 2017
  Link: https://eprint.iacr.org/2018/707.pdf
- Secure Computation with Preprocessing via Function Secret Sharing, Boyle 2019
  Link: https://eprint.iacr.org/2019/1095

Note that the protocols are quite different in aspect from those papers
"""
import math
import asyncio

import torch as th
import syft as sy

from syft.exceptions import EmptyCryptoPrimitiveStoreError
from syft.workers.websocket_client import WebsocketClientWorker
from syft.generic.utils import allow_command
from syft.generic.utils import remote

if th.cuda.is_available():
    import torchcsprng as csprng

    generator = csprng.create_random_device_generator("/dev/urandom")

λ = 127  # security parameter
n = 32  # bit precision
λs = math.ceil(λ / 64)  # how many int64 are needed to store λ, here 2
assert λs == 2

no_wrap = {"no_wrap": True}

# internal codes
EQ = 0
COMP = 1


def full_name(f):
    return f"syft.frameworks.torch.mpc.cuda.fss.{f.__name__}"


def keygen(n_values, op):
    """
    Run FSS keygen in parallel to accelerate the offline part of the protocol

    Args:
        n_values (int): number of primitives to generate
        op (str): eq or comp <=> DPF or DIF
    """
    if op == "eq":
        return DPF.keygen(n_values=n_values)
    elif op == "comp":
        return DIF.keygen(n_values)
    else:
        raise ValueError


def fss_op(x1, x2, op="eq"):
    """
    Define the workflow for a binary operation using Function Secret Sharing

    Currently supported operand are = & <=, respectively corresponding to
    op = 'eq' and 'comp'

    Args:
        x1: first AST
        x2: second AST
        op: type of operation to perform, should be 'eq' or 'comp'

    Returns:
        shares of the comparison
    """

    assert th.cuda.is_available()

    if isinstance(x1, sy.AdditiveSharingTensor):
        locations = x1.locations
        class_attributes = x1.get_class_attributes()
    else:
        locations = x2.locations
        class_attributes = x2.get_class_attributes()

    dtype = class_attributes.get("dtype")
    asynchronous = isinstance(locations[0], WebsocketClientWorker)

    workers_args = [
        (
            x1.child[location.id]
            if isinstance(x1, sy.AdditiveSharingTensor)
            else (x1 if i == 0 else 0),
            x2.child[location.id]
            if isinstance(x2, sy.AdditiveSharingTensor)
            else (x2 if i == 0 else 0),
            op,
        )
        for i, location in enumerate(locations)
    ]

    try:
        shares = []
        for i, location in enumerate(locations):
            share = remote(mask_builder, location=location)(*workers_args[i], return_value=True)
            shares.append(share)
    except EmptyCryptoPrimitiveStoreError as e:
        if sy.local_worker.crypto_store.force_preprocessing:
            raise
        sy.local_worker.crypto_store.provide_primitives(workers=locations, kwargs_={}, **e.kwargs_)
        return fss_op(x1, x2, op)

    # async has a cost which is too expensive for this command
    # shares = asyncio.run(sy.local_worker.async_dispatch(
    #     workers=locations,
    #     commands=[
    #         (full_name(mask_builder), None, workers_args[i], {})
    #         for i in [0, 1]
    #     ],
    #     return_value=True
    # ))

    mask_value = sum(shares) % 2 ** n

    for location, share in zip(locations, shares):
        location.de_register_obj(share)
        del share

    workers_args = [(th.tensor([i], device="cuda").int(), mask_value, op, dtype) for i in range(2)]
    if not asynchronous:
        shares = []
        for i, location in enumerate(locations):
            share = remote(evaluate, location=location)(*workers_args[i], return_value=False)
            shares.append(share)
    else:
        print("async")
        shares = asyncio.run(
            sy.local_worker.async_dispatch(
                workers=locations,
                commands=[(full_name(evaluate), None, workers_args[i], {}) for i in [0, 1]],
            )
        )

    shares = {loc.id: share for loc, share in zip(locations, shares)}

    response = sy.AdditiveSharingTensor(shares, **class_attributes)
    return response


# share level
@allow_command
def mask_builder(x1, x2, op):
    if not isinstance(x1, int):
        worker = x1.owner
        numel = x1.numel()
    else:
        worker = x2.owner
        numel = x2.numel()
    x = x1 - x2
    # Keep the primitive in store as we use it after
    # you actually get a share of alpha
    alpha, s_0, *CW = worker.crypto_store.get_keys(f"fss_{op}", n_instances=numel, remove=False)
    r = x + alpha.type(th.long).reshape(x.shape)
    return r


# share level
@allow_command
def evaluate(b, x_masked, op, dtype):
    if op == "eq":
        return eq_evaluate(b, x_masked)
    elif op == "comp":
        return comp_evaluate(b, x_masked, dtype=dtype)
    else:
        raise ValueError


# process level
def eq_evaluate(b, x_masked):
    alpha, s_0, *CW = x_masked.owner.crypto_store.get_keys(
        op="fss_eq", n_instances=x_masked.numel(), remove=True
    )
    return DPF.eval(b, x_masked, s_0, *CW)


# process level
def comp_evaluate(b, x_masked, dtype=None):
    alpha, s_0, *CW = x_masked.owner.crypto_store.get_keys(
        op="fss_comp", n_instances=x_masked.numel(), remove=True
    )
    result = DIF.eval(b, x_masked, s_0, *CW)

    # dtype_options = {None: th.long, "int": th.int32, "long": th.long}
    # result = th.tensor(result, dtype=dtype_options[dtype], device="cuda")

    return result


def eq(x1, x2):
    return fss_op(x1, x2, "eq")


def le(x1, x2):
    return fss_op(x1, x2, "comp")


class DPF:
    """Distributed Point Function - used for equality"""

    @staticmethod
    def keygen(n_values=1):
        alpha = th.empty(n_values, dtype=th.long, device="cuda").random_(
            0, 2 ** n, generator=generator
        )
        beta = th.tensor([1], device="cuda")
        α = bit_decomposition(alpha)
        s, t, CW = (
            Array(n + 1, 2, λs, n_values),
            Array(n + 1, 2, n_values),
            Array(n, 2, (λs + 1), n_values),
        )
        _CW = []
        s[0] = randbit(shape=(2, λ, n_values))
        t[0] = th.tensor([[0, 1]] * n_values, dtype=th.long, device="cuda").t()
        for i in range(0, n):
            g0 = G(s[i, 0])
            g1 = G(s[i, 1])
            # Re-use useless randomness
            sL_0, _, sR_0, _ = split(g0, (EQ, λs, 1, λs, 1))
            sL_1, _, sR_1, _ = split(g1, (EQ, λs, 1, λs, 1))
            s_rand = (sL_0 ^ sL_1) * α[i] + (sR_0 ^ sR_1) * (1 - α[i])

            cw_i = SwitchTableDPF(s_rand, α[i])
            CW[i] = cw_i ^ g0 ^ g1
            _CW.append(compress(CW[i], α[i], op=EQ))
            CWi = uncompress(_CW[i])

            for b in (0, 1):
                dual_state = [g0, g1][b] ^ (t[i, b] * CWi)
                state = multi_dim_filter(dual_state, α[i])
                s[i + 1, b], t[i + 1, b] = split(state, (EQ, λs, 1))
        CW_n = (-1) ** t[n, 1] * (beta - convert(s[n, 0]) + convert(s[n, 1]))
        CW_n = CW_n.type(th.long)
        return (alpha, s[0][0], s[0][1], *_CW, CW_n)

    @staticmethod
    def eval(b, x, *k_b):
        x = x.long()
        original_shape = x.shape
        x = x.reshape(-1)
        n_values = x.shape[0]
        x = bit_decomposition(x)
        s, t = Array(n + 1, λs, n_values), Array(n + 1, 1, n_values)
        s[0], *_CW, _CWn = k_b
        t[0] = b
        for i in range(0, n):
            CWi = uncompress(_CW[i])
            dual_state = G(s[i]) ^ (t[i] * CWi)
            state = multi_dim_filter(dual_state, x[i])
            s[i + 1], t[i + 1] = split(state, (EQ, λs, 1))

        flat_result = (-1) ** b * (t[n].squeeze() * _CWn + convert(s[n]))
        return flat_result.type(th.long).reshape(original_shape)


class DIF:
    """Distributed Point Function - used for comparison"""

    @staticmethod
    def keygen(n_values=1):
        alpha = th.empty(n_values, dtype=th.long, device="cuda").random_(
            0, 2 ** n, generator=generator
        )
        α = bit_decomposition(alpha)

        s, σ, t, τ, CW, CW_leaf = (
            Array(n + 1, 2, λs, n_values),
            Array(n + 1, 2, 1, n_values),  # the sigma are on n bits -> one 64bit block is enough
            Array(n + 1, 2, n_values),
            Array(n + 1, 2, n_values),
            Array(n, 2, 2 * λs + 1, n_values),  # update accordingly
            Array(n + 1, n_values),
        )
        _CW = []
        s[0] = randbit(shape=(2, λ, n_values))
        t[0] = th.tensor([[0, 1]] * n_values, dtype=th.long, device="cuda").t()

        for i in range(0, n):
            h0 = H(s[i, 0], 0)
            h1 = H(s[i, 1], 1)
            # Re-use useless randomness
            σL_0, sL_0, σR_0, sR_0 = split(h0, (COMP, 1, λs, 1, λs))  # modified!
            σL_1, sL_1, σR_1, sR_1 = split(h1, (COMP, 1, λs, 1, λs))
            s_rand = (sL_0 ^ sL_1) * α[i] + (sR_0 ^ sR_1) * (1 - α[i])
            σ_rand = (σL_0 ^ σL_1) * α[i] + (σR_0 ^ σR_1) * (1 - α[i])
            cw_i = SwitchTableDIF(s_rand, σ_rand, α[i])
            CW[i] = cw_i ^ h0 ^ h1
            _CW.append(compress(CW[i], α[i], op=COMP))
            CWi = uncompress(_CW[i], op=COMP)

            for b in (0, 1):
                dual_state = [h0, h1][b] ^ (t[i, b] * CWi)
                # the state obtained by following the special path
                state = multi_dim_filter(dual_state, α[i])
                _, _, s[i + 1, b], t[i + 1, b] = split(state, (COMP, 1, 1, λs, 1))
                # the state obtained by leaving the special path
                anti_state = multi_dim_filter(dual_state, 1 - α[i])
                σ[i + 1, b], τ[i + 1, b], _, _ = split(anti_state, (COMP, 1, 1, λs, 1))

                if b:
                    # note that we subtract (1 - α[i]), so that leaving the special path can't lead
                    # to an output == 1 when α[i] == 0 (because it means that your bit is 1 so your
                    # value is > α)
                    CW_leaf[i] = (1 - 2 * τ[i + 1, 1]) * (
                        1 - convert(σ[i + 1, 0]) + convert(σ[i + 1, 1]) - (1 - α[i])
                    )

        CW_leaf[n] = 1 - 2 * t[n, 1]  # * (1 - convert(s[n, 0]) + convert(s[n, 1]))

        CW_leaf = CW_leaf.type(th.long)

        return (alpha, s[0][0], s[0][1], *_CW, CW_leaf)

    @staticmethod
    def eval(b, x, *k_b):
        x = x.long()
        original_shape = x.shape
        x = x.reshape(-1)
        n_values = x.shape[0]
        x = bit_decomposition(x)
        s, σ, t, τ = (
            Array(n + 1, λs, n_values),
            Array(n + 1, 1, n_values),  # the sigma are on n bits -> one 64bit block is enough
            Array(n + 1, 1, n_values),
            Array(n + 1, 1, n_values),
        )
        s[0], *_CW, CW_leaf = k_b
        CW_leaf = CW_leaf.type(th.long)
        t[0] = b

        out = 0

        for i in range(0, n):
            CWi = uncompress(_CW[i], op=COMP)
            dual_state = H(s[i]) ^ (t[i] * CWi)
            state = multi_dim_filter(dual_state, x[i])
            σ[i + 1], τ[i + 1], s[i + 1], t[i + 1] = split(state, (COMP, 1, 1, λs, 1))
            out += (-1) ** b * (τ[i + 1] * CW_leaf[i] + convert(σ[i + 1]))

        # Last node, the other σ is also a leaf
        out += (-1) ** b * (t[n].squeeze() * CW_leaf[n])  # + convert(s[n]))

        return out.type(th.long).reshape(original_shape)


def compress(CWi, alpha_i, op=EQ):
    """Compression technique which reduces the size of the CWi by dropping some
    non-necessary randomness.

    The original paper on FSS explains that this trick doesn't affect the security.
    """
    if op == EQ:
        sL, tL, sR, tR = split(CWi, (op, λs, 1, λs, 1))
        return (tL.type(th.bool), tR.type(th.bool), (1 - alpha_i) * sR + alpha_i * sL)
    else:
        σL, τL, sL, tL, σR, τR, sR, tR = split(CWi, (op, 1, 1, λs, 1, 1, 1, λs, 1))
        return (
            τL.type(th.bool),
            tL.type(th.bool),
            τR.type(th.bool),
            tR.type(th.bool),
            alpha_i * σR + (1 - alpha_i) * σL,
            (1 - alpha_i) * sR + alpha_i * sL,
        )


def uncompress(_CWi, op=EQ):
    """Decompress the compressed CWi by duplicating the randomness to recreate
    the original shape."""
    if op == EQ:
        CWi = concat(
            _CWi[2],
            _CWi[0].reshape(1, -1).type(th.long),
            _CWi[2],
            _CWi[1].reshape(1, -1).type(th.long),
        ).reshape(2, 3, -1)
    else:
        CWi = concat(
            _CWi[4],
            _CWi[0].reshape(1, -1).type(th.long),
            _CWi[5],
            _CWi[1].reshape(1, -1).type(th.long),
            _CWi[4],
            _CWi[2].reshape(1, -1).type(th.long),
            _CWi[5],
            _CWi[3].reshape(1, -1).type(th.long),
        ).reshape(2, 5, -1)
    return CWi


def Array(*shape):
    return th.empty(shape, dtype=th.long, device="cuda")


if th.cuda.is_available():
    bit_pow_n = th.flip(2 ** th.arange(n, device="cuda"), (0,))


def bit_decomposition(x):
    x = x.unsqueeze(-1)
    z = bit_pow_n & x
    z = z.t()
    return (z > 0).to(th.uint8)


def randbit(shape):
    assert len(shape) == 3
    byte_dim = shape[-2]
    shape_with_bytes = shape[:-2] + (math.ceil(byte_dim / 64), shape[-1])
    randvalues = th.empty(*shape_with_bytes, dtype=th.long, device="cuda").random_(
        -(2 ** 63), 2 ** 63 - 1, generator=generator
    )
    # randvalues = th.randint(-2 ** 63, 2 ** 63 - 1, shape_with_bytes, dtype=th.long, device="cuda")
    # randvalues[:, 0] = randvalues[:, 0] % 2 ** (byte_dim % 64)
    return randvalues


def concat(*args, **kwargs):
    return th.cat(args, **kwargs)


split_helpers = {
    (EQ, 2, 1): lambda x: (x[:2], x[2]),
    (EQ, 2, 1, 2, 1): lambda x: (x[0, :2], x[0, 2], x[1, :2], x[1, 2]),
    (COMP, 1, 1, 2, 1): lambda x: (x[:1], x[1], x[2:4], x[4]),
    (COMP, 1, 2, 1, 2): lambda x: (
        x[0, :1],
        # x[0, 1],
        x[0, 2:4],
        # x[0, 4],
        x[1, :1],
        # x[1, 1],
        x[1, 2:4],
        # x[1, 4],
    ),
    (COMP, 1, 1, 2, 1, 1, 1, 2, 1): lambda x: (
        x[0, :1],
        x[0, 1],
        x[0, 2:4],
        x[0, 4],
        x[1, :1],
        x[1, 1],
        x[1, 2:4],
        x[1, 4],
    ),
}


def split(list_, idx):
    return split_helpers[idx](list_)


ones_dict = {}
ones_dict2 = {}


def SwitchTableDPF(s, α_i):
    if s.shape not in ones_dict:
        ones_dict[s.shape] = th.ones((1, s.shape[1]), device="cuda", dtype=th.long)

    one = ones_dict[s.shape]
    s_one = concat(s, one)

    if s_one.shape not in ones_dict2:
        ones_dict2[s_one.shape] = th.ones((1, *s_one.shape), dtype=th.long, device="cuda")

    ones = ones_dict2[s_one.shape]
    pad = (α_i * ones).type(th.long)
    pad = concat(1 - pad, pad, axis=0)
    Table = pad * s_one
    return Table


def SwitchTableDIF(s, σ, α_i):
    leafTable = SwitchTableDPF(σ, 1 - α_i)
    nextTable = SwitchTableDPF(s, α_i)

    Table = concat(leafTable, nextTable, axis=1)
    return Table


def multi_dim_filter(τ, idx):
    filtered_τ = (1 - idx) * τ[0] + idx * τ[1]
    return filtered_τ


def convert(x):
    """
    convert a multi dim big tensor to a "random" single tensor
    """
    # Select the 31st least significant bits
    r = x[-1] & 0b1111_1111_1111_1111_1111_1111_1111_111
    return r.type(th.long)


# PRG


if th.cuda.is_available():
    # TODO
    keys = []
    for _ in range(4):
        key = th.tensor([4192687974420127714, 651949261783629689], device="cuda", dtype=th.long)
        keys.append(key)
    # key = th.tensor(
    #    [224, 28, 13, 108, 97, 35, 195, 240, 14, 221, 233, 215, 0, 67, 174, 129], dtype=th.uint8
    # )


def split_last_bit(buffer):
    # Numbers are on 64 bits signed
    return buffer.native_abs(), buffer.native_ge(0)


def G(seed):
    # print('G', seed.shape)
    assert len(seed.shape) == 2
    n_values = seed.shape[1]
    assert seed.shape[0] == λs

    buffers = []
    for i in range(2):
        key = keys[i]
        buffer = th.empty(λs, n_values, dtype=th.long, device="cuda")
        csprng.encrypt(seed, buffer, key, "aes128", "ecb")
        buffers.append(buffer)

    valuebits = th.empty(2, 3, n_values, dtype=th.long, device="cuda")
    valuebits[0, 0], last_bit = split_last_bit(buffers[0][0])
    valuebits[0, 1] = buffers[0][1]
    valuebits[0, 2] = last_bit
    valuebits[1, 0], last_bit = split_last_bit(buffers[1][0])
    valuebits[1, 1] = buffers[1][1]
    valuebits[1, 2] = last_bit

    # seed = seed  # .cuda()
    # urandom_gen = csprng.create_const_generator(key)
    # mask = th.empty(2 * λs, n_values, dtype=th.long, device="cuda").random_(generator=urandom_gen)
    # repl_seed = seed.repeat(2, 1)
    # # print('mask, repl_seed', mask.shape, repl_seed.shape)
    # buffer = mask + repl_seed
    # valuebits = th.empty(2, 3, n_values, dtype=th.long, device="cuda")
    #
    # # [λ, 1, λ, 1]
    # # [λ - 64, 64, 1, λ - 64, 64, 1]
    # valuebits[0, 0], last_bit = split_last_bit(buffer[0])
    # valuebits[0, 1] = buffer[1]
    # valuebits[0, 2] = last_bit
    # valuebits[1, 0], last_bit = split_last_bit(buffer[2])
    # valuebits[1, 1] = buffer[3]
    # valuebits[1, 2] = last_bit

    return valuebits


H_cache = {}
valuebits_cache = {}


def H(seed, idx=0):
    """
    Pseudo Random Generator λ -> 4(λ + 1)

    idx is here to allow not reusing the same empty dict. Otherwise in key generation
    h0 is erased by h1
    """
    assert len(seed.shape) == 2
    n_values = seed.shape[1]
    assert seed.shape[0] == λs

    if n_values not in H_cache:
        H_cache[n_values] = [th.empty(λs, n_values, dtype=th.long, device="cuda") for _ in range(4)]

    buffers = H_cache[n_values]
    for i in range(4):
        key = keys[i]
        csprng.encrypt(seed, buffers[i], key, "aes128", "ecb")

    if (n_values, idx) not in valuebits_cache:
        valuebits_cache[(n_values, idx)] = th.empty(2, 5, n_values, dtype=th.long, device="cuda")

    valuebits = valuebits_cache[(n_values, idx)]

    valuebits[0, 0], last_bit = split_last_bit(buffers[0].native___getitem__(0))
    # valuebits[0, 1] = buffers[0][1]
    valuebits[0, 1] = last_bit
    valuebits[0, 2], last_bit = split_last_bit(buffers[1].native___getitem__(0))
    valuebits[0, 3] = buffers[1].native___getitem__(1)
    valuebits[0, 4] = last_bit
    valuebits[1, 0], last_bit = split_last_bit(buffers[2].native___getitem__(0))
    # valuebits[1, 1] = buffers[2][1]
    valuebits[1, 1] = last_bit
    valuebits[1, 2], last_bit = split_last_bit(buffers[3].native___getitem__(0))
    valuebits[1, 3] = buffers[3].native___getitem__(1)
    valuebits[1, 4] = last_bit

    # seed = seed  # .cuda()
    # urandom_gen = csprng.create_const_generator(key)
    # mask = th.empty(4 * λs, n_values, dtype=th.long, device="cuda").random_(generator=urandom_gen)
    # repl_seed = seed.repeat(4, 1)
    # # print('mask, repl_seed', mask.shape, repl_seed.shape)
    # buffer = mask + repl_seed
    # valuebits = th.empty(2, 6, n_values, dtype=th.long, device="cuda")
    #
    # # [λ, 1, λ, 1, λ, 1, λ, 1]
    # # [λ - 64, 64, 1, λ - 64, 64, 1, λ - 64, 64, 1, λ - 64, 64, 1]
    #
    # valuebits[0, 0], last_bit = split_last_bit(buffer[0])
    # valuebits[0, 1] = buffer[1]
    # valuebits[0, 2] = last_bit
    # valuebits[0, 3], last_bit = split_last_bit(buffer[2])
    # valuebits[0, 4] = buffer[3]
    # valuebits[0, 5] = last_bit
    # valuebits[1, 0], last_bit = split_last_bit(buffer[4])
    # valuebits[1, 1] = buffer[5]
    # valuebits[1, 2] = last_bit
    # valuebits[1, 3], last_bit = split_last_bit(buffer[6])
    # valuebits[1, 4] = buffer[7]
    # valuebits[1, 5] = last_bit
    return valuebits
