"""
This is an implementation of Function Secret Sharing

Useful papers are:
- Function Secret Sharing- Improvements and Extensions, Boyle 2017 https://eprint.iacr.org/2018/707.pdf
- Secure Computation with Preprocessing via Function Secret Sharing, Boyle 2019 https://eprint.iacr.org/2019/1095

Note that the protocols are quite different in aspect from those papers
"""
import hashlib
import math
import time
import numpy as np
import sha_loop
import multiprocessing
import asyncio

import torch as th
import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
from syft.generic.utils import allow_command
from syft.generic.utils import remote


λ = 127  # 6  # 110 or 63  # security parameter
n = 32  # 8  # 32  # bit precision
λs = math.ceil(λ / 64)  # how many dtype values are needed to store λ, typically 2
assert λs == 2

no_wrap = {"no_wrap": True}


def full_name(f):
    return f"syft.frameworks.torch.mpc.fss.{f.__name__}"


# internal codes
EQ = 0
COMP = 1

# number of processes
N_CORES = 8
MULTI_LIMIT = 10_000


def fss_op(x1, x2, type_op="eq"):
    """
    Define the workflow for a binary operation using Function Secret Sharing

    Currently supported operand are = & <=, respectively corresponding to
    type_op = 'eq' and 'comp'

    Args:
        x1: first AST
        x2: second AST
        type_op: type of operation to perform, should be 'eq' or 'comp'

    Returns:
        shares of the comparison
    """
    # simple vs multi depending on num elements
    # {
    #     5_000: [0.085, 0.10]
    #     10_000: [0.15, 0.15],
    #     20_000: [0.28, 0.23]
    #     50_000: [0.75, 0.47],
    #     100_000: [1.50, 0.95]
    # }

    if isinstance(x1, sy.AdditiveSharingTensor):
        locations = x1.locations
        class_attributes = x1.get_class_attributes()
    else:
        locations = x2.locations
        class_attributes = x2.get_class_attributes()

    asynchronous = isinstance(locations[0], WebsocketClientWorker)

    workers_args = [
        (
            x1.child[location.id]
            if isinstance(x1, sy.AdditiveSharingTensor)
            else (x1 if i == 0 else 0),
            x2.child[location.id]
            if isinstance(x2, sy.AdditiveSharingTensor)
            else (x2 if i == 0 else 0),
            type_op,
        )
        for i, location in enumerate(locations)
    ]

    shares = []
    for i, location in enumerate(locations):
        share = remote(mask_builder, location=location)(*workers_args[i], return_value=True)
        shares.append(share)

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

    workers_args = [(th.IntTensor([i]), mask_value, type_op) for i in range(2)]
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
def mask_builder(x1, x2, type_op):
    if not isinstance(x1, int):
        worker = x1.owner
        numel = x1.numel()
    else:
        worker = x2.owner
        numel = x2.numel()
    x = x1 - x2
    # Keep the primitive in store as we use it after
    # you actually get a share of alpha
    alpha, s_0, *CW = worker.crypto_store.get_keys(
        f"fss_{type_op}", n_instances=numel, remove=False
    )
    r = x + th.tensor(alpha.astype(np.int64)).reshape(x.shape)
    return r


@allow_command
def evaluate(b, x_masked, type_op):
    MULTI_LIMIT = 10_000
    if type_op == "eq":
        return eq_evaluate(b, x_masked)
    elif type_op == "comp":
        numel = x_masked.numel()
        if numel > MULTI_LIMIT:
            # print('MULTI EVAL', numel, x_masked.owner)
            owner = x_masked.owner
            multiprocessing_args = []
            original_shape = x_masked.shape
            x_masked = x_masked.reshape(-1)
            slice_size = math.ceil(numel / N_CORES)
            for j in range(N_CORES):
                x_masked_slice = x_masked[j * slice_size : (j + 1) * slice_size]
                x_masked_slice.owner = owner
                process_args = (b, x_masked_slice, owner.id, j, j * slice_size)
                multiprocessing_args.append(process_args)
            p = multiprocessing.Pool()
            partitions = p.starmap(comp_evaluate, multiprocessing_args)
            p.close()
            partitions = sorted(partitions, key=lambda k: k[0])
            partitions = [partition[1] for partition in partitions]
            result = th.cat(partitions)

            # Burn the primitives (copies of the workers were sent)
            owner.crypto_store.get_keys(f"fss_{type_op}", n_instances=numel, remove=True)

            return result.reshape(*original_shape)
        else:
            # print('EVAL', numel)
            return comp_evaluate(b, x_masked)
    else:
        raise ValueError


# share level
def eq_evaluate(b, x_masked):
    alpha, s_0, *CW = x_masked.owner.crypto_store.get_keys(
        type_op="fss_eq", n_instances=x_masked.numel(), remove=True
    )
    result_share = DPF.eval(b.numpy().item(), x_masked.numpy(), s_0, *CW)
    return th.tensor(result_share)


# share level
def comp_evaluate(b, x_masked, owner_id=None, core_id=None, burn_offset=0):
    if owner_id is not None:
        x_masked.owner = x_masked.owner.get_worker(owner_id)

    if burn_offset > 0:
        _ = x_masked.owner.crypto_store.get_keys(
            type_op="fss_comp", n_instances=burn_offset, remove=True
        )
    alpha, s_0, *CW = x_masked.owner.crypto_store.get_keys(
        type_op="fss_comp", n_instances=x_masked.numel(), remove=True
    )
    result_share = DIF.eval(b.numpy().item(), x_masked.numpy(), s_0, *CW)
    if core_id is None:
        return th.tensor(result_share)
    else:
        return core_id, th.tensor(result_share)


def eq(x1, x2):
    return fss_op(x1, x2, "eq")


def le(x1, x2):
    return fss_op(x1, x2, "comp")


class DPF:
    """Distributed Point Function - used for equality"""

    @staticmethod
    def keygen(n_values=1):
        alpha = np.random.randint(0, 2 ** n, size=(n_values,), dtype=np.uint64)
        beta = np.array([1])
        α = bit_decomposition(alpha)
        s, t, CW = (
            Array(n + 1, 2, λs, n_values),
            Array(n + 1, 2, n_values),
            Array(n, 2, (λs + 1), n_values),
        )
        s[0] = randbit(shape=(2, λ, n_values))
        t[0] = np.array([[0, 1]] * n_values).T
        for i in range(0, n):
            g0 = G(s[i, 0])
            g1 = G(s[i, 1])
            # Re-use useless randomness
            sL_0, _, sR_0, _ = split(g0, (EQ, λs, 1, λs, 1))
            sL_1, _, sR_1, _ = split(g1, (EQ, λs, 1, λs, 1))
            s_rand = (sL_0 ^ sL_1) * α[i] + (sR_0 ^ sR_1) * (1 - α[i])

            cw_i = SwitchTableDPF(s_rand, α[i])
            CW[i] = cw_i ^ g0 ^ g1

            for b in (0, 1):
                τ = [g0, g1][b] ^ (t[i, b] * CW[i])
                filtered_τ = multi_dim_filter(τ, α[i])
                s[i + 1, b], t[i + 1, b] = split(filtered_τ, (EQ, λs, 1))

        CW_n = (-1) ** t[n, 1] * (beta - convert(s[n, 0]) + convert(s[n, 1]))
        CW_n = CW_n.astype(np.int64)
        return (alpha, s[0][0], s[0][1], *CW, CW_n)

    @staticmethod
    def eval(b, x, *k_b):
        x = x.astype(np.uint64)
        original_shape = x.shape
        x = x.reshape(-1)
        n_values = x.shape[0]
        x = bit_decomposition(x)
        s, t = Array(n + 1, λs, n_values), Array(n + 1, 1, n_values)
        s[0], *CW = k_b
        t[0] = b
        for i in range(0, n):
            τ = G(s[i]) ^ (t[i] * CW[i])
            filtered_τ = multi_dim_filter(τ, x[i])
            s[i + 1], t[i + 1] = split(filtered_τ, (EQ, λs, 1))

        flat_result = (-1) ** b * (t[n].squeeze() * CW[n] + convert(s[n]))
        return flat_result.astype(np.int64).reshape(original_shape)


class DIF:
    """Distributed Point Function - used for equality"""

    @staticmethod
    def keygen(n_values=1):
        alpha = np.random.randint(0, 2 ** n, size=(n_values,), dtype=np.uint64)
        α = bit_decomposition(alpha)
        s, σ, t, τ, CW, CW_leaf = (
            Array(n + 1, 2, λs, n_values),
            Array(n + 1, 2, λs, n_values),
            Array(n + 1, 2, n_values),
            Array(n + 1, 2, n_values),
            Array(n, 2, 2 * (λs + 1), n_values),
            Array(n + 1, n_values),
        )
        s[0] = randbit(shape=(2, λ, n_values))
        t[0] = np.array([[0, 1]] * n_values).T

        for i in range(0, n):
            h0 = H(s[i, 0], 0)
            h1 = H(s[i, 1], 1)
            # Re-use useless randomness
            σL_0, _, sL_0, _, σR_0, _, sR_0, _ = split(h0, (COMP, λs, 1, λs, 1, λs, 1, λs, 1))
            σL_1, _, sL_1, _, σR_1, _, sR_1, _ = split(h1, (COMP, λs, 1, λs, 1, λs, 1, λs, 1))
            s_rand = (sL_0 ^ sL_1) * α[i] + (sR_0 ^ sR_1) * (1 - α[i])
            σ_rand = (σL_0 ^ σL_1) * α[i] + (σR_0 ^ σR_1) * (1 - α[i])
            cw_i = SwitchTableDIF(s_rand, σ_rand, α[i])
            CW[i] = cw_i ^ h0 ^ h1

            for b in (0, 1):
                dual_state = [h0, h1][b] ^ (t[i, b] * CW[i])
                # the state obtained by following the special path
                state = multi_dim_filter(dual_state, α[i])
                _, _, s[i + 1, b], t[i + 1, b] = split(state, (COMP, λs, 1, λs, 1))
                # the state obtained by leaving the special path
                anti_state = multi_dim_filter(dual_state, 1 - α[i])
                σ[i + 1, b], τ[i + 1, b], _, _ = split(anti_state, (COMP, λs, 1, λs, 1))

                if b:
                    # note that we subtract (1 - α[i]), so that leaving the special path can't lead
                    # to an output == 1 when α[i] == 0 (because it means that your bit is 1 so your
                    # value is > α)
                    CW_leaf[i] = (-1) ** τ[i + 1, 1] * (
                        1 - convert(σ[i + 1, 0]) + convert(σ[i + 1, 1]) - (1 - α[i])
                    )

        CW_leaf[n] = (-1) ** t[n, 1] * (1 - convert(s[n, 0]) + convert(s[n, 1]))

        CW_leaf = CW_leaf.astype(np.int64)

        return (alpha, s[0][0], s[0][1], *CW, CW_leaf)

    @staticmethod
    def eval(b, x, *k_b):
        x = x.astype(np.uint64)
        original_shape = x.shape
        x = x.reshape(-1)
        n_values = x.shape[0]
        x = bit_decomposition(x)
        s, σ, t, τ, out = (
            Array(n + 1, λs, n_values),
            Array(n + 1, λs, n_values),
            Array(n + 1, 1, n_values),
            Array(n + 1, 1, n_values),
            Array(n + 1, n_values),
        )
        s[0], *CW, CW_leaf = k_b
        t[0] = b

        for i in range(0, n):
            dual_state = H(s[i]) ^ (t[i] * CW[i])
            state = multi_dim_filter(dual_state, x[i])
            σ[i + 1], τ[i + 1], s[i + 1], t[i + 1] = split(state, (COMP, λs, 1, λs, 1))
            out[i] = (-1) ** b * (τ[i + 1] * CW_leaf[i] + convert(σ[i + 1]))

        # Last node, the other σ is also a leaf
        out[n] = (-1) ** b * (t[n].squeeze() * CW_leaf[n] + convert(s[n]))

        return out.sum(axis=0).astype(np.int64).reshape(original_shape)


def Array(*shape):
    return np.empty(shape, dtype=np.uint64)


def bit_decomposition(x):
    x = x.astype(np.uint32)
    n_values = x.shape[0]
    x = x.reshape(-1, 1).view(np.uint8)
    x = x.reshape(n_values, 4, 1)
    x = x >> np.arange(8, dtype=np.uint8)
    x = x & 0b1
    x = np.flip(x.reshape(n_values, -1)[:, :n], axis=1).T
    return x


def randbit(shape):
    assert len(shape) == 3
    byte_dim = shape[-2]
    shape_with_bytes = shape[:-2] + (math.ceil(byte_dim / 64), shape[-1])
    randvalues = np.random.randint(0, 2 ** 64, size=shape_with_bytes, dtype=np.uint64)
    randvalues[:, 0] = randvalues[:, 0] % 2 ** (byte_dim % 64)
    return randvalues


def concat(*args, **kwargs):
    return np.concatenate(args, **kwargs)


def split_last_bit(buffer):
    # Numbers are on 64 bits
    return buffer & 0b1111111111111111111111111111111111111111111111111111111111111110, buffer & 0b1


def huge_loop(seed_t_bytes, n_iter):
    return [hashlib.sha3_256(seed_t_bytes[i * 16 : (i + 1) * 16]).digest() for i in range(n_iter)]


def G(seed):
    """ λ -> 2(λ + 1)"""

    assert len(seed.shape) == 2
    n_values = seed.shape[1]
    assert seed.shape[0] == λs
    x = seed
    x = x.T
    dt1 = np.dtype((np.uint64, [("uint8", np.uint8, 8)]))
    x2 = x.view(dtype=dt1)
    x = x2["uint8"].reshape(*x.shape[:-1], -1)

    assert x.shape == (n_values, 2 * 8)

    out = np.empty((n_values, 4 * 8), dtype=np.uint8)

    out = sha_loop.sha256_loop_func(x, out)

    buffer = out.view(np.uint64).T

    valuebits = np.empty((2, 3, n_values), dtype=np.uint64)

    # [λ, 1, λ, 1]
    # [λ - 64, 64, 1, λ - 64, 64, 1]
    valuebits[0, 0], last_bit = split_last_bit(buffer[0])
    valuebits[0, 1] = buffer[1]
    valuebits[0, 2] = last_bit
    valuebits[1, 0], last_bit = split_last_bit(buffer[2])
    valuebits[1, 1] = buffer[3]
    valuebits[1, 2] = last_bit

    return valuebits


empty_dict = {}


def H(seed, idx=0):
    """ λ -> 4(λ + 1)

    idx is here to allow not reusing the same empty dict. Otherwise in key generation
    h0 is erased by h1
    """

    assert len(seed.shape) == 2
    n_values = seed.shape[1]
    assert seed.shape[0] == λs
    x = seed
    x = x.T
    dt1 = np.dtype((np.uint64, [("uint8", np.uint8, 8)]))
    x2 = x.view(dtype=dt1)
    x = x2["uint8"].reshape(*x.shape[:-1], -1)

    assert x.shape == (n_values, 2 * 8)

    if (n_values, idx) not in empty_dict:
        # 64 bytes are needed to store a sha512
        empty_dict[(n_values, idx)] = (
            np.empty((n_values, 64), dtype=np.uint8),
            np.empty((2, 6, n_values), dtype=np.uint64),
        )

    out, valuebits = empty_dict[(n_values, idx)]

    out = sha_loop.sha512_loop_func(x, out)

    buffer = out.view(np.uint64).T  # is of size 8 * 64 bits

    # [λ, 1, λ, 1, λ, 1, λ, 1]
    # [λ - 64, 64, 1, λ - 64, 64, 1, λ - 64, 64, 1, λ - 64, 64, 1]

    # valuebits[0] = buffer[0] & b63_, buffer[1], buffer[0] & b_1, buffer[2] & b63_, buffer[3], buffer[2] & b_1
    # valuebits[1] = buffer[4] & b63_, buffer[5], buffer[4] & b_1, buffer[6] & b63_, buffer[7], buffer[6] & b_1
    valuebits[0, 0], last_bit = split_last_bit(buffer[0])
    valuebits[0, 1] = buffer[1]
    valuebits[0, 2] = last_bit
    valuebits[0, 3], last_bit = split_last_bit(buffer[2])
    valuebits[0, 4] = buffer[3]
    valuebits[0, 5] = last_bit
    valuebits[1, 0], last_bit = split_last_bit(buffer[4])
    valuebits[1, 1] = buffer[5]
    valuebits[1, 2] = last_bit
    valuebits[1, 3], last_bit = split_last_bit(buffer[6])
    valuebits[1, 4] = buffer[7]
    valuebits[1, 5] = last_bit

    return valuebits


split_helpers = {
    (EQ, 2, 1): lambda x: (x[:2], x[2]),
    (EQ, 2, 1, 2, 1): lambda x: (x[0, :2], x[0, 2], x[1, :2], x[1, 2]),
    (COMP, 2, 1, 2, 1): lambda x: (x[:2], x[2], x[3:5], x[5]),
    (COMP, 2, 1, 2, 1, 2, 1, 2, 1): lambda x: (
        x[0, :2],
        x[0, 2],
        x[0, 3:5],
        x[0, 5],
        x[1, :2],
        x[1, 3],
        x[1, 3:5],
        x[1, 5],
    ),
}


def split(list_, idx):
    return split_helpers[idx](list_)


ones_dict2 = {}


def SwitchTableDPF(s, α_i):
    one = np.ones((1, s.shape[1]), dtype=np.uint64)
    s_one = concat(s, one)

    if s_one.shape not in ones_dict2:
        ones_dict2[s_one.shape] = np.ones((1, *s_one.shape), dtype=np.uint64)
    ones = ones_dict2[s_one.shape]
    pad = (α_i * ones).astype(np.uint64)
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
    # Select the 16th least significant bits
    r = x[-1] & 0b1111111111111111
    return r.astype(np.int64)
