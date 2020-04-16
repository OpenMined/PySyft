"""
This is an implementation of Function Secret Sharing

Useful papers are:
- Function Secret Sharing- Improvements and Extensions, Boyle 2017 https://eprint.iacr.org/2018/707.pdf
- Secure Computation with Preprocessing via Function Secret Sharing, Boyle 2019 https://eprint.iacr.org/2019/1095

Note that the protocols are quite different in aspect from those papers
"""
import hashlib
import math
from numba import jit
import numpy as np

import torch as th
import syft as sy
from syft.execution.plan import func2plan
from syft.generic.frameworks.hook.trace import tracer
from syft.workers.base import BaseWorker


λ = 110  # 6  # 110 or 63  # security parameter
n = 32  # 8  # 32  # bit precision
λs = math.ceil(λ / 64)  # how many dtype values are needed to store λ, typically 2
assert λs == 2

no_wrap = {"no_wrap": True}

NAMESPACE = "syft.frameworks.torch.mpc.fss"
authorized = set(
    f"{NAMESPACE}.{name}"
    for name in ["mask_builder", "evaluate", "xor_add_convert_1", "xor_add_convert_2"]
)


def full_name(f):
    return f"{NAMESPACE}.{f.__name__}"


def remote_exec(
    command_name,
    location,
    args=tuple(),
    kwargs=dict(),
    worker=None,
    return_value=False,
    return_arity=1,
):
    if worker is None:
        worker = sy.local_worker

    response_ids = [sy.ID_PROVIDER.pop() for _ in range(return_arity)]

    command = (command_name, None, args, kwargs)

    response = worker.send_command(
        message=command, recipient=location, return_ids=response_ids, return_value=return_value
    )
    return response


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

    me = sy.local_worker
    locations = x1.locations

    shares = []
    for location in locations:
        args = (x1.child[location.id], x2.child[location.id], type_op)
        share = remote_exec(full_name(mask_builder), location, args=args, return_value=True)
        shares.append(share)

    mask_value = sum(shares) % 2 ** n

    shares = []
    for i, location in enumerate(locations):
        args = (th.IntTensor([i]), mask_value, type_op)
        share = remote_exec(full_name(evaluate), location, args=args, return_value=False)
        shares.append(share)

    if type_op == "comp":
        prev_shares = shares
        shares = []
        for prev_share, location in zip(prev_shares, locations):
            args = (prev_share,)
            share = remote_exec(
                full_name(xor_add_convert_1), location, args=args, return_value=True
            )
            shares.append(share)

        masked_value = shares[0] ^ shares[1]  # TODO case >2 workers ?

        shares = {}
        for i, prev_share, location in zip(range(len(locations)), prev_shares, locations):
            args = (th.IntTensor([i]), masked_value)
            share = remote_exec(
                full_name(xor_add_convert_2), location, args=args, return_value=False
            )
            shares[location.id] = share
    else:
        shares = {loc.id: share for loc, share in zip(locations, shares)}

    response = sy.AdditiveSharingTensor(shares, **x1.get_class_attributes())
    return response


# share level
def mask_builder(x1, x2, type_op):
    x = x1 - x2
    # Keep the primitive in store as we use it after
    # you actually get a share of alpha
    alpha, s_0, *CW = x1.owner.crypto_store.get_keys(
        f"fss_{type_op}", n_instances=x1.numel(), remove=False
    )
    r = x + th.tensor(alpha.astype(np.int64)).reshape(x.shape)
    return r


def evaluate(b, x_masked, type_op):
    if type_op == "eq":
        return eq_evaluate(b, x_masked)
    elif type_op == "comp":
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
def comp_evaluate(b, x_masked):
    alpha, s_0, *CW = x_masked.owner.crypto_store.get_keys(
        type_op="fss_comp", n_instances=x_masked.numel(), remove=True
    )
    result_share = DIF.eval(b.numpy().item(), x_masked.numpy(), s_0, *CW)
    return th.tensor(result_share)


def xor_add_convert_1(x):
    xor_share, add_share = x.owner.crypto_store.get_keys(
        type_op="xor_add_couple", n_instances=x.numel(), remove=False
    )
    return x ^ xor_share.reshape(x.shape)


def xor_add_convert_2(b, x):
    xor_share, add_share = x.owner.crypto_store.get_keys(
        type_op="xor_add_couple", n_instances=x.numel(), remove=True
    )
    return add_share.reshape(x.shape) * (1 - 2 * x) + x * b


def eq(x1, x2):
    return fss_op(x1, x2, "eq")


def le(x1, x2):
    return fss_op(x1, x2, "comp")


class DPF:
    """Distributed Point Function - used for equality"""

    def __init__(self):
        pass

    @staticmethod
    def keygen(n_values=1):
        alpha = np.random.randint(
            0, 2 ** n, size=(n_values,), dtype=np.uint64
        )  # this is IID in int32
        beta = np.array([1])
        α = bit_decomposition(alpha)
        s, t, CW = (
            Array(n + 1, 2, λs, n_values),
            Array(n + 1, 2, n_values),
            Array(n, 2 * (λs + 1), n_values),
        )
        s[0] = randbit(shape=(2, λ, n_values))
        t[0] = np.array([[0, 1]] * n_values).T
        for i in range(0, n):
            g0 = G(s[i, 0])
            g1 = G(s[i, 1])
            # Re-use useless randomness
            sL_0, _, sR_0, _ = split(g0, (λs, 1, λs, 1))
            sL_1, _, sR_1, _ = split(g1, (λs, 1, λs, 1))
            s_rand = (sL_0 ^ sL_1) * α[i] + (sR_0 ^ sR_1) * (1 - α[i])

            cw_i = SwitchTableDPF(s_rand, α[i])
            CW[i] = cw_i ^ g0 ^ g1

            for b in (0, 1):
                τ = [g0, g1][b] ^ (t[i, b] * CW[i])
                τ = τ.reshape(2, λs + 1, n_values)
                filtered_τ = multi_dim_filter(τ, α[i])
                s[i + 1, b], t[i + 1, b] = split(filtered_τ, (λs, 1))

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
            τ = τ.reshape(2, λs + 1, n_values)
            filtered_τ = multi_dim_filter(τ, x[i])
            s[i + 1], t[i + 1] = split(filtered_τ, (λs, 1))

        flat_result = (-1) ** b * (t[n].squeeze() * CW[n] + convert(s[n]))
        return flat_result.astype(np.int64).reshape(original_shape)


class DIF:
    """Distributed Point Function - used for equality"""

    def __init__(self):
        pass

    @staticmethod
    def keygen(n_values=1):
        alpha = np.random.randint(
            0, 2 ** n, size=(n_values,), dtype=np.uint64
        )  # this is IID in int32
        α = bit_decomposition(alpha)
        s, t, CW = (
            Array(n + 1, 2, λs, n_values),
            Array(n + 1, 2, n_values),
            Array(n, 2 + 2 * (λs + 1), n_values),
        )
        s[0] = randbit(shape=(2, λ, n_values))
        t[0] = np.array([[0, 1]] * n_values).T
        for i in range(0, n):
            h0 = H(s[i, 0])
            h1 = H(s[i, 1])
            # Re-use useless randomness
            _, _, sL_0, _, sR_0, _ = split(h0, (1, λs, 1, 1, λs, 1))
            _, _, sL_1, _, sR_1, _ = split(h1, (1, λs, 1, 1, λs, 1))
            s_rand = (sL_0 ^ sL_1) * α[i] + (sR_0 ^ sR_1) * (1 - α[i])
            cw_i = SwitchTableDIF(s_rand, α[i])
            CW[i] = cw_i ^ h0 ^ h1

            for b in (0, 1):
                τ = [h0, h1][b] ^ (t[i, b] * CW[i])
                τ = τ.reshape(2, λs + 2, n_values)
                # filtered_τ = τ[𝛼[i]] OLD
                filtered_τ = multi_dim_filter(τ, α[i])
                σ_leaf, s[i + 1, b], t[i + 1, b] = split(filtered_τ, (1, λs, 1))

        return (alpha, s[0][0], s[0][1], *CW)

    @staticmethod
    def eval(b, x, *k_b):
        x = x.astype(np.uint64)
        original_shape = x.shape
        x = x.reshape(-1)
        n_values = x.shape[0]
        x = bit_decomposition(x)
        FnOutput = Array(n + 1, n_values)
        s, t = Array(n + 1, λs, n_values), Array(n + 1, 1, n_values)
        s[0], *CW = k_b
        t[0] = b
        for i in range(0, n):
            τ = H(s[i]) ^ (t[i] * CW[i])
            τ = τ.reshape(2, λs + 2, n_values)
            filtered_τ = multi_dim_filter(τ, x[i])
            σ_leaf, s[i + 1], t[i + 1] = split(filtered_τ, (1, λs, 1))
            FnOutput[i] = σ_leaf

        # Last tour, the other σ is also a leaf:
        FnOutput[n] = t[n]
        # print(FnOutput)
        flat_result = FnOutput.sum(axis=0) % 2
        return flat_result.astype(np.int64).reshape(original_shape)


def Array(*shape):
    return np.empty(shape, dtype=np.uint64)


def bit_decomposition(x, nbits=n):
    return np.flip((x.reshape(-1, 1) >> np.arange(nbits, dtype=np.uint64)) % 2, axis=1).T


def randbit(shape):
    byte_dim = shape[-2]
    shape_with_bytes = shape[:-2] + (math.ceil(byte_dim / 64), shape[-1])
    randvalues = np.random.randint(0, 2 ** 64, size=shape_with_bytes, dtype=np.uint64)
    randvalues[0] = randvalues[0] % 2 ** (byte_dim % 64)
    return randvalues


def concat(*args, **kwargs):
    return np.concatenate(args, **kwargs)


def consume(buffer, nbits):
    new_buffer = buffer >> nbits
    extracted = buffer - (new_buffer << nbits)
    return new_buffer, extracted


def huge_loop(seed_t_bytes, n_iter):
    return [hashlib.sha3_256(seed_t_bytes[i * 16 : (i + 1) * 16]).digest() for i in range(n_iter)]


def G(seed):
    """ λ -> 2(λ + 1)"""
    assert seed.shape[0] == λs
    seed_t = seed.T
    seed_t_bytes = seed_t.tobytes()

    buffers = huge_loop(seed_t_bytes, seed_t.shape[0])

    buffer = b"".join(buffers)

    buffer = np.frombuffer(buffer, dtype=np.uint64).reshape(-1, 4)

    # [λ, 1, λ, 1]
    # [λ - 64, 64, 1, λ - 64, 64, 1]
    buffer0, part0 = consume(buffer[:, 0], λ - 64)
    part1 = buffer[:, 1]
    part2 = buffer0 % 2
    buffer2, part3 = consume(buffer[:, 2], λ - 64)
    part4 = buffer[:, 3]
    part5 = buffer2 % 2

    valuebits = np.stack([part0, part1, part2, part3, part4, part5], axis=1)
    return valuebits.T


def H(seed):
    """ λ -> 2 + 2(λ + 1)"""
    assert seed.shape[0] == λs
    seed_t = seed.T
    seed_t_bytes = seed_t.tobytes()

    buffers = huge_loop(seed_t_bytes, seed_t.shape[0])

    buffer = b"".join(buffers)

    buffer = np.frombuffer(buffer, dtype=np.uint64).reshape(-1, 4)

    # [1, λ, 1, 1, λ, 1]
    # [1, λ - 64, 64, 1, 1, λ - 64, 64, 1]
    buffer0, part1 = consume(buffer[:, 0], λ - 64)
    part2 = buffer[:, 1]
    doublebit = buffer0 % 4
    part0 = doublebit // 2
    part3 = doublebit % 2
    buffer2, part4 = consume(buffer[:, 2], λ - 64)
    part5 = buffer[:, 3]
    doublebit = buffer2 % 4
    part6 = doublebit // 2
    part7 = doublebit % 2

    valuebits = np.stack([part0, part1, part2, part3, part4, part5, part6, part7], axis=1)
    return valuebits.T


split_helpers = {
    (2, 1): lambda x: (x[:2], x[2]),
    (2, 1, 2, 1): lambda x: (x[:2], x[2], x[3:5], x[5]),
    (1, 2, 1): lambda x: (x[0], x[1:3], x[3]),
    (1, 2, 1, 1, 2, 1): lambda x: (x[0], x[1:3], x[3], x[4], x[5:7], x[7]),
}


def split(list_, idx):
    return split_helpers[idx](list_)


ones_dict2 = {}


def SwitchTableDPF(s, α_i, reshape=True):
    one = np.ones((1, s.shape[1]), dtype=np.uint64)
    s_one = concat(s, one)

    if s_one.shape not in ones_dict2:
        ones_dict2[s_one.shape] = np.ones((1, *s_one.shape), dtype=np.uint64)
    ones = ones_dict2[s_one.shape]
    pad = (α_i * ones).astype(np.uint64)
    pad = concat(1 - pad, pad, axis=0)
    Table = pad * s_one

    if reshape:
        return Table.reshape(-1, Table.shape[2])
    else:
        return Table


ones_dict3 = {}


def SwitchTableDIF(s, α_i):
    # if α_i is 0, then ending on the leaf branch means your bit is 1 to you're > α so you should get 0
    # if α_i is 1, then ending on the leaf branch means your bit is 0 to you're < α so you should get 1
    # so we're doing leafTable[1-α_i] = α_i
    # example [1 1 0]
    # returns
    # [[[1 1 0]]
    #
    #  [[0 0 0]]]
    zeros = np.zeros((1, 1, len(α_i)), dtype=np.uint64)
    leafTable = concat(α_i.reshape(1, 1, -1), zeros, axis=0)

    nextTable = SwitchTableDPF(s, α_i, reshape=False)

    Table = concat(leafTable, nextTable, axis=1)
    return Table.reshape(-1, Table.shape[2])


ones_dict = {}


def multi_dim_filter(τ, idx):
    if τ.shape[1:] not in ones_dict:
        ones_dict[τ.shape[1:]] = np.ones(τ.shape[1:], dtype=np.uint64)
    ones = ones_dict[τ.shape[1:]]
    pad = idx * ones
    pad = pad.reshape(1, *pad.shape)
    filtered = concat(1 - pad, pad, axis=0) * τ
    filtered_τ = filtered.sum(axis=0)
    return filtered_τ


def convert(x):
    """
    convert a multi dim big tensor to a "random" single tensor
    """
    r = x[-1] % 2 ** 50
    return r
