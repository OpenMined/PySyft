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
import sha_loop

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
authorized = set(f"{NAMESPACE}.{name}" for name in ["mask_builder", "evaluate"])

# internal codes
EQ = 0
COMP = 1


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


def eq(x1, x2):
    return fss_op(x1, x2, "eq")


def le(x1, x2):
    return fss_op(x1, x2, "comp")


class DPF:
    """Distributed Point Function - used for equality"""

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
        alpha = np.random.randint(
            0, 2 ** n, size=(n_values,), dtype=np.uint64
        )  # this is IID in int32
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
            h0 = H(s[i, 0])
            h1 = H(s[i, 1])
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
    x = np.flip(x.reshape(n_values, -1), axis=1).T
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


def consume(buffer, nbits):
    new_buffer = buffer >> nbits
    extracted = buffer - (new_buffer << nbits)
    return extracted, new_buffer


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
    buffer0, valuebits[0, 0] = consume(buffer[0], λ - 64)
    valuebits[0, 1] = buffer[1]
    valuebits[0, 2] = buffer0 & 0b1
    buffer2, valuebits[1, 0] = consume(buffer[2], λ - 64)
    valuebits[1, 1] = buffer[3]
    valuebits[1, 2] = buffer2 & 0b1

    return valuebits


empty_dict = {}


def H(seed):
    """ λ -> 2 + 2(λ + 1)"""

    assert len(seed.shape) == 2
    n_values = seed.shape[1]
    assert seed.shape[0] == λs
    x = seed
    x = x.T
    dt1 = np.dtype((np.uint64, [("uint8", np.uint8, 8)]))
    x2 = x.view(dtype=dt1)
    x = x2["uint8"].reshape(*x.shape[:-1], -1)

    assert x.shape == (n_values, 2 * 8)

    if n_values not in empty_dict:
        # 64 bytes are needed to store a sha512
        empty_dict[n_values] = np.empty((n_values, 64), dtype=np.uint8)

    out = empty_dict[n_values]

    out = sha_loop.sha512_loop_func(x, out)

    buffer = out.view(np.uint64).T  # is of size 8 * 64 bits

    # [λ, 1, λ, 1, λ, 1, λ, 1]
    # [λ - 64, 64, 1, λ - 64, 64, 1, λ - 64, 64, 1, λ - 64, 64, 1]
    valuebits = np.empty((2, 6, n_values), dtype=np.uint64)

    valuebits[0, 0], buffer0 = consume(buffer[0], λ - 64)
    valuebits[0, 1] = buffer[1]
    valuebits[0, 2] = buffer0 & 0b1
    valuebits[0, 3], buffer2 = consume(buffer[2], λ - 64)
    valuebits[0, 4] = buffer[3]
    valuebits[0, 5] = buffer2 & 0b1
    valuebits[1, 0], buffer4 = consume(buffer[4], λ - 64)
    valuebits[1, 1] = buffer[5]
    valuebits[1, 2] = buffer4 & 0b1
    valuebits[1, 3], buffer6 = consume(buffer[6], λ - 64)
    valuebits[1, 4] = buffer[7]
    valuebits[1, 5] = buffer6 & 0b1

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
    r = x[-1] % 2 ** 8
    return r.astype(np.int64)
