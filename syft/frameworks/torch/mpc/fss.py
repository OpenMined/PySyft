"""
This is an implementation of Function Secret Sharing

Useful papers are:
- Function Secret Sharing- Improvements and Extensions, Boyle 2017
  Link: https://eprint.iacr.org/2018/707.pdf
- Secure Computation with Preprocessing via Function Secret Sharing, Boyle 2019
  Link: https://eprint.iacr.org/2019/1095

Note that the protocols are quite different in aspect from those papers
"""
import hashlib

import torch as th
import syft as sy


λ = 110  # 6  # 110 or 63  # security parameter
n = 32  # 8  # 32  # bit precision
dtype = th.int32

no_wrap = {"no_wrap": True}


def initialize_crypto_plans(worker):
    """
    This is called manually for the moment, to build the plan used to perform
    Function Secret Sharing on a specific worker.
    """
    eq_plan_1 = sy.Plan(
        forward_func=lambda x, y: mask_builder(x, y, "eq"),
        owner=worker,
        tags=["#fss_eq_plan_1"],
        is_built=True,
    )
    worker.register_obj(eq_plan_1)
    eq_plan_2 = sy.Plan(
        forward_func=eq_eval_plan, owner=worker, tags=["#fss_eq_plan_2"], is_built=True
    )
    worker.register_obj(eq_plan_2)

    comp_plan_1 = sy.Plan(
        forward_func=lambda x, y: mask_builder(x, y, "comp"),
        owner=worker,
        tags=["#fss_comp_plan_1"],
        is_built=True,
    )
    worker.register_obj(comp_plan_1)
    comp_plan_2 = sy.Plan(
        forward_func=comp_eval_plan, owner=worker, tags=["#fss_comp_plan_2"], is_built=True
    )
    worker.register_obj(comp_plan_2)

    xor_add_plan = sy.Plan(
        forward_func=xor_add_convert_1, owner=worker, tags=["#xor_add_1"], is_built=True
    )
    worker.register_obj(xor_add_plan)
    xor_add_plan = sy.Plan(
        forward_func=xor_add_convert_2, owner=worker, tags=["#xor_add_2"], is_built=True
    )
    worker.register_obj(xor_add_plan)


def request_run_plan(worker, plan_tag, location, return_value, args=(), kwargs={}):
    response_ids = (sy.ID_PROVIDER.pop(),)
    args = (args, response_ids)

    response = worker.send_command(
        cmd_name="run",
        target=plan_tag,
        recipient=location,
        return_ids=response_ids,
        return_value=return_value,
        kwargs_=kwargs,
        args_=args,
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
        args = (x1.child[location.id], x2.child[location.id])
        share = request_run_plan(
            me, f"#fss_{type_op}_plan_1", location, return_value=True, args=args
        )
        shares.append(share)

    mask_value = sum(shares) % 2 ** n

    shares = []
    for i, location in enumerate(locations):
        args = (th.IntTensor([i]), mask_value)
        share = request_run_plan(
            me, f"#fss_{type_op}_plan_2", location, return_value=False, args=args
        )
        shares.append(share)

    if type_op == "comp":
        prev_shares = shares
        shares = []
        for prev_share, location in zip(prev_shares, locations):
            share = request_run_plan(
                me, "#xor_add_1", location, return_value=True, args=(prev_share,)
            )
            shares.append(share)

        masked_value = shares[0] ^ shares[1]  # TODO case >2 workers ?

        shares = {}
        for i, prev_share, location in zip(range(len(locations)), prev_shares, locations):
            share = request_run_plan(
                me,
                "#xor_add_2",
                location,
                return_value=False,
                args=(th.IntTensor([i]), masked_value),
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
    alpha, s_0, *CW = x1.owner.crypto_store.get_keys(
        f"fss_{type_op}", n_instances=x1.numel(), remove=False
    )
    return x + alpha.reshape(x.shape)


# share level
def eq_eval_plan(b, x_masked):
    alpha, s_0, *CW = x_masked.owner.crypto_store.get_keys(
        type_op="fss_eq", n_instances=x_masked.numel(), remove=True
    )
    result_share = DPF.eval(b, x_masked, s_0, *CW)
    return result_share


# share level
def comp_eval_plan(b, x_masked):
    alpha, s_0, *CW = x_masked.owner.crypto_store.get_keys(
        type_op="fss_comp", n_instances=x_masked.numel(), remove=True
    )
    result_share = DIF.eval(b, x_masked, s_0, *CW)
    return result_share


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
        beta = th.tensor([1], dtype=dtype)
        alpha = th.randint(0, 2 ** n, (n_values,))

        α = bit_decomposition(alpha)
        s, t, CW = (
            Array(n + 1, 2, λ, n_values),
            Array(n + 1, 2, n_values),
            Array(n, 2 * (λ + 1), n_values),
        )
        s[0] = randbit(size=(2, λ, n_values))
        t[0] = th.tensor([[0, 1]] * n_values, dtype=th.uint8).t()
        for i in range(0, n):
            g0 = G(s[i, 0])
            g1 = G(s[i, 1])
            # Re-use useless randomness
            sL_0, _, sR_0, _ = split(g0, [λ, 1, λ, 1])
            sL_1, _, sR_1, _ = split(g1, [λ, 1, λ, 1])
            s_rand = (sL_0 ^ sL_1) * α[i] + (sR_0 ^ sR_1) * (1 - α[i])

            cw_i = TruthTableDPF(s_rand, α[i])
            CW[i] = cw_i ^ g0 ^ g1

            for b in (0, 1):
                τ = [g0, g1][b] ^ (t[i, b] * CW[i])
                τ = τ.reshape(2, λ + 1, n_values)
                # filtered_τ = τ[𝛼[i]] OLD
                α_i = α[i].unsqueeze(0).expand(λ + 1, n_values).unsqueeze(0).long()
                filtered_τ = th.gather(τ, 0, α_i).squeeze(0)
                s[i + 1, b], t[i + 1, b] = split(filtered_τ, [λ, 1])

        CW_n = (-1) ** t[n, 1].to(dtype) * (beta - Convert(s[n, 0]) + Convert(s[n, 1]))

        return (alpha,) + s[0].unbind() + (CW, CW_n)

    @staticmethod
    def eval(b, x, *k_b):
        original_shape = x.shape
        x = x.reshape(-1)
        n_values = x.shape[0]
        x = bit_decomposition(x)
        s, t = Array(n + 1, λ, n_values), Array(n + 1, 1, n_values)
        s[0] = k_b[0]
        # here k[1:] is (CW, CW_n)
        CW = k_b[1].unbind() + (k_b[2],)
        t[0] = b
        for i in range(0, n):
            τ = G(s[i]) ^ (t[i] * CW[i])
            τ = τ.reshape(2, λ + 1, n_values)
            x_i = x[i].unsqueeze(0).expand(λ + 1, n_values).unsqueeze(0).long()
            filtered_τ = th.gather(τ, 0, x_i).squeeze(0)
            s[i + 1], t[i + 1] = split(filtered_τ, [λ, 1])
        flat_result = (-1) ** b * (Convert(s[n]) + t[n].squeeze() * CW[n])
        return flat_result.reshape(original_shape)


class DIF:
    """Distributed Interval Function - used for comparison <="""

    def __init__(self):
        pass

    @staticmethod
    def keygen(n_values=1):
        alpha = th.randint(0, 2 ** n, (n_values,))
        α = bit_decomposition(alpha)
        s, t, CW = (
            Array(n + 1, 2, λ, n_values),
            Array(n + 1, 2, n_values),
            Array(n, 2 + 2 * (λ + 1), n_values),
        )
        s[0] = randbit(size=(2, λ, n_values))
        t[0] = th.tensor([[0, 1]] * n_values, dtype=th.uint8).t()
        for i in range(0, n):
            h0 = H(s[i, 0])
            h1 = H(s[i, 1])
            # Re-use useless randomness
            _, _, sL_0, _, sR_0, _ = split(h0, [1, 1, λ, 1, λ, 1])
            _, _, sL_1, _, sR_1, _ = split(h1, [1, 1, λ, 1, λ, 1])
            s_rand = (sL_0 ^ sL_1) * α[i] + (sR_0 ^ sR_1) * (1 - α[i])
            cw_i = TruthTableDIF(s_rand, α[i])
            CW[i] = cw_i ^ h0 ^ h1

            for b in (0, 1):
                τ = [h0, h1][b] ^ (t[i, b] * CW[i])
                τ = τ.reshape(2, λ + 2, n_values)
                # filtered_τ = τ[𝛼[i]] OLD
                α_i = α[i].unsqueeze(0).expand(λ + 2, n_values).unsqueeze(0).long()
                filtered_τ = th.gather(τ, 0, α_i).squeeze(0)
                σ_leaf, s[i + 1, b], t[i + 1, b] = split(filtered_τ, [1, λ, 1])

        return (alpha,) + s[0].unbind() + (CW,)

    @staticmethod
    def eval(b, x, *k_b):
        original_shape = x.shape
        x = x.reshape(-1)
        n_values = x.shape[0]
        x = bit_decomposition(x)
        FnOutput = Array(n + 1, n_values)
        s, t = Array(n + 1, λ, n_values), Array(n + 1, 1, n_values)
        s[0] = k_b[0]
        CW = k_b[1].unbind()
        t[0] = b
        for i in range(0, n):
            τ = H(s[i]) ^ (t[i] * CW[i])
            τ = τ.reshape(2, λ + 2, n_values)
            x_i = x[i].unsqueeze(0).expand(λ + 2, n_values).unsqueeze(0).long()
            filtered_τ = th.gather(τ, 0, x_i).squeeze(0)
            σ_leaf, s[i + 1], t[i + 1] = split(filtered_τ, [1, λ, 1])
            FnOutput[i] = σ_leaf

        # Last tour, the other σ is also a leaf:
        FnOutput[n] = t[n]
        flat_result = FnOutput.sum(axis=0) % 2
        return flat_result.reshape(original_shape)


# PRG
def G(seed):
    assert seed.shape[0] == λ
    seed_t = seed.t().tolist()
    gen_list = []
    for seed_bit in seed_t:
        enc_str = str(seed_bit).encode()
        h = hashlib.sha3_256(enc_str)
        r = h.digest()
        binary_str = bin(int.from_bytes(r, byteorder="big"))[2 : 2 + (2 * (λ + 1))]
        gen_list.append(list(map(int, binary_str)))

    return th.tensor(gen_list, dtype=th.uint8).t()


def H(seed):
    assert seed.shape[0] == λ
    seed_t = seed.t().tolist()
    gen_list = []
    for seed_bit in seed_t:
        enc_str = str(seed_bit).encode()
        h = hashlib.sha3_256(enc_str)
        r = h.digest()
        binary_str = bin(int.from_bytes(r, byteorder="big"))[2 : 2 + 2 + (2 * (λ + 1))]
        gen_list.append(list(map(int, binary_str)))

    return th.tensor(gen_list, dtype=th.uint8).t()


def Convert(bits):
    bit_pow_lambda = th.flip(2 ** th.arange(λ), (0,)).unsqueeze(-1).to(th.long)
    return (bits.to(th.long) * bit_pow_lambda).sum(dim=0).to(dtype)


def Array(*shape):
    return th.empty(shape, dtype=th.uint8)


bit_pow_n = th.flip(2 ** th.arange(n), (0,))


def bit_decomposition(x):
    x = x.unsqueeze(-1)
    z = bit_pow_n & x
    z = z.t()
    return (z > 0).to(th.uint8)


def randbit(size):
    return th.randint(2, size=size)


def concat(*args, **kwargs):
    return th.cat(args, **kwargs)


def split(x, idx):
    return th.split(x, idx)


def TruthTableDPF(s, α_i):
    one = th.ones((1, s.shape[1])).to(th.uint8)
    s_one = concat(s, one)
    Table = th.zeros((2, λ + 1, len(α_i)), dtype=th.uint8)
    for j, el in enumerate(α_i):
        Table[el.item(), :, j] = s_one[:, j]
    return Table.reshape(-1, Table.shape[2])


def TruthTableDIF(s, α_i):
    leafTable = th.zeros((2, 1, len(α_i)), dtype=th.uint8)
    # TODO optimize: just put alpha on first line
    leaf_value = α_i
    for j, el in enumerate(α_i):
        leafTable[(1 - el).item(), 0, j] = leaf_value[j]

    one = th.ones((1, s.shape[1])).to(th.uint8)
    s_one = concat(s, one)
    nextTable = th.zeros((2, λ + 1, len(α_i)), dtype=th.uint8)
    for j, el in enumerate(α_i):
        nextTable[el.item(), :, j] = s_one[:, j]

    Table = concat(leafTable, nextTable, axis=1)
    Table = Table.reshape(-1, Table.shape[2])
    return Table
