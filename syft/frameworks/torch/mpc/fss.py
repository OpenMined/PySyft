"""
This is an implementation of Function Secret Sharing

Useful papers are:
- Function Secret Sharing- Improvements and Extensions, Boyle 2017 https://eprint.iacr.org/2018/707.pdf
- Secure Computation with Preprocessing via Function Secret Sharing, Boyle 2019 https://eprint.iacr.org/2019/1095

Note that the protocols are quite different in aspect from those papers
"""
import hashlib

import torch as th
import syft as sy
from syft.exceptions import EmptyCryptoPrimitiveStoreError
from syft.execution.plan import func2plan
from syft.generic.frameworks.hook.trace import tracer
from syft.workers.base import BaseWorker


λ = 6  # 63  # 6  # 63  # security parameter
n = 8  # 32  # 8  # 32  # bit precision

no_wrap = {"no_wrap": True}


def manual_init_store(worker):
    """
    This is called manually for the moment, to build the plan used to perform
    Function Secret Sharing on a specific worker.
    """
    # # Init the plans for equality and comparison
    # for type_op, fss_class in zip(["eq", "comp"], [DPF, DIF]):
    #     keygen = func2plan()(fss_class.keygen)
    #     keygen.build()
    #
    #     alpha, s_00, s_01, *CW = keygen()
    #     k = [(s_00, *CW), (s_01, *CW)]
    #     evaluate = func2plan()(fss_class.eval)
    #     evaluate.build(th.IntTensor([0]), alpha, *k[0])
    #
    #     keygen.owner = worker
    #     keygen.tag(f"#fss-{type_op}-keygen")
    #     evaluate.owner = worker
    #     evaluate.tag(f"#fss-{type_op}-eval")
    #
    #     keygen.forward = None
    #     evaluate.forward = None

    # (future)
    # eq_plan_1 = sy.Plan(forward_func=mask_builder, owner=worker, tags=["#fss_eq_plan_1"], is_built=True)
    # worker.register_obj(eq_plan_1)
    # eq_plan_2 = sy.Plan(forward_func=eval_plan, owner=worker, tags=["#fss_eq_plan_2"], is_built=True)
    # worker.register_obj(eq_plan_2)

    #
    x1 = th.tensor([1, 2])
    x1.owner = worker
    x2 = th.tensor([3, 2])
    x2.owner = worker
    eq_plan_1 = func2plan()(mask_builder)
    eq_plan_1.build(x1, x2)
    eq_plan_1.owner = worker
    eq_plan_1.forward = None
    eq_plan_1.tag(f"#fss_eq_plan_1")

    b = th.IntTensor([0])
    b.owner = worker
    x_masked = th.tensor([19])
    x_masked.owner = worker
    eq_plan_2 = func2plan()(eval_plan)
    eq_plan_2.build(b, x_masked)
    eq_plan_2.owner = worker
    eq_plan_2.forward = None
    eq_plan_2.tag(f"#fss_eq_plan_2")


def request_run_plan(worker, plan_tag, location, return_value, args=tuple(), kwargs=dict()):
    response_ids = [sy.ID_PROVIDER.pop()]
    args = [args, response_ids]

    command = ("run", plan_tag, args, kwargs)

    response = worker.send_command(
        message=command, recipient=location, return_ids=response_ids, return_value=return_value
    )
    return response


def fss_op_tracer(x1, x2, type_op="eq"):

    me = sy.local_worker
    locations = x1.locations

    shares = []
    for location in locations:
        share = request_run_plan(
            me,
            "#fss_eq_plan_1",
            location,
            return_value=True,
            args=(x1.child[location.id], x2.child[location.id]),
        )
        shares.append(share)

    mask_value = sum(shares) % 2 ** n

    shares = {}
    for i, location in enumerate(locations):
        b = th.IntTensor([i])
        b.owner = location
        mask_value.owner = location
        share = request_run_plan(
            me, "#fss_eq_plan_2", location, return_value=False, args=(b, mask_value)
        )
        shares[location] = share

    response = sy.AdditiveSharingTensor(shares, **x1.get_class_attributes())

    return response


@tracer(func_name="sy.frameworks.torch.mpc.fss.get_keys")
def get_keys(worker_id, remove=True):
    worker: BaseWorker = sy.local_worker.get_worker(worker_id)
    try:
        if remove:
            return worker.crypto_store.fss_eq.pop(0)
        else:
            return worker.crypto_store.fss_eq[0]
    except IndexError:
        raise EmptyCryptoPrimitiveStoreError(worker.crypto_store, "fss_eq")


@tracer(func_name="sy.frameworks.torch.mpc.fss.evaluation")
def evaluation(b, x_masked):
    if hasattr(x_masked, "child"):
        x_masked = x_masked.child
        b = b.child
    alpha, s_0, *CW = get_keys(x_masked.owner.id, remove=True)
    share = DPF.eval(b, x_masked, s_0, *CW)
    return share


# share level
def mask_builder(x1, x2):
    x = x1 - x2
    # Keep the primitive in store as we use it after
    alpha, s_0, *CW = get_keys(x1.owner.id, remove=False)
    return x + alpha


# share level
def eval_plan(b, x_masked):
    return evaluation(b, x_masked)


def fss_op(x1, x2, type_op):
    """
    Define the workflow for a binary operation using Function Secret Sharing

    Currently supported operand are = & <=, respectively corresponding to
    type_op = 'eq' and 'comp'

    Args:
        x1: first AST
        x2: second AST
        type_op: type of operation to perform, should be 'eq' or 'comp'
    """

    # Equivalence x1==x2 ~ x1-x2==0
    x_sh = x1 - x2

    locations = x1.locations
    crypto_provider = x1.crypto_provider
    me = sy.local_worker

    # Retrieve keygen plan
    (keygen_ptr,) = me.find_or_request(f"#fss-{type_op}-keygen", location=crypto_provider)

    # Run key generation
    alpha, s_00, s_01, *CW = keygen_ptr()

    # build shares of the mask
    alpha_sh = alpha.share(*locations, crypto_provider=crypto_provider, **no_wrap).get().child

    # reveal masked values and send to locations
    x_masked = (x_sh + alpha_sh).get().send(*x_sh.locations, **no_wrap)

    s_0 = sy.MultiPointerTensor(children=[s.move(loc) for s, loc in zip([s_00, s_01], locations)])
    k = [s_0] + [c.get().send(*locations, **no_wrap) for c in CW]

    # Eval
    b = sy.MultiPointerTensor(
        children=[th.IntTensor([i]).send(loc, **no_wrap) for i, loc in enumerate(locations)]
    )

    # Search multi ptr plan Eval
    eval_tag = f"#fss-{type_op}-eval-{'-'.join([loc.id for loc in locations])}"
    if not me.find_by_tag(eval_tag):  # if not registered, build it
        (evaluate_ptr,) = me.find_or_request(f"#fss-{type_op}-eval", location=crypto_provider)
        evaluate_ptr = evaluate_ptr.get().send(*locations).tag(eval_tag)
    else:  # else retrieve it directly
        (evaluate_ptr,) = me.find_by_tag(eval_tag)

    # Run evaluation
    int_shares = evaluate_ptr(b, x_masked, *k)

    # Build response
    long_shares = {k: v.long() for k, v in int_shares.items()}
    response = sy.AdditiveSharingTensor(long_shares, **x_sh.get_class_attributes())
    # # Response is binary
    # response.field = 2

    return response


def eq(x1, x2):
    return fss_op(x1, x2, "eq")


def le(x1, x2):
    return fss_op(x1, x2, "comp")


class DPF:
    """Distributed Point Function - used for equality"""

    def __init__(self):
        pass

    @staticmethod
    def keygen():
        beta = th.tensor([1], dtype=th.int32)
        (alpha,) = th.randint(0, 2 ** n, (1,))
        print("alpha", alpha)

        α = bit_decomposition(alpha)
        s, t, CW = Array(n + 1, 2, λ), Array(n + 1, 2), Array(n, 2 * (λ + 1))
        s[0] = randbit(size=(2, λ))
        t[0] = th.tensor([0, 1], dtype=th.uint8)
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
                τ = τ.reshape(2, λ + 1)
                s[i + 1, b], t[i + 1, b] = split(τ[α[i]], [λ, 1])

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


class DIF:
    "Distributed Interval Function - used for comparison <="

    def __init__(self):
        pass

    @staticmethod
    def keygen():
        (alpha,) = th.randint(0, 2 ** n, (1,))
        α = bit_decomposition(alpha)
        s, t, CW = Array(n + 1, 2, λ), Array(n + 1, 2), Array(n, 2 + 2 * (λ + 1))
        s[0] = randbit(size=(2, λ))
        t[0] = th.tensor([0, 1], dtype=th.uint8)
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
                τ = τ.reshape(2, λ + 2)
                σ_leaf, s[i + 1, b], t[i + 1, b] = split(τ[α[i]], [1, λ, 1])

        return (alpha,) + s[0].unbind() + (CW,)

    @staticmethod
    def eval(b, x, *k_b):
        FnOutput = Array(n + 1, 1)
        x = bit_decomposition(x)
        s, t = Array(n + 1, λ), Array(n + 1, 1)
        s[0] = k_b[0]
        CW = k_b[1].unbind()
        t[0] = b
        for i in range(0, n):
            τ = H(s[i]) ^ (t[i] * CW[i])
            τ = τ.reshape(2, λ + 2)
            σ_leaf, s[i + 1], t[i + 1] = split(τ[x[i]], [1, λ, 1])
            FnOutput[i] = σ_leaf

        # Last tour, the other σ is also a leaf:
        FnOutput[n] = t[n]
        return FnOutput.sum() % 2


# PRG
def G(seed):
    return prg(seed)
    assert len(seed) == λ
    th.manual_seed(Convert(seed))
    return th.randint(2, size=(2 * (λ + 1),), dtype=th.uint8)


@tracer(func_name="sy.frameworks.torch.mpc.fss.prg")
def prg(seed):
    h = hashlib.sha3_256(str(seed).encode())
    r = h.digest()
    r = th.tensor(
        int.from_bytes(r, byteorder="big") % 2 ** (2 * (λ + 1)), dtype=th.long
    )  # r >> (256 - 2*(λ + 1))
    bit_pow_n = th.flip(2 ** th.arange(2 * (λ + 1)), (0,))
    return ((r & bit_pow_n) > 0).to(th.uint8)


def H(seed):
    assert len(seed) == λ
    th.manual_seed(Convert(seed))
    return th.randint(2, size=(2 + 2 * (λ + 1),), dtype=th.uint8)


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


def TruthTableDIF(s, α_i):
    leafTable = th.zeros((2, 1), dtype=th.uint8)
    # if α_i is 0, then ending on the leaf branch means your bit is 1 to you're > α so you should get 0
    # if α_i is 1, then ending on the leaf branch means your bit is 0 to you're < α so you should get 1
    leaf_value = α_i
    leafTable[1 - α_i] = leaf_value

    nextTable = th.zeros((2, λ + 1), dtype=th.uint8)
    one = th.tensor([1], dtype=th.uint8)
    nextTable[α_i] = concat(s, one)

    return concat(leafTable, nextTable, axis=1).flatten()
