"""SPDZ Protocol.

SPDZ mechanism used for multiplication Contains functions that are run at:

* the party that orchestrates the computation
* the parties that hold the shares
"""

# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import TYPE_CHECKING

# relative
from .....ast.klass import get_run_class_method
from ....tensor.smpc import utils
from ...store import CryptoPrimitiveProvider

EXPECTED_OPS = {"mul", "matmul"}

if TYPE_CHECKING:
    # relative
    from ....tensor.smpc.mpc_tensor import MPCTensor


def mul_master(
    x: MPCTensor, y: MPCTensor, op_str: str, **kwargs: Dict[Any, Any]
) -> MPCTensor:

    """Function that is executed by the orchestrator to multiply two secret values.

    Args:
        x (MPCTensor): First value to multiply with.
        y (MPCTensor): Second value to multiply with.
        op_str (str): Operation string.

    Raises:
        ValueError: If op_str not in EXPECTED_OPS.

    Returns:
        MPCTensor: Result of the multiplication.
    """

    # relative
    from ....tensor.tensor import TensorPointer

    parties = x.parties
    parties_info = x.parties_info

    shape_x = tuple(x.shape)  # type: ignore
    shape_y = tuple(y.shape)  # type: ignore

    ring_size = utils.get_ring_size(x.ring_size, y.ring_size)

    if ring_size != 2:
        # For ring_size 2 we generate those before hand
        CryptoPrimitiveProvider.generate_primitives(
            f"beaver_{op_str}",
            parties=parties,
            g_kwargs={
                "a_shape": shape_x,
                "b_shape": shape_y,
                "parties_info": parties_info,
            },
            p_kwargs={"a_shape": shape_x, "b_shape": shape_y},
            ring_size=ring_size,
        )

    # TODO: Should modify to parallel execution.
    if not isinstance(x.child[0], TensorPointer):
        res_shares = [
            getattr(a, "__mul__")(a, b, shape_x, shape_y, **kwargs)
            for a, b in zip(x.child, y.child)
        ]
    else:
        res_shares = []
        attr_path_and_name = f"{x.child[0].path_and_name}.__{op_str}__"
        op = get_run_class_method(attr_path_and_name, SMPC=True)
        for a, b in zip(x.child, y.child):
            res_shares.append(op(a, a, b, shape_x, shape_y, **kwargs))

    return res_shares  # type: ignore


def lt_master(x: MPCTensor, y: MPCTensor, op_str: str) -> MPCTensor:
    """Function that is executed by the orchestrator to multiply two secret values.

    Args:
        x (MPCTensor): First value to multiply with.
        y (MPCTensor): Second value to multiply with.
        op_str (str): Operation string.

    Raises:
        ValueError: If op_str not in EXPECTED_OPS.

    Returns:
        MPCTensor: Result of the multiplication.
    """
    # relative
    from ....tensor.smpc.mpc_tensor import MPCTensor
    from ....tensor.tensor import TensorPointer

    # diff = a - b
    # bit decomposition
    # sum carry adder
    # res = sign(diff)

    res_shares = x - y
    res_shares.block
    # time.sleep(2)
    msb = MSB(res_shares)
    tensor_shares = []
    final_shares = []

    if isinstance(x, MPCTensor):
        if isinstance(x.child[0], TensorPointer):
            for t1, t2 in zip(x.child, y.child):
                tensor_shares.append(t1.__lt__(t2))

            for p1, p2 in zip(tensor_shares, msb.child):
                p2.block
                final_shares.append(p1.mpc_swap(p2))

            msb.child = final_shares
    else:
        if isinstance(y.child[0], TensorPointer):  # type: ignore
            for t1 in y.child:
                tensor_shares.append(t1.__lt__(x))

            for p1, p2 in zip(tensor_shares, msb.child):
                p2.block
                final_shares.append(p1.mpc_swap(p2))

            msb.child = final_shares

    return msb


def MSB(x: MPCTensor) -> MPCTensor:
    # relative
    from ..aby3 import ABY3

    """Computes the MSB of the underlying share

    Args:
        x (MPCTensor): Input MPCTensor to compute MSB on.

    Returns:
        msb (MPCTensor): returns arithmetic shares of the MSB.
    """
    ring_size = 2**32  # TODO : Should extract ring_size elsewhere for generality.
    decomposed_shares = ABY3.bit_decomposition(x)

    for share in decomposed_shares:
        share.block

    msb_share = decomposed_shares[-1]
    msb = ABY3.bit_injection(msb_share, ring_size)

    return msb


def public_divide(
    x: MPCTensor, y: Union[torch.Tensor, int], **kwargs: Dict[Any, Any]
) -> MPCTensor:
    """Function that is executed by the DS (orchestrator) to divide an MPC
    by a public value.

    Args:
        x (MPCTensor): Private numerator.
        y (Union[torch.Tensor, int]): Public denominator.

    Returns:
        MPCTensor: A new set of shares that represents the division.
    """
    res_shape = x.shape
    nr_parties = len(x.parties_info)

    if nr_parties == 2:
        res_shares = []
        for share in x.child:
            method = getattr(share, "__truediv__")
            share = method(y, **kwargs)
            res_shares.append(share)
        return res_shares

    # TODO: Needs refactoring to work with only one action
    primitives = CryptoPrimitiveProvider.generate_primitives(
        f"beaver_wraps",
        parties=parties,
        g_kwargs={
            "nr_parties": shape_y,
            "shape": res_shape,
        },
        p_kwargs=None,
        ring_size=ring_size,
    )

    r_sh, theta_r_sh = list(zip(*list(zip(*primitives))[0]))

    r_mpc = MPCTensor(shares=r_sh, session=session, shape=x.shape)

    z = r_mpc + x
    z_shares_local = z.get_shares()

    common_args = [z_shares_local, y]
    args = zip(
        r_mpc.share_ptrs,
        theta_r_sh,
        x.share_ptrs,
    )
    args = [list(el) + common_args for el in args]

    theta_x = parallel_execution(div_wraps, session.parties)(args)
    theta_x_plaintext = MPCTensor(shares=theta_x, session=session).reconstruct()

    res = x - theta_x_plaintext * 4 * ((session.ring_size // 4) // y)

    return res.child


def div_wraps(
    r_share: ShareTensor,
    theta_r: ShareTensor,
    x_share: ShareTensor,
    z_shares: List[torch.Tensor],
    y: Union[torch.Tensor, int],
) -> ShareTensor:
    """From CrypTen Privately computes the number of wraparounds for a set a shares.

    To do so, we note that:
        [theta_x] = theta_z + [beta_xr] - [theta_r] - [eta_xr]

    Where:
        [theta_x] is the wraps for a variable x
        [beta_xr] is the differential wraps for variables x and r
        [eta_xr]  is the plaintext wraps for variables x and r

    Note: Since [eta_xr] = 0 with probability 1 - |x| / Q for modulus Q, we
    can make the assumption that [eta_xr] = 0 with high probability.

    Args:
        r_share (ShareTensor): share for a random variable "r"
        theta_r (ShareTensor): share for the number of wraparounds for "r"
        x_share (ShareTensor): shares for which we want to compute the number of wraparounds
        z_shares (List[torch.Tensor]): list of shares for a random value
        y (Union[torch.Tensor, int]): the number/tensor by which we divide

    Returns:
        ShareTensor representing the number of wraparounds
    """
    session = get_session(r_share.session_uuid)

    beta_xr = utils.count_wraps([x_share.tensor, r_share.tensor])
    theta_x = ShareTensor(config=Config(encoder_precision=0))
    theta_x.tensor = beta_xr - theta_r.tensor

    if session.rank == 0:
        theta_z = count_wraps(z_shares)
        theta_x.tensor += theta_z

    x_share.tensor //= y

    return theta_x
