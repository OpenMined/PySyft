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
        res_shares = []
        for a, b in zip(x.child, y.child):
            if hasattr(a, "__mul__"):
                res_shares.append(a.__mul__(a, b, shape_x, shape_y, **kwargs))
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
