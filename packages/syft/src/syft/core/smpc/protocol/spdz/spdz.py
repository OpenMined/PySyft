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
from typing import Optional
from typing import TYPE_CHECKING

# syft absolute
import syft as sy

# relative
from ....common.uid import UID
from ....node.abstract.node import AbstractNode
from ....node.common.client import Client
from ....store.storeable_object import StorableObject
from ...store import CryptoPrimitiveProvider

# from sympc.utils import parallel_execution

EXPECTED_OPS = {"mul", "matmul"}
cache_clients: Dict[Client, Client] = {}

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
    # from ....tensor.smpc.mpc_tensor import MPCTensor

    parties = x.parties
    parties_info = x.parties_info

    shape_x = tuple(x.shape)  # type: ignore
    shape_y = tuple(y.shape)  # type: ignore

    primitives = CryptoPrimitiveProvider.generate_primitives(
        f"beaver_{op_str}",
        parties=parties,
        g_kwargs={
            "a_shape": shape_x,
            "b_shape": shape_y,
            "parties_info": parties_info,
        },
        p_kwargs={"a_shape": shape_x, "b_shape": shape_y},
    )

    # TODO: Should modify to parallel execution.

    res_shares = [
        getattr(a, "__mul__")(a, b, **kwargs) for a, b in zip(x.child, y.child)
    ]

    return res_shares  # type: ignore


def gt_master(x: MPCTensor, y: MPCTensor, op_str: str) -> MPCTensor:
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
    # from ....tensor.smpc.mpc_tensor import MPCTensor

    if op_str not in EXPECTED_OPS:
        raise ValueError(f"{op_str} should be in {EXPECTED_OPS}")

    parties = x.parties
    parties_info = x.parties_info

    shape_x = tuple(x.shape)  # type: ignore
    shape_y = tuple(y.shape)  # type: ignore

    CryptoPrimitiveProvider.generate_primitives(
        f"beaver_{op_str}",
        parties=parties,
        g_kwargs={
            "a_shape": shape_x,
            "b_shape": shape_y,
            "parties_info": parties_info,
        },
        p_kwargs={"a_shape": shape_x, "b_shape": shape_y},
    )

    # TODO: get nr of bits in another way
    for i in range(32):
        # There are needed 32 values for each bit
        CryptoPrimitiveProvider.generate_primitives(
            f"beaver_{op_str}",
            parties=parties,
            g_kwargs={
                "a_shape": shape_x,
                "b_shape": shape_y,
                "parties_info": parties_info,
            },
            p_kwargs={"a_shape": shape_x, "b_shape": shape_y},
        )

    # TODO: Should modify to parallel execution.
    kwargs = {"seed_id_locations": secrets.randbits(64)}
    res_shares = [a.__gt__(b, **kwargs) for a, b in zip(x.child, y.child)]

    return res_shares  # type: ignore


def beaver_populate(
    data: Any, id_at_location: UID, node: Optional[AbstractNode] = None
) -> None:
    """Populate the given input Tensor in the location specified.

    Args:
        data (Tensor): input Tensor to store in the node.
        id_at_location (UID): the location to store the data in.
        node Optional[AbstractNode] : The node on which the data is stored.
    """
    obj = node.store.get_object(key=id_at_location)  # type: ignore
    if obj is None:
        list_data = sy.lib.python.List([data])
        result = StorableObject(
            id=id_at_location,
            data=list_data,
            read_permissions={},
        )
        node.store[id_at_location] = result  # type: ignore
    elif isinstance(obj.data, sy.lib.python.List):
        obj = obj.data
        obj.append(data)
        result = StorableObject(
            id=id_at_location,
            data=obj,
            read_permissions={},
        )
        node.store[id_at_location] = result  # type: ignore
    else:
        raise Exception(f"Object at {id_at_location} should be a List or None")
