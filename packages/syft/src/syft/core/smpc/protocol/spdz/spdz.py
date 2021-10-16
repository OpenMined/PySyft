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
    print("Primtives: ")
    print(primitives)
    print("***************************")

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



# def mul_parties(
#     x: ShareTensor,
#     y: ShareTensor,
#     crypto_store: CryptoStore,
#     op_str: str,
#     eps_id: UID,
#     delta_id: UID,
#     node: Optional[AbstractNode] = None,
# ) -> ShareTensor:
#     """SPDZ Multiplication.

#     Args:
#         x (ShareTensor): UUID to identify the session on each party side.
#         y (ShareTensor): Epsilon value of the protocol.
#         crypto_store (CryptoStore): CryptoStore at each parties side.
#         op_str (str): Operator string.
#         eps_id (UID): UID to store public epsilon value.
#         delta_id (UID): UID to store public delta value.
#         clients (List[Client]): Clients of parties involved in the computation.
#         node (Optional[AbstractNode]): The  node which the input ShareTensor belongs to.

#     Returns:
#         ShareTensor: Shared result of the division.
#     """
#     clients = x.clients
#     shape_x = tuple(x.child.shape)
#     shape_y = tuple(y.child.shape)

#     primitives = crypto_store.get_primitives_from_store(
#         f"beaver_{op_str}", shape_x, shape_y  # type: ignore
#     )

#     a_share, b_share, c_share = primitives
#     print("primitives", primitives)

#     eps = x - a_share
#     delta = y - b_share
#     print("Eps: ##################", eps)
#     print("Delta: ##################", delta)
#     for client in clients:
#         if client.id != node.id:  # type: ignore
#             print("///////////////////////Exec loop")
#             print(client)
#             client.syft.core.smpc.protocol.spdz.spdz.beaver_populate(eps, eps_id)  # type: ignore
#             client.syft.core.smpc.protocol.spdz.spdz.beaver_populate(delta, delta_id)  # type: ignore

#     ctr = 3000
#     while True:
#         obj = node.store.get_object(key=eps_id)  # type: ignore
#         print("Obj^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^:", obj)
#         if obj is not None:
#             obj = obj.data
#             if not isinstance(obj, sy.lib.python.List):
#                 raise Exception(
#                     f"Epsilon value at {eps_id},{type(obj)} should be a List"
#                 )
#             if len(obj) == len(clients) - 1:
#                 eps = eps + sum(obj)
#                 break
#         time.sleep(0.1)
#         ctr -= 1
#         if ctr == 0:
#             raise Exception("All Epsilon values did not arrive at store")

#     ctr = 3000
#     while True:
#         obj = node.store.get_object(key=delta_id)  # type: ignore
#         if obj is not None:
#             obj = obj.data
#             if not isinstance(obj, sy.lib.python.List):
#                 raise Exception(
#                     f"Epsilon value at {delta_id},{type(obj)} should be a List"
#                 )
#             if len(obj) == len(clients) - 1:
#                 delta = delta + sum(obj)
#                 break
#         time.sleep(0.1)
#         ctr -= 1
#         if ctr == 0:
#             raise Exception("All Delta values did not arrive at store")

#     op = getattr(operator, op_str)

#     eps_b = op(eps, b_share.child)
#     delta_a = op(a_share.child, delta)

#     tensor = c_share.child + eps_b + delta_a
#     if x.rank == 0:
#         eps_delta = op(eps, delta)
#         tensor += eps_delta

#     share = x.copy_tensor()
#     share.child = tensor  # As we do not use fixed point we neglect truncation.

#     return share


def beaver_populate(
    data: Any, id_at_location: UID, node: Optional[AbstractNode] = None
) -> None:
    """Populate the given input Tensor in the location specified.

    Args:
        data (Tensor): input Tensor to store in the node.
        id_at_location (UID): the location to store the data in.
        node Optional[AbstractNode] : The node on which the data is stored.
    """
    print("Location", id_at_location)
    print("Beaver")
    print("Data--------------------------------------", data)
    obj = node.store.get_object(key=id_at_location)  # type: ignore
    print("Object ", obj)
    if obj is None:
        list_data = sy.lib.python.List([data])
        result = StorableObject(
            id=id_at_location,
            data=list_data,
            read_permissions={},
        )
        node.store[id_at_location] = result  # type: ignore
    elif isinstance(obj.data, sy.lib.python.List):
        print("Entered")
        obj = obj.data
        print("First", obj)
        obj.append(data)
        print("Second", obj)
        result = StorableObject(
            id=id_at_location,
            data=obj,
            read_permissions={},
        )
        print("Result", result)
        node.store[id_at_location] = result  # type: ignore
    else:
        raise Exception(f"Object at {id_at_location} should be a List or None")
    print("Node value ::::::::::::", node.store.get_object(key=id_at_location))  # type: ignore
    print("Beaver Finish*************************************************")
