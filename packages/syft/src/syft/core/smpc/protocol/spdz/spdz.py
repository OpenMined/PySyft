"""SPDZ Protocol.

SPDZ mechanism used for multiplication Contains functions that are run at:

* the party that orchestrates the computation
* the parties that hold the shares
"""

# stdlib
import operator
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# syft absolute
# absolute
import syft as sy

# relative
from ..... import Tensor
from ....common.uid import UID
from ....node.common.client import Client
from ....node.common.node import Node
from ....store.storeable_object import StorableObject
from ....tensor.smpc.mpc_tensor import MPCTensor
from ....tensor.smpc.share_tensor import ShareTensor
from ...store import CryptoPrimitiveProvider
from ...store import CryptoStore

# from sympc.utils import parallel_execution

EXPECTED_OPS = {"mul", "matmul"}
cache_clients: Dict[Client, Client] = {}


def mul_master(
    x: MPCTensor, y: MPCTensor, op_str: str, kwargs_: Dict[Any, Any]
) -> MPCTensor:
    """Function that is executed by the orchestrator to multiply two secret values.

    Args:
        x (MPCTensor): First value to multiply with.
        y (MPCTensor): Second value to multiply with.
        op_str (str): Operation string.
        kwargs_ (dict): TODO:Add docstring.

    Raises:
        ValueError: If op_str not in EXPECTED_OPS.

    Returns:
        MPCTensor: Result of the multiplication.
    """
    if op_str not in EXPECTED_OPS:
        raise ValueError(f"{op_str} should be in {EXPECTED_OPS}")

    parties = x.parties
    cache_store = CryptoPrimitiveProvider.cache_store

    shape_x = tuple(x.shape)
    shape_y = tuple(y.shape)
    eps_id = UID()
    delta_id = UID()
    args = [
        [x, y, cache_store[party], op_str, eps_id, delta_id, parties]
        for x, y, party in zip(x.share_ptrs, y.share_ptrs, parties)
    ]

    CryptoPrimitiveProvider.generate_primitives(
        f"beaver_{op_str}",
        parties=parties,
        g_kwargs={
            "a_shape": shape_x,
            "b_shape": shape_y,
            **kwargs_,
        },
        p_kwargs={"a_shape": shape_x, "b_shape": shape_y},
    )

    # TODO: Should modify to parallel execution
    shares = [
        party.syft.core.smpc.protocol.spdz.spdz.mul_parties(*args)
        for arg, party in zip(args, parties)
    ]
    shape = MPCTensor.__get_shape(op_str, shape_x, shape_y)
    result = MPCTensor(shares=shares, parties=parties, shape=shape)

    return result


def mul_parties(
    x: ShareTensor,
    y: ShareTensor,
    crypto_store: CryptoStore,
    op_str: str,
    eps_id: UID,
    delta_id: UID,
    parties: List[Client],
    node: Optional[Node] = None,
    **kwargs: Dict[Any, Any],
) -> ShareTensor:
    """SPDZ Multiplication.

    Args:
        x (ShareTensor): UUID to identify the session on each party side.
        y (ShareTensor): Epsilon value of the protocol.
        crypto_store (CryptoStore): CryptoStore at each parties side.
        op_str (str): Operator string.
        eps_id (UID): UID to store public epsilon value.
        delta_id (UID): UID to store public delta value.
        parties (Any): Clients of parties involved in the computation.
        node (Any): The  node which the input ShareTensor belongs to.
        kwargs (Dict[Any,Any]): Keywords arguments for the operator.

    Returns:
        ShareTensor: Shared result of the division.
    """
    shape_x = tuple(x.child.shape)
    shape_y = tuple(y.child.shape)

    primitives = crypto_store.get_primitives_from_store(
        f"beaver_{op_str}", a_shape=shape_x, b_shape=shape_y  # type: ignore
    )

    a_share, b_share, c_share = primitives

    eps = x - a_share
    delta = y - b_share

    for party in parties:
        if party.id != node.id:  # type: ignore
            party.syft.core.smpc.protocol.spdz.spdz.beaver_populate(eps, eps_id)  # type: ignore
            party.syft.core.smpc.protocol.spdz.spdz.beaver_populate(delta, delta_id)  # type: ignore

    ctr = 3000
    while True:
        obj = node.store.get_object(key=eps_id)  # type: ignore
        if obj is not None:
            obj = obj.data
            if not isinstance(obj, sy.lib.python.List):
                raise Exception(
                    f"Epsilon value at {eps_id},{type(obj)} should be a List"
                )
            if len(obj) == len(parties) - 1:
                eps = eps + sum(obj)
                break
        time.sleep(0.1)
        ctr -= 1
        if ctr == 0:
            raise Exception("All Epsilon values did not arrive at store")

    ctr = 3000
    while True:
        obj = node.store.get_object(key=delta_id)  # type: ignore
        if obj is not None:
            obj = obj.data
            if not isinstance(obj, sy.lib.python.List):
                raise Exception(
                    f"Epsilon value at {delta_id},{type(obj)} should be a List"
                )
            if len(obj) == len(parties) - 1:
                delta = delta + sum(obj)
                break
        time.sleep(0.1)
        ctr -= 1
        if ctr == 0:
            raise Exception("All Delta values did not arrive at store")

    op = getattr(operator, op_str)

    eps_b = op(eps, b_share.child, **kwargs)
    delta_a = op(a_share.child, delta, **kwargs)

    tensor = c_share.child + eps_b + delta_a
    if x.rank == 0:
        eps_delta = op(eps, delta, **kwargs)
        tensor += eps_delta

    share = x.copy_tensor()
    share.child = tensor  # As we do not use fixed point we neglect truncation.

    return share


def beaver_populate(
    data: Tensor, id_at_location: UID, node: Optional[Node] = None
) -> None:
    """Populate the given input Tensor in the location specified.

    Args:
        data (Tensor): input Tensor to store in the node.
        id_at_location (UID): the location to store the data in.
        node : The node on which the data is stored.
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
