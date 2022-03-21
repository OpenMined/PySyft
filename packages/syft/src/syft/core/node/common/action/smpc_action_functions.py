# stdlib
from copy import deepcopy
import functools
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from uuid import UUID

# third party
import numpy as np

# relative
from ....common.uid import UID
from ....store.storeable_object import StorableObject
from ....tensor.smpc import utils
from ....tensor.smpc.share_tensor import ShareTensor
from ...abstract.node import AbstractNode
from .beaver_action import BeaverAction
from .greenlets_switch import beaver_retrieve_object
from .smpc_action_message import SMPCActionMessage
from .smpc_action_seq_batch_message import SMPCActionSeqBatchMessage


def get_action_generator_from_op(
    operation_str: str, nr_parties: int
) -> Callable[[UID, UID, int, Any], Any]:
    """ "
    Get the generator for the operation provided by the argument
    Arguments:
        operation_str (str): the name of the operation

    """
    return functools.partial(MAP_FUNC_TO_ACTION[operation_str], nr_parties)


def get_id_at_location_from_op(seed: bytes, operation_str: str) -> UID:
    generator = np.random.default_rng(seed)
    return UID(UUID(bytes=generator.bytes(16)))


# TODO: node.address is not really ,should be removed later.


def smpc_basic_op(
    op_str: str,
    nr_parties: int,
    self_id: UID,
    other_id: UID,
    seed_id_locations: int,
    node: AbstractNode,
) -> List[SMPCActionMessage]:
    # relative
    from ..... import Tensor

    """Generator for SMPC public/private operations add/sub"""

    generator = np.random.default_rng(seed_id_locations)
    result_id = UID(UUID(bytes=generator.bytes(16)))
    other = node.store.get(other_id).data

    actions = []
    if isinstance(other, (ShareTensor, Tensor)):
        # All parties should add the other share if empty list
        actions.append(
            SMPCActionMessage(
                f"mpc_{op_str}",
                self_id=self_id,
                args_id=[other_id],
                kwargs_id={},
                ranks_to_run_action=list(range(nr_parties)),
                result_id=result_id,
                address=node.address,
            )
        )
    else:
        actions.append(
            SMPCActionMessage(
                "mpc_noop",
                self_id=self_id,
                args_id=[],
                kwargs_id={},
                ranks_to_run_action=list(range(1, nr_parties)),
                result_id=result_id,
                address=node.address,
            )
        )

        # Only rank 0 (the first party) would do the add/sub for the public value
        actions.append(
            SMPCActionMessage(
                f"mpc_{op_str}",
                self_id=self_id,
                args_id=[other_id],
                kwargs_id={},
                ranks_to_run_action=[0],
                result_id=result_id,
                address=node.address,
            )
        )

    return actions


# TODO : Should move to spdz directly in syft/core/smpc
def spdz_multiply(
    x: ShareTensor,
    y: ShareTensor,
    eps_id: UID,
    delta_id: UID,
    a_share: ShareTensor,
    b_share: ShareTensor,
    c_share: ShareTensor,
    node: Optional[Any] = None,
) -> ShareTensor:
    # relative
    from ..... import Tensor

    TENSOR_FLAG = False
    if isinstance(x, Tensor) and isinstance(y, Tensor):
        TENSOR_FLAG = True
        t1 = x
        t2 = y
        x = x.child.child
        y = y.child.child

    nr_parties = x.nr_parties

    ring_size = utils.get_ring_size(x.ring_size, y.ring_size)

    eps = beaver_retrieve_object(node, eps_id, nr_parties)  # type: ignore
    delta = beaver_retrieve_object(node, delta_id, nr_parties)  # type: ignore

    # TODO : Should refactor fixed precision tensor later

    eps = sum(eps.data)  # type: ignore
    delta = sum(delta.data)  # type:ignore

    eps_b = eps * b_share.child
    delta_a = delta * a_share.child  # Symmetric only for mul

    tensor = c_share + eps_b + delta_a
    if x.rank == 0:
        mul_op = ShareTensor.get_op(ring_size, "mul")
        eps_delta = mul_op(eps.child, delta.child)  # type: ignore
        tensor = tensor + eps_delta

    share = x.copy_tensor()
    share.child = tensor.child  # As we do not use fixed point we neglect truncation.

    if TENSOR_FLAG:
        t = t1 * t2
        t.child.child = share
        return t
    else:
        return share


# TODO : Should move to spdz directly in syft/core/smpc
def spdz_mask(
    x: ShareTensor,
    y: ShareTensor,
    eps_id: UID,
    delta_id: UID,
    a_share: ShareTensor,
    b_share: ShareTensor,
    c_share: ShareTensor,
    node: Optional[AbstractNode] = None,
) -> None:
    # relative
    from ..... import Tensor

    if isinstance(x, Tensor) and isinstance(y, Tensor):
        x = x.child.child
        y = y.child.child

    if node is None:
        raise ValueError("Node context should be passed to spdz mask")

    clients = ShareTensor.login_clients(x.parties_info)

    eps = x - a_share  # beaver intermediate values.
    delta = y - b_share

    client_id_map = {client.id: client for client in clients}
    curr_client = client_id_map[node.id]  # type: ignore
    beaver_action = BeaverAction(
        eps=eps,
        eps_id=eps_id,
        delta=delta,
        delta_id=delta_id,
        address=curr_client.address,
    )
    beaver_action.execute_action(node, None)

    for _, client in enumerate(clients):
        if client != curr_client:
            beaver_action.address = client.address
            client.send_immediate_msg_without_reply(msg=beaver_action)


def smpc_mul(
    nr_parties: int,
    self_id: UID,
    other_id: UID,
    a_shape_id: Optional[UID] = None,
    b_shape_id: Optional[UID] = None,
    seed_id_locations: Optional[int] = None,
    node: Optional[AbstractNode] = None,
) -> SMPCActionSeqBatchMessage:
    """Generator for the smpc_mul with a public value"""
    # relative
    from ..... import Tensor

    if seed_id_locations is None or node is None:
        raise ValueError(
            f"The values seed_id_locations{seed_id_locations}, Node:{node} should not be None"
        )
    generator = np.random.default_rng(seed_id_locations)
    result_id = UID(UUID(bytes=generator.bytes(16)))
    other = node.store.get(other_id).data

    actions = []
    if isinstance(other, (ShareTensor, Tensor)) and (a_shape_id and b_shape_id):
        # crypto_store = ShareTensor.crypto_store
        # _self = node.store.get(self_id).data
        # a_share, b_share, c_share = crypto_store.get_primitives_from_store("beaver_mul", _self.shape, other.shape)
        if isinstance(other, ShareTensor):
            ring_size = other.ring_size
        else:
            ring_size = other.child.child.ring_size

        mask_result = UID(UUID(bytes=generator.bytes(16)))
        eps_id = UID(UUID(bytes=generator.bytes(16)))
        delta_id = UID(UUID(bytes=generator.bytes(16)))
        a_shape = node.store.get(a_shape_id).data  # type: ignore
        b_shape = node.store.get(b_shape_id).data  # type: ignore
        crypto_store = ShareTensor.crypto_store
        a_share, b_share, c_share = crypto_store.get_primitives_from_store(
            "beaver_mul", a_shape=a_shape, b_shape=b_shape, ring_size=ring_size, remove=True  # type: ignore
        )

        actions.append(
            SMPCActionMessage(
                "spdz_mask",
                self_id=self_id,
                args_id=[other_id],
                kwargs_id={},
                kwargs={
                    "eps_id": eps_id,
                    "delta_id": delta_id,
                    "a_share": a_share,
                    "b_share": b_share,
                    "c_share": c_share,
                },
                ranks_to_run_action=list(range(nr_parties)),
                result_id=mask_result,
                address=node.address,
            )
        )

        actions.append(
            SMPCActionMessage(
                "spdz_multiply",
                self_id=self_id,
                args_id=[other_id],
                kwargs_id={},
                kwargs={
                    "eps_id": eps_id,
                    "delta_id": delta_id,
                    "a_share": a_share,
                    "b_share": b_share,
                    "c_share": c_share,
                },
                ranks_to_run_action=list(range(nr_parties)),
                result_id=result_id,
                address=node.address,
            )
        )

    else:
        # All ranks should multiply by that public value
        actions.append(
            SMPCActionMessage(
                "mpc_mul",
                self_id=self_id,
                args_id=[other_id],
                kwargs_id={},
                ranks_to_run_action=list(range(nr_parties)),
                result_id=result_id,
                address=node.address,
            )
        )
    batch = SMPCActionSeqBatchMessage(smpc_actions=actions, address=node.address)
    return batch


def local_decomposition(
    x: ShareTensor,
    ring_size: int,
    bitwise: bool,
    seed_id_locations: str,
    node: Any,
    read_permissions: Dict[Any, Any],
) -> None:
    """Performs local decomposition to generate shares of shares.

    Args:
        x (ShareTensor) : input ShareTensor.
        ring_size (str) : Ring size to generate decomposed shares in.
        bitwise (bool): Perform bit level decomposition on bits if set.

    Returns:
        List[List[ShareTensor]]: Decomposed shares in the given ring size.
    """
    # Currently we cannot serialize integers exceeding signed positive integer
    # TODO: can be modified to google.protobuf.any
    seed_id_locations = int(seed_id_locations)  # type: ignore
    generator = np.random.default_rng(seed_id_locations)
    # Skip the first ID ,as it is used for None return type in run class method.
    _ = UID(UUID(bytes=generator.bytes(16)))
    # relative
    from ..... import Tensor

    TENSOR_FLAG = False
    if isinstance(x, Tensor):
        TENSOR_FLAG = True
        t = x
        x = x.child.child

    rank = x.rank
    nr_parties = x.nr_parties
    numpy_type = utils.RING_SIZE_TO_TYPE[ring_size]
    shape = x.shape
    zero = np.zeros(shape, numpy_type)

    input_shares = []

    if bitwise:
        ring_bits = utils.get_nr_bits(x.ring_size)  # for bit-wise decomposition
        input_shares = [x.bit_extraction(idx) for idx in range(ring_bits)]
    else:
        input_shares.append(x)

    for share in input_shares:

        for i in range(nr_parties):
            id_at_location = UID(UUID(bytes=generator.bytes(16)))
            sh = x.copy_tensor()
            sh.ring_size = ring_size
            if rank != i:
                sh.child = deepcopy(zero)
            else:
                sh.child = deepcopy(share.child.astype(numpy_type))
            if TENSOR_FLAG:
                t_sh = t.copy()
                t_sh.child.child = sh
                data = t_sh
            else:
                data = sh
            store_obj = StorableObject(
                id=id_at_location,
                data=data,
                read_permissions=read_permissions,
            )
            node.store[id_at_location] = store_obj

    return None


def bit_decomposition(
    nr_parties: int,
    self_id: UID,
    ring_size: UID,
    bitwise: UID,
    seed_id_locations: int,
    node: Any,
) -> List[SMPCActionMessage]:
    generator = np.random.default_rng(seed_id_locations)
    result_id = UID(UUID(bytes=generator.bytes(16)))

    actions = []

    actions.append(
        SMPCActionMessage(
            "local_decomposition",
            self_id=self_id,
            args_id=[ring_size, bitwise],
            kwargs_id={},
            kwargs={"seed_id_locations": str(seed_id_locations)},
            ranks_to_run_action=list(range(nr_parties)),
            result_id=result_id,
            address=node.address,
        )
    )

    return actions


# Given an SMPC Action map it to an action constructor
MAP_FUNC_TO_ACTION: Dict[
    str, Callable[[int, UID, UID, int, Any], List[SMPCActionMessage]]
] = {
    "__add__": functools.partial(smpc_basic_op, "add"),
    "__sub__": functools.partial(smpc_basic_op, "sub"),
    "__mul__": smpc_mul,  # type: ignore
    "bit_decomposition": bit_decomposition,  # type: ignore
    # "__gt__": smpc_gt,  # type: ignore TODO: this should be added back when we have only one action
}


# Map given an action map it to a function that should be run on the shares"
_MAP_ACTION_TO_FUNCTION: Dict[str, Callable[..., Any]] = {
    "mpc_add": operator.add,
    "mpc_sub": operator.sub,
    "mpc_mul": operator.mul,
    "spdz_mask": spdz_mask,
    "spdz_multiply": spdz_multiply,
    "local_decomposition": local_decomposition,
    "mpc_noop": deepcopy,
}


#
# def smpc_gt(
#     nr_parties: int,
#     self_id: UID,
#     other_id: UID,
#     seed_id_locations: int,
#     node: Any,
#     client: Any,
# ) -> List[SMPCActionMessage]:
#     """Generator for the smpc_mul with a public value"""
#     generator = np.random.default_rng(seed_id_locations)

#     result_id = UID(UUID(bytes=generator.bytes(16)))
#     sub_result = UID(UUID(bytes=generator.bytes(16)))

#     x = node.store.get(self_id).data  # noqa
#     y = node.store.get(other_id).data

#     if not isinstance(y, ShareTensor):
#         raise ValueError("Only private compare works at the moment")

#     actions = []
#     actions.append(
#         SMPCActionMessage(
#             "mpc_sub",
#             self_id=self_id,
#             args_id=[other_id],
#             kwargs_id={},
#             ranks_to_run_action=list(range(nr_parties)),
#             result_id=sub_result,
#             address=client.address,
#         )
#     )

#     actions.append(
#         SMPCActionMessage(
#             "bit_decomposition",
#             self_id=sub_result,
#             args_id=[],
#             # TODO: This value needs to be changed to something else and probably used
#             # directly the przs_generator from ShareTensor - check bit_decomposition function
#             kwargs_id={},
#             ranks_to_run_action=list(range(nr_parties)),
#             result_id=result_id,
#             address=client.address,
#         )
#     )
#     return actions
