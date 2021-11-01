# future
from __future__ import annotations

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
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np

# syft absolute
import syft as sy

# relative
from .....proto.core.node.common.action.smpc_action_message_pb2 import (
    SMPCActionMessage as SMPCActionMessage_PB,
)
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ....tensor.smpc import utils
from ....tensor.smpc.share_tensor import ShareTensor
from ....tensor.smpc.tensor_list import TensorList
from .exceptions import ObjectNotInStore


@serializable()
class SMPCActionMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        name_action: str,
        self_id: UID,
        args_id: List[UID],
        kwargs_id: Dict[str, UID],
        result_id: UID,
        address: Address,
        kwargs: Dict[str, Any] = {},
        ranks_to_run_action: Optional[List[int]] = None,
        msg_id: Optional[UID] = None,
    ) -> None:
        self.name_action = name_action
        self.self_id = self_id
        self.args_id = args_id
        self.kwargs_id = kwargs_id
        self.kwargs = kwargs
        self.id_at_location = result_id
        self.ranks_to_run_action = ranks_to_run_action if ranks_to_run_action else []
        super().__init__(address=address, msg_id=msg_id)

    @staticmethod
    def filter_actions_after_rank(
        rank: int, actions: List[SMPCActionMessage]
    ) -> List[SMPCActionMessage]:
        """
        Filter the actions depending on the rank of each party

        Arguments:
            rank (int): the rank of the party
            actions (List[SMPCActionMessage]):

        """
        res_actions = []
        for action in actions:
            if rank in action.ranks_to_run_action:
                res_actions.append(action)

        return res_actions

    @staticmethod
    def get_action_generator_from_op(
        operation_str: str, nr_parties: int
    ) -> Callable[[UID, UID, int, Any], Any]:
        """ "
        Get the generator for the operation provided by the argument
        Arguments:
            operation_str (str): the name of the operation

        """
        return functools.partial(MAP_FUNC_TO_ACTION[operation_str], nr_parties)

    @staticmethod
    def get_id_at_location_from_op(seed: bytes, operation_str: str) -> UID:
        generator = np.random.default_rng(seed)
        return UID(UUID(bytes=generator.bytes(16)))

    def __str__(self) -> str:
        res = f"SMPCAction: {self.name_action}, "
        res = f"{res}Self ID: {self.self_id}, "
        res = f"{res}Args IDs: {self.args_id}, "
        res = f"{res}Kwargs IDs: {self.kwargs_id}, "
        res = f"{res}Kwargs : {self.kwargs}, "
        res = f"{res}Result ID: {self.id_at_location}, "
        res = f"{res}Ranks to run action: {self.ranks_to_run_action}"
        return res

    __repr__ = __str__

    def _object2proto(self) -> SMPCActionMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: SMPCActionMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return SMPCActionMessage_PB(
            name_action=self.name_action,
            self_id=sy.serialize(self.self_id),
            args_id=list(map(lambda x: sy.serialize(x), self.args_id)),
            kwargs_id={k: sy.serialize(v) for k, v in self.kwargs_id.items()},
            kwargs={k: sy.serialize(v, to_bytes=True) for k, v in self.kwargs.items()},
            id_at_location=sy.serialize(self.id_at_location),
            address=sy.serialize(self.address),
            msg_id=sy.serialize(self.id),
        )

    @staticmethod
    def _proto2object(proto: SMPCActionMessage_PB) -> SMPCActionMessage:
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of SMPCActionMessage
        :rtype: SMPCActionMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return SMPCActionMessage(
            name_action=proto.name_action,
            self_id=sy.deserialize(blob=proto.self_id),
            args_id=list(map(lambda x: sy.deserialize(blob=x), proto.args_id)),
            kwargs_id={k: sy.deserialize(blob=v) for k, v in proto.kwargs_id.items()},
            kwargs={
                k: sy.deserialize(blob=v, from_bytes=True)
                for k, v in proto.kwargs.items()
            },
            result_id=sy.deserialize(blob=proto.id_at_location),
            address=sy.deserialize(blob=proto.address),
            msg_id=sy.deserialize(blob=proto.msg_id),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type

        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.

        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return SMPCActionMessage_PB


def smpc_basic_op(
    op_str: str,
    nr_parties: int,
    self_id: UID,
    other_id: UID,
    seed_id_locations: int,
    node: Any,
    client: Any,
) -> List[SMPCActionMessage]:
    # relative
    from ..... import Tensor

    """Generator for SMPC public/private operations add/sub"""

    generator = np.random.default_rng(seed_id_locations)
    result_id = UID(UUID(bytes=generator.bytes(16)))
    other = node.store[other_id].data

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
                address=client.address,
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
                address=client.address,
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
                address=client.address,
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

    print(")))))))))))))))))))))))))")
    print("SPDZ multiply")
    nr_parties = x.nr_parties
    eps = node.store.get_object(key=eps_id)  # type: ignore
    delta = node.store.get_object(key=delta_id)  # type: ignore
    ring_size = utils.get_ring_size(x.ring_size, y.ring_size)

    print("RING SIZE", ring_size)
    print("EPS Store", eps)
    print("Delta Store", delta)
    print("NR parties", nr_parties)
    if eps is None or len(eps.data) != nr_parties:
        raise ObjectNotInStore
    if delta is None or len(delta.data) != nr_parties:
        raise ObjectNotInStore
    print("Beaver Error surpassed*******************************")

    # TODO : Should refactor fixed precision tensor later

    eps = sum(eps.data)  # type: ignore
    delta = sum(delta.data)  # type:ignore
    print(" Final EPS", eps)
    print("Final Delta", delta)
    print("A_share", a_share.child, "\n")
    print("B_share", b_share.child, "\n")
    print("C_share", c_share.child, "\n")

    eps_b = eps * b_share.child
    print("EPS_B", eps_b, "\n")
    delta_a = delta * a_share.child  # Symmetric only for mul
    print("DELTA_A", delta_a, "\n")

    tensor = c_share + eps_b + delta_a
    print("C addedTensor", tensor, "\n")
    if x.rank == 0:
        mul_op = ShareTensor.get_op(ring_size, "mul")
        eps_delta = mul_op(eps.child, delta.child)
        print("EPS_DELTA", eps_delta, "\n")
        tensor = tensor + eps_delta

    share = x.copy_tensor()
    share.child = tensor.child  # As we do not use fixed point we neglect truncation.
    print("Final Tensor", tensor)
    print("Finish SPDZ Multiply @@@@@@@@@@@@@@@@@@@@@@@@@@@@")

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
) -> None:
    # relative
    from ..... import Tensor

    if isinstance(x, Tensor) and isinstance(y, Tensor):
        x = x.child.child
        y = y.child.child

    print(")))))))))))))))))))))))))")
    print("SPDZ Mask")
    clients = ShareTensor.login_clients(x.parties_info)

    eps = x - a_share  # beaver intermediate values.
    delta = y - b_share
    print("x ShareTensor:", x, "\n")
    print("y ShareTensor", y, "\n")
    print("a ShareTensor:", a_share, "\n")
    print("b ShareTensor", b_share, "\n")
    print("EPS::::::::::::", eps, "\n")
    print("Delta::::::::::::", delta, "\n")
    # TODO : Should modify , no need to send for the current client
    # As the curent client is local.
    # TODO: clients is empty
    for rank, client in enumerate(clients):
        # if x.rank == rank:
        #    continue
        # George, commenting for now as we need to have node context when storing locally

        print("Client here", client)
        client.syft.core.smpc.protocol.spdz.spdz.beaver_populate(eps, eps_id)  # type: ignore
        client.syft.core.smpc.protocol.spdz.spdz.beaver_populate(delta, delta_id)  # type: ignore
        print("++++++++++++++++++++++++++++++++++++++++++++++")
        print("Values sent")
        print("EPS_ID", eps_id)
        print("DELTA_ID", delta_id)
    # As they are asynchronous , include them in a single action


def smpc_mul(
    nr_parties: int,
    self_id: UID,
    other_id: UID,
    a_shape_id: Optional[UID] = None,
    b_shape_id: Optional[UID] = None,
    seed_id_locations: Optional[int] = None,
    node: Optional[Any] = None,
    client: Optional[Any] = None,
) -> List[SMPCActionMessage]:
    """Generator for the smpc_mul with a public value"""
    # relative
    from ..... import Tensor

    if seed_id_locations is None or node is None or client is None:
        raise ValueError(
            f"The values seed_id_locations{seed_id_locations}, Node:{node} , client:{client} should not be None"
        )
    generator = np.random.default_rng(seed_id_locations)
    result_id = UID(UUID(bytes=generator.bytes(16)))
    other = node.store[other_id].data

    actions = []
    if isinstance(other, (ShareTensor, Tensor)):
        # crypto_store = ShareTensor.crypto_store
        # _self = node.store[self_id].data
        # a_share, b_share, c_share = crypto_store.get_primitives_from_store("beaver_mul", _self.shape, other.shape)
        if isinstance(other, ShareTensor):
            ring_size = other.ring_size
        else:
            ring_size = other.child.child.ring_size

        mask_result = UID(UUID(bytes=generator.bytes(16)))
        eps_id = UID(UUID(bytes=generator.bytes(16)))
        delta_id = UID(UUID(bytes=generator.bytes(16)))
        a_shape = node.store[a_shape_id].data
        b_shape = node.store[b_shape_id].data
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
                address=client.address,
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
                address=client.address,
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
                address=client.address,
            )
        )

    return actions


def local_decomposition(x: ShareTensor, ring_size: int, bitwise: bool) -> TensorList:
    """Performs local decomposition to generate shares of shares.

    Args:
        x (ShareTensor) : input ShareTensor.
        ring_size (str) : Ring size to generate decomposed shares in.
        bitwise (bool): Perform bit level decomposition on bits if set.

    Returns:
        List[List[ShareTensor]]: Decomposed shares in the given ring size.
    """
    # TODO: George or Rasswanth check if we can use directly the generator from shareTensor
    # Having this value here is not ok
    # seed_przs = 42
    # generator = np.random.default_rng(seed_przs)
    # absolute
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

    share_lst = TensorList()

    input_shares = []

    if bitwise:
        ring_bits = utils.get_nr_bits(x.ring_size)  # for bit-wise decomposition
        input_shares = [x.bit_extraction(idx) for idx in range(ring_bits)]
    else:
        input_shares.append(x)

    for share in input_shares:
        share_sh = TensorList()
        for i in range(nr_parties):
            sh = x.copy_tensor()
            sh.ring_size = ring_size
            if rank != i:
                sh.child = deepcopy(zero)
            else:
                sh.child = deepcopy(share.child.astype(numpy_type))
            if TENSOR_FLAG:
                t_sh = t.copy()
                t_sh.child.child = sh
                share_sh.append(t_sh)  # type: ignore
            else:
                share_sh.append(sh)
        share_lst.append(share_sh)

    return share_lst


def bit_decomposition(
    nr_parties: int,
    self_id: UID,
    ring_size: UID,
    bitwise: UID,
    seed_id_locations: int,
    node: Any,
    client: Any,
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
            ranks_to_run_action=list(range(nr_parties)),
            result_id=result_id,
            address=client.address,
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

#     x = node.store[self_id].data  # noqa
#     y = node.store[other_id].data

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
