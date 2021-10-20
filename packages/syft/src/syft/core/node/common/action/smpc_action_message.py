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
from ....tensor.smpc.share_tensor import ShareTensor


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
        ranks_to_run_action: Optional[List[int]] = None,
        msg_id: Optional[UID] = None,
    ) -> None:
        self.name_action = name_action
        self.self_id = self_id
        self.args_id = args_id
        self.kwargs_id = kwargs_id
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
    """Generator for SMPC public/private operations add/sub"""

    generator = np.random.default_rng(seed_id_locations)
    result_id = UID(UUID(bytes=generator.bytes(16)))
    other = node.store[other_id].data

    actions = []
    if isinstance(other, ShareTensor):
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


# Purposefully raise a custom error to retry the task in celery worker.
class ObjectNotInStore(Exception):
    pass


# TODO : Should move to spdz directly in syft/core/smpc
def spdz_multiply(
    x: ShareTensor,
    y: ShareTensor,
    eps_id: UID,
    delta_id: UID,
    node: Optional[Any] = None,
) -> ShareTensor:
    print(")))))))))))))))))))))))))")
    print("SPDZ multiply")
    crypto_store = ShareTensor.crypto_store
    nr_parties = x.nr_parties
    eps = node.store.get_object(key=eps_id)  # type: ignore
    delta = node.store.get_object(key=delta_id)  # type: ignore

    print("EPS Store", eps)
    print("Delta Store", delta)
    print("NR parties", nr_parties)
    if eps is None or len(eps.data) != nr_parties:
        raise ObjectNotInStore
    if delta is None or len(delta.data) != nr_parties:
        raise ObjectNotInStore
    print("Beaver Error surpassed*******************************")

    a_share, b_share, c_share = crypto_store.get_primitives_from_store(
        "beaver_mul", x.shape, y.shape
    )

    eps = sum(eps.data).child  # type: ignore
    delta = sum(delta.data).child  # type:ignore
    print(" Final EPS", eps)
    print("Final Delta", delta)
    print("A_share", a_share.child, "\n")
    print("B_share", b_share.child, "\n")
    print("C_share", c_share.child, "\n")
    op = operator.mul
    eps_b = op(eps, b_share.child)
    print("EPS_B", eps_b, "\n")
    delta_a = op(a_share.child, delta)
    print("DELTA_A", delta_a, "\n")

    tensor = c_share.child + eps_b + delta_a
    print("C addedTensor", tensor, "\n")
    if x.rank == 0:
        eps_delta = op(eps, delta)
        print("EPS_DELTA", eps_delta, "\n")
        tensor += eps_delta

    share = x.copy_tensor()
    share.child = tensor  # As we do not use fixed point we neglect truncation.
    print("Final Tensor", tensor)
    print("Finish SPDZ Multiply @@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    return share


# TODO : Should move to spdz directly in syft/core/smpc
def spdz_mask(x: ShareTensor, y: ShareTensor, eps_id: UID, delta_id: UID) -> None:  # type: ignore
    print(")))))))))))))))))))))))))")
    print("SPDZ Mask")
    crypto_store = ShareTensor.crypto_store
    clients = ShareTensor.login_clients(x.parties_info)

    a, b, _ = crypto_store.get_primitives_from_store(
        "beaver_mul", x.shape, y.shape, remove=False  # type: ignore
    )

    eps = x - a  # beaver intermediate values.
    delta = y - b
    print("x ShareTensor:", x, "\n")
    print("y ShareTensor", y, "\n")
    print("a ShareTensor:", a, "\n")
    print("b ShareTensor", b, "\n")
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
    seed_id_locations: int,
    node: Any,
    client: Any,
) -> List[SMPCActionMessage]:
    """Generator for the smpc_mul with a public value"""
    generator = np.random.default_rng(seed_id_locations)
    result_id = UID(UUID(bytes=generator.bytes(16)))
    other = node.store[other_id].data

    actions = []
    if isinstance(other, ShareTensor):
        # crypto_store = ShareTensor.crypto_store
        # _self = node.store[self_id].data
        # a_share, b_share, c_share = crypto_store.get_primitives_from_store("beaver_mul", _self.shape, other.shape)

        mask_result = UID(UUID(bytes=generator.bytes(16)))
        eps_id = UID(UUID(bytes=generator.bytes(16)))
        delta_id = UID(UUID(bytes=generator.bytes(16)))

        actions.append(
            SMPCActionMessage(
                "spdz_mask",
                self_id=self_id,
                args_id=[other_id],
                kwargs_id={"eps_id": eps_id, "delta_id": delta_id},
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
                kwargs_id={"eps_id": eps_id, "delta_id": delta_id},
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


def smpc_gt(
    nr_parties: int,
    self_id: UID,
    other_id: UID,
    seed_id_locations: int,
    node: Any,
    client: Any,
) -> List[SMPCActionMessage]:
    """Generator for the smpc_mul with a public value"""
    generator = np.random.default_rng(seed_id_locations)

    result_id = UID(UUID(bytes=generator.bytes(16)))
    sub_result = UID(UUID(bytes=generator.bytes(16)))

    x = node.store[self_id].data
    y = node.store[other_id].data

    if not isinstance(y, ShareTensor):
        raise ValueError("Only private compare works at the moment")

    actions = []
    actions.append(
        SMPCActionMessage(
            "mpc_sub",
            self_id=self_id,
            args_id=[other_id],
            kwargs_id={},
            ranks_to_run_action=list(range(nr_parties)),
            result_id=sub_result,
            address=client.address,
        )
    )

    actions.append(
        SMPCActionMessage(
            "bit_decomposition",
            self_id=sub_result,
            args_id=[],
            # TODO: This value needs to be changed to something else and probably used
            # directly the przs_generator from ShareTensor - check bit_decomposition function
            kwargs_id={},
            ranks_to_run_action=list(range(nr_parties)),
            result_id=result_id,
            address=client.address,
        )
    )
    return actions


def bit_decomposition(share: ShareTensor) -> None:  # type: ignore
    # TODO: Probably better it would be to use the PRZS from the ShareTensor
    seed_przs = 42
    generator = np.random.default_rng(seed_przs)

    print("NR PARTIES", share.nr_parties)
    print("parties_info", share.parties_info)

    # TODO: We need to take this 32 from the share ring_size
    shares = []
    for rank in range(share.nr_parties):
        if rank == share.rank:
            # we need to share the secret
            value = []
            for i in range(32):
                new_share = share.copy_tensor()
                new_share.child = ((new_share.child >> i) & 1).astype(np.int32)
                value.append(new_share)

        else:
            # just generate a random number for PRZS
            value = [None] * 32

        shares = [
            ShareTensor.generate_przs(
                value=value[i],
                ring_size=2,
                rank=share.rank,
                shape=share.child.shape,
                generator_przs=generator,
                parties_info=share.parties_info,
            )
            for i in range(32)
        ]

    print("Those are the shares", shares)
    return shares


# Given an SMPC Action map it to an action constructor
MAP_FUNC_TO_ACTION: Dict[
    str, Callable[[int, UID, UID, int, Any], List[SMPCActionMessage]]
] = {
    "__add__": functools.partial(smpc_basic_op, "add"),
    "__sub__": functools.partial(smpc_basic_op, "sub"),
    "__mul__": smpc_mul,  # type: ignore
    "__gt__": smpc_gt,  # type: ignore
}


# Map given an action map it to a function that should be run on the shares"
_MAP_ACTION_TO_FUNCTION: Dict[str, Callable[..., Any]] = {
    "mpc_add": operator.add,
    "mpc_sub": operator.sub,
    "mpc_mul": operator.mul,
    "spdz_mask": spdz_mask,
    "spdz_multiply": spdz_multiply,
    "bit_decomposition": bit_decomposition,
    "mpc_noop": deepcopy,
}
