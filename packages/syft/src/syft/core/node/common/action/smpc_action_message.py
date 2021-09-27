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

# How many intermediary ids we generate in each smpc function
MAP_FUNC_TO_NR_GENERATOR_INVOKES = {"__add__": 0, "__mul__": 0, "__sub__": 0}


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
        self.address = address
        self.msg_id = msg_id
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
        nr_ops = MAP_FUNC_TO_NR_GENERATOR_INVOKES[operation_str]
        for _ in range(nr_ops):
            generator.bytes(16)

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
            kwargs_id={k: v for k, v in proto.kwargs_id.items()},
            result_id=sy.deserialize(blob=proto.id_at_location),
            address=proto,
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
) -> List[SMPCActionMessage]:
    """Generator for SMPC public/private operations add/sub"""

    generator = np.random.default_rng(seed_id_locations)

    for _ in range(MAP_FUNC_TO_NR_GENERATOR_INVOKES[f"__{op_str}__"]):
        generator.bytes(16)

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


def smpc_mul(
    nr_parties: int, self_id: UID, other_id: UID, seed_id_locations: int, node: Any
) -> List[SMPCActionMessage]:
    """Generator for the smpc_mul with a public value"""
    generator = np.random.default_rng(seed_id_locations)

    for _ in range(MAP_FUNC_TO_NR_GENERATOR_INVOKES["__mul__"]):
        generator.bytes(16)

    result_id = UID(UUID(bytes=generator.bytes(16)))
    other = node.store[other_id].data

    actions = []
    if isinstance(other, ShareTensor):
        raise ValueError("Not yet implemented Private Multiplication")
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

    return actions


# Given an SMPC Action map it to an action constructor
MAP_FUNC_TO_ACTION: Dict[
    str, Callable[[int, UID, UID, int, Any], List[SMPCActionMessage]]
] = {
    "__add__": functools.partial(smpc_basic_op, "add"),
    "__sub__": functools.partial(smpc_basic_op, "sub"),
    "__mul__": smpc_mul,
}


# Map given an action map it to a function that should be run on the shares"
_MAP_ACTION_TO_FUNCTION: Dict[str, Callable[..., Any]] = {
    "mpc_add": operator.add,
    "mpc_sub": operator.sub,
    "mpc_mul": operator.mul,
    "mpc_noop": deepcopy,
}
