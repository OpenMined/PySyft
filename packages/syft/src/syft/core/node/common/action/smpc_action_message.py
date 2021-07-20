# stdlib
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from uuid import UUID

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
import numpy as np

# syft absolute
from syft import lib

# relative
# syft relative
from ..... import serialize
from .....logger import traceback_and_raise
from .....proto.core.node.common.action.smpc_action_message_pb2 import (
    SMPCActionMessage as SMPCActionMessage_PB,
)
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.deserialize import _deserialize
from ....common.serde.serializable import bind_protobuf
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ....tensor.smpc.share_tensor import ShareTensor
from ...abstract.node import AbstractNode

MAP_FUNC_TO_NR_GENERATOR_INVOKES = {"__add__": 0, "__mul__": 0, "__sub__": 0}


@bind_protobuf
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
        self.ranks_to_run_action = ranks_to_run_action
        self.address = address
        self.msg_id = msg_id
        super().__init__(address=address, msg_id=msg_id)

    @staticmethod
    def filter_actions_after_rank(
        rank: int, actions: List["SMPCActionMessage"]
    ) -> List["SMPCActionMessage"]:
        res_actions = []
        for action in actions:
            if action.ranks_to_run_action is None:
                raise ValueError(
                    "Attribute ranks_to_run_action should not be None when filtering"
                )

            if action.ranks_to_run_action == [] or rank in action.ranks_to_run_action:
                res_actions.append(action)
        return res_actions

    @staticmethod
    def get_action_generator_from_op(
        operation_str: str,
    ) -> Callable[[UID, UID, int, Any], Any]:
        return MAP_FUNC_TO_ACTION[operation_str]

    def _object2proto(self) -> SMPCActionMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: RunClassMethodAction_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return SMPCActionMessage_PB(
            name_action=self.name_action,
            self_id=serialize(self.self_id),
            args_id=list(map(lambda x: serialize(x), self.args_id)),
            kwargs_id={k: serialize(v) for k, v in self.kwargs_id.items()},
            id_at_location=serialize(self.id_at_location),
        )

    @staticmethod
    def _proto2object(proto: SMPCActionMessage_PB) -> "SMPCActionMessage":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of RunClassMethodAction
        :rtype: RunClassMethodAction

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return SMPCActionMessage(
            name_action=proto.name_action,
            self_id=_deserialize(blob=proto.self_id),
            args_id=list(map(lambda x: _deserialize(blob=x), proto.args_id)),
            kwargs_id={k: v for k, v in proto.kwargs_id.items()},
            result_id=_deserialize(blob=proto.id_at_location),
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


def smpc_add(
    self_id: UID, other_id: UID, seed_id_locations: int, node: Any
) -> List[SMPCActionMessage]:
    generator = np.random.default_rng(seed_id_locations)

    for _ in range(MAP_FUNC_TO_NR_GENERATOR_INVOKES["__add__"]):
        generator.bytes(16)

    result_id = UID(UUID(bytes=generator.bytes(16)))
    other = node.store[other_id].data

    actions = []
    if isinstance(other, ShareTensor):
        # All parties should add the other share if empty list
        actions.append(
            SMPCActionMessage(
                "mpc_add",
                self_id=self_id,
                args_id=[other_id],
                kwargs_id={},
                ranks_to_run_action=[],
                result_id=result_id,
                address=node.address,
            )
        )
    else:
        # Only rank 0 (the first party) would add that public value
        actions.append(
            SMPCActionMessage(
                "mpc_add",
                self_id=self_id,
                args_id=[other_id],
                kwargs_id={},
                ranks_to_run_action=[0],
                result_id=result_id,
                address=node.address,
            )
        )

    return actions


def smpc_sub(
    self_id: UID, other_id: UID, seed_id_locations: int, node: Any
) -> List[SMPCActionMessage]:
    generator = np.random.default_rng(seed_id_locations)

    for _ in range(MAP_FUNC_TO_NR_GENERATOR_INVOKES["__sub__"]):
        generator.bytes(16)

    result_id = UID(UUID(bytes=generator.bytes(16)))
    other = node.store[other_id].data

    actions = []
    if isinstance(other, ShareTensor):
        # All parties should add the other share if empty list
        actions.append(
            SMPCActionMessage(
                "mpc_sub",
                self_id=self_id,
                args_id=[other_id],
                kwargs_id={},
                ranks_to_run_action=[],
                result_id=result_id,
                address=node.address,
            )
        )
    else:
        # Only rank 0 (the first party) would add that public value
        actions.append(
            SMPCActionMessage(
                "mpc_sub",
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
    self_id: UID, other_id: UID, seed_id_locations: int, node: Any
) -> List[SMPCActionMessage]:
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
                ranks_to_run_action=[],
                result_id=result_id,
                address=node.address,
            )
        )

    return actions


MAP_FUNC_TO_ACTION: Dict[
    str, Callable[[UID, UID, int, Any], List[SMPCActionMessage]]
] = {
    "__add__": smpc_add,
    "__mul__": smpc_mul,
    "__sub__": smpc_sub,
}


_MAP_ACTION_TO_FUNCTION: Dict[str, Callable[..., Any]] = {
    "mpc_add": operator.add,
    "mpc_sub": operator.sub,
    "mpc_mul": operator.mul,
}
