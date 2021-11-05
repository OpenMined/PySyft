# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

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
        kwargs: Optional[Dict[str, Any]] = None,
        ranks_to_run_action: Optional[List[int]] = None,
        msg_id: Optional[UID] = None,
    ) -> None:
        self.name_action = name_action
        self.self_id = self_id
        self.args_id = args_id
        self.kwargs_id = kwargs_id
        if kwargs is None:
            self.kwargs = {}
        else:
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
