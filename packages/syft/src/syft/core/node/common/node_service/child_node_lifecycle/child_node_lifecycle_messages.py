# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ......logger import debug
from ......proto.core.node.common.service.child_node_lifecycle_service_pb2 import (
    RegisterChildNodeMessage as RegisterChildNodeMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize as deserialize
from .....common.serde.serializable import serializable
from .....common.serde.serialize import _serialize as serialize
from .....common.uid import UID
from .....io.address import Address


@serializable()
class RegisterChildNodeMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        lookup_id: UID,
        child_node_client_address: Address,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.lookup_id = lookup_id
        self.child_node_client_address = child_node_client_address

    def _object2proto(self) -> RegisterChildNodeMessage_PB:
        debug(f"> {self.icon} -> Proto 🔢")
        return RegisterChildNodeMessage_PB(
            lookup_id=serialize(
                self.lookup_id
            ),  # TODO: not sure if this is needed anymore
            child_node_client_address=serialize(self.child_node_client_address),
            address=serialize(self.address),
            msg_id=serialize(self.id),
        )

    @staticmethod
    def _proto2object(proto: RegisterChildNodeMessage_PB) -> "RegisterChildNodeMessage":
        msg = RegisterChildNodeMessage(
            lookup_id=deserialize(blob=proto.lookup_id),
            child_node_client_address=deserialize(blob=proto.child_node_client_address),
            address=deserialize(blob=proto.address),
            msg_id=deserialize(blob=proto.msg_id),
        )
        debug(f"> {msg.icon} <- 🔢 Proto")
        return msg

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return RegisterChildNodeMessage_PB
