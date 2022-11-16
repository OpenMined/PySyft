# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ......proto.core.node.common.service.heritage_update_service_pb2 import (
    HeritageUpdateMessage as HeritageUpdateMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize as deserialize
from .....common.serde.serializable import serializable
from .....common.serde.serialize import _serialize as serialize
from .....common.uid import UID
from .....io.address import Address


@serializable()
class HeritageUpdateMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        new_ancestry_address: Address,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.new_ancestry_address = new_ancestry_address

    def _object2proto(self) -> HeritageUpdateMessage_PB:
        return HeritageUpdateMessage_PB(
            new_ancestry_address=serialize(self.new_ancestry_address),
            address=serialize(self.address),
            msg_id=serialize(self.id),
        )

    @staticmethod
    def _proto2object(proto: HeritageUpdateMessage_PB) -> "HeritageUpdateMessage":
        return HeritageUpdateMessage(
            new_ancestry_address=deserialize(blob=proto.new_ancestry_address),
            address=deserialize(blob=proto.address),
            msg_id=deserialize(blob=proto.msg_id),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return HeritageUpdateMessage_PB
