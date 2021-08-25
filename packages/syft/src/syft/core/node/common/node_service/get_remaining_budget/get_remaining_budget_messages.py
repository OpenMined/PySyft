# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# relative
from ...... import serialize
from ......core.common.serde.serializable import bind_protobuf
from ......proto.core.node.domain.service.get_remaining_budget_service_pb2 import GetRemainingBudgetMessage as GetRemainingBudgetMessage_PB
from .....common.message import ImmediateSyftMessageWithReply
from .....common.serde.deserialize import _deserialize
from .....common.uid import UID
from .....io.address import Address


@bind_protobuf
@final
class GetRemainingBudgetMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        id_at_location: UID,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.id_at_location = id_at_location

    def _object2proto(self) -> GetRemainingBudgetMessage_PB:
        return GetRemainingBudgetMessage_PB(
            id_at_location=serialize(self.id_at_location),
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(proto: GetRemainingBudgetMessage_PB) -> "GetRemainingBudgetMessage":
        return GetRemainingBudgetMessage(
            id_at_location=_deserialize(blob=proto.id_at_location),
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GetRemainingBudgetMessage_PB