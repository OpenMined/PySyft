# stdlib
from typing import List as TypeList
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# relative
# syft relative
from ......proto.core.node.domain.service.publish_scalars_service_pb2 import (
    GetRemainingBudgetAction as GetRemainingBudgetAction_PB,
)  # type: ignore
from .....common.message import ImmediateSyftMessageWithReply  # type: ignore
from .....common.message import ImmediateSyftMessageWithoutReply  # type: ignore
from .....common.serde.deserialize import _deserialize  # type: ignore
from .....common.serde.serializable import bind_protobuf  # type: ignore
from .....common.serde.serialize import _serialize as serialize  # type: ignore
from .....common.uid import UID  # type: ignore
from .....io.address import Address  # type: ignore


@bind_protobuf
@final
class GetRemainingBudgetMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def _object2proto(self) -> GetRemainingBudgetAction_PB:
        return GetRemainingBudgetAction_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(proto: GetRemainingBudgetAction_PB) -> "GetRemainingBudgetAction":
        return GetRemainingBudgetAction(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GetRemainingBudgetAction_PB


@bind_protobuf
@final
class GetRemainingBudgetAction(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        budget: float,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.budget = budget

    def _object2proto(self) -> GetRemainingBudgetAction_PB:
        return GetRemainingBudgetAction_PB(
            budget=self.budget,
            msg_id=serialize(self.id),
            address=serialize(self.address),
        )

    @staticmethod
    def _proto2object(proto: GetRemainingBudgetAction_PB) -> "GetRemainingBudgetAction":
        return GetRemainingBudgetAction(
            budget=proto.budget,
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GetRemainingBudgetAction_PB
