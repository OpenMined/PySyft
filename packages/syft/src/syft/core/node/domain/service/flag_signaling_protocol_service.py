# stdlib
from typing import List
from typing import Optional
from enum import Enum

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
from syft import Domain
from typing_extensions import final

# syft relative
from ..... import serialize, deserialize
from ....common.message import SyftMessage
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.serializable import bind_protobuf
from ....common.uid import UID
from ....io.address import Address
from ...common.service.node_service import ImmediateNodeServiceWithReply
from ....common.message import ImmediateSyftMessageWithReply
from .....proto.core.node.domain.service.flag_signaling_protocol_service_pb2 import SetProtocolMessage as SetProtocolMessage_PB
from .....proto.core.node.domain.service.flag_signaling_protocol_service_pb2 import SetProtocolMessageReply as SetProtocolMessageReply_PB


@bind_protobuf
@final
class SetProtocolMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        flags: Enum,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.flags = flags

    def _object2proto(self) -> SetProtocolMessage_PB:
        proto = SetProtocolMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
            name=self.flags.__name__
        )

        for enum_entry in self.flags.__members__.values():
            proto.enum_keys.append(enum_entry.name)
            proto.enum_values.append(enum_entry.value)

        return proto

    @staticmethod
    def _proto2object(
        proto: SetProtocolMessage_PB,
    ) -> "SetProtocolMessage":
        flags = Enum(proto.name, list(zip(proto.enum_keys, proto.enum_values)))


        return SetProtocolMessage(
            flags=flags,
            msg_id=deserialize(proto.msg_id, from_proto=True),
            address=deserialize(proto.address, from_proto=True),
            reply_to=deserialize(proto.reply_to, from_proto=True),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return SetProtocolMessage_PB

@bind_protobuf
@final
class SetProtocolMessageReply(ImmediateSyftMessageWithoutReply):
    def __init__(
            self,
            response: bool,
            address: Address,
            msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.response = response

    def _object2proto(self) -> SetProtocolMessageReply_PB:
        return SetProtocolMessageReply_PB(
            response=self.response,
            msg_id=serialize(self.id),
            address=serialize(self.address),
        )

    @staticmethod
    def _proto2object(
            proto: SetProtocolMessageReply_PB,
    ) -> "SetProtocolMessageReply":
        return SetProtocolMessageReply(
            response=proto.response,
            msg_id=deserialize(proto.msg_id, from_proto=True),
            address=deserialize(proto.address, from_proto=True),
        )


    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return SetProtocolMessageReply_PB


class FlagSignalingProtocolService(ImmediateNodeServiceWithReply):
    @staticmethod
    def process(
        node: Domain,
        msg: ImmediateSyftMessageWithReply,
        verify_key: Optional[VerifyKey] = None,
    ) -> SyftMessage:
        if isinstance(msg, SetProtocolMessage):
            answer = SetProtocolMessageReply(
                response=False,
                address=msg.reply_to
            )

            if node.flags is None:
                node.flags = msg.flags
                answer.response = True
                return answer
            return answer

    @staticmethod
    def message_handler_types() -> List[SyftMessage]:
        return [SetProtocolMessage_PB]
