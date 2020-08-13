from typing import List, Type, Optional

from google.protobuf.reflection import GeneratedProtocolMessageType

from syft.proto.core.auth.hello_root_pb2 import HelloRootRequest as HelloRootRequest_PB
from syft.proto.core.auth.hello_root_pb2 import (
    HelloRootResponse as HelloRootResponse_PB,
)

from syft.core.common.message import (
    ImmediateSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
)
from syft.core.io.address import Address
from syft.core.common.uid import UID
from syft.core.common.serde.deserialize import _deserialize

from .....decorators import syft_decorator
from ...abstract.node import AbstractNode
from ...common.service.node_service import ImmediateNodeServiceWithReply


class ImmediateAuthorizedMessageWithReply(ImmediateSyftMessageWithReply):
    pass


class ImmediateAuthorizedServiceWithReply(ImmediateNodeServiceWithReply):
    @syft_decorator(typechecking=True)
    def process(
        self,
        node: AbstractNode,
        msg: ImmediateAuthorizedMessageWithReply,
        valid: bool = False,
    ) -> ImmediateSyftMessageWithoutReply:
        print(node.__repr__(), msg)
        return msg.execute_action(node=node, valid=valid)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[ImmediateAuthorizedMessageWithReply]]:
        return [ImmediateAuthorizedMessageWithReply]


# This is a demo service for working with SignedMessage / Auth
# It should only say hello to the validated root user
class HelloRootServiceWithReply(ImmediateAuthorizedServiceWithReply):
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[ImmediateAuthorizedMessageWithReply]]:
        return [HelloRootRequest]


class HelloRootRequest(ImmediateAuthorizedMessageWithReply):
    def __init__(
        self,
        username: str,
        reply_to: Address,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(reply_to=reply_to, address=address, msg_id=msg_id)
        self.username = username

    def execute_action(
        self, node: AbstractNode, valid: bool = False
    ) -> ImmediateSyftMessageWithoutReply:
        if valid is True:
            message = f"Hello {self.username}"
        else:
            message = f"Unauthorized. Go away {self.username}!"

        return HelloRootResponse(address=self.reply_to, message=message)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> HelloRootRequest_PB:
        proto = HelloRootRequest_PB()
        proto.reply_to.CopyFrom(self.reply_to.proto())
        proto.address.CopyFrom(self.address.proto())
        proto.username = self.username
        proto.msg_id.CopyFrom(self.id.proto())

        return proto

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: HelloRootRequest_PB) -> "HelloRootRequest":
        return HelloRootRequest(
            reply_to=_deserialize(blob=proto.reply_to),
            address=_deserialize(blob=proto.address),
            username=proto.username,
            msg_id=_deserialize(blob=proto.msg_id),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return HelloRootRequest_PB


class HelloRootResponse(ImmediateSyftMessageWithoutReply):
    def __init__(self, address: Address, message: str, msg_id: Optional[UID] = None):
        super().__init__(address=address, msg_id=msg_id)
        self.message = message

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> HelloRootResponse_PB:
        proto = HelloRootResponse_PB()
        proto.address.CopyFrom(self.address.proto())
        proto.message = self.message
        proto.msg_id.CopyFrom(self.id.proto())

        return proto

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: HelloRootResponse_PB) -> "HelloRootResponse":
        return HelloRootResponse(
            address=_deserialize(blob=proto.address),
            message=proto.message,
            msg_id=_deserialize(blob=proto.msg_id),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return HelloRootResponse_PB
