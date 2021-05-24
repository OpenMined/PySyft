# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
from typing_extensions import final

# syft relative
from ..... import serialize
from .....core.common.serde.serializable import bind_protobuf
from .....proto.core.node.common.service.get_repr_service_pb2 import (
    GetReprMessage as GetReprMessage_PB,
)
from .....proto.core.node.common.service.get_repr_service_pb2 import (
    GetReprReplyMessage as GetReprReplyMessage_PB,
)
from .....util import traceback_and_raise
from ....common.group import VERIFYALL
from ....common.message import ImmediateSyftMessageWithReply
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.deserialize import _deserialize
from ....common.uid import UID
from ....io.address import Address
from ...abstract.node import AbstractNode
from .auth import service_auth
from .node_service import ImmediateNodeServiceWithReply


@bind_protobuf
@final
class GetReprMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        id_at_location: UID,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.id_at_location = id_at_location

    def _object2proto(self) -> GetReprMessage_PB:
        return GetReprMessage_PB(
            id_at_location=serialize(self.id_at_location),
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(proto: GetReprMessage_PB) -> "GetReprMessage":
        return GetReprMessage(
            id_at_location=_deserialize(blob=proto.id_at_location),
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GetReprMessage_PB


@bind_protobuf
class GetReprReplyMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        repr: str,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.repr = repr

    def _object2proto(self) -> GetReprReplyMessage_PB:
        return GetReprReplyMessage_PB(
            repr=self.repr,
            msg_id=serialize(self.id),
            address=serialize(self.address),
        )

    @staticmethod
    def _proto2object(proto: GetReprReplyMessage_PB) -> "GetReprReplyMessage":
        return GetReprReplyMessage(
            repr=proto.repr,
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GetReprReplyMessage_PB


class GetReprService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(root_only=True)
    def process(
        node: AbstractNode,
        msg: GetReprMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> GetReprReplyMessage:
        if verify_key is None:
            traceback_and_raise(
                "Can't process an GetReprService with no verification key."
            )

        obj = node.store[msg.id_at_location]
        contains_all_in_permissions = any(
            key is VERIFYALL for key in obj.read_permissions.keys()
        )

        if not (
            verify_key in obj.read_permissions.keys()
            or verify_key == node.root_verify_key
            or contains_all_in_permissions
        ):
            raise PermissionError("Permission to get repr of object not granted!")
        else:
            # TODO: Create a remote print interface for objects which displays them in a
            # nice way, we could also even buffer this between chained ops until we
            # return so that we can print once and display a nice list of data and ops
            # issue: https://github.com/OpenMined/PySyft/issues/5167
            result = repr(obj.data)
            return GetReprReplyMessage(repr=result, address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[GetReprMessage]]:
        return [GetReprMessage]
