# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from ..... import serialize
from .....core.common.serde.serializable import bind_protobuf
from .....proto.core.node.common.service.resolve_pointer_type_service_pb2 import (
    ResolvePointerTypeAnswerMessage as ResolvePointerTypeAnswerMessage_PB,
)
from .....proto.core.node.common.service.set_result_permission_service_pb2 import (
    SetResultPermissionMessage as SetResultPermissionMessage_PB, SetResultPermissionAnswerMessage as SetResultPermissionAnswerMessage_PB
)
from ....common.message import ImmediateSyftMessageWithReply
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.deserialize import _deserialize
from ....common.uid import UID
from ....io.address import Address
from ...abstract.node import AbstractNode
from .node_service import ImmediateNodeServiceWithReply
from syft.lib.python.dict import Dict
from syft.lib.python.string import String
from syft.core.store.permission.result_permission import ResultPermission

@bind_protobuf
class SetResultPermissionMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        id_at_location: UID,
        inputs,
        method_name,
        verify_key,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.id_at_location = id_at_location
        self.inputs = inputs
        self.method_name = method_name
        self.verify_key = verify_key

    def _object2proto(self) -> SetResultPermissionMessage_PB:
        return SetResultPermissionMessage_PB(
            id_at_location=serialize(self.id_at_location),
            address=serialize(self.address),
            msg_id=serialize(self.id),
            reply_to=serialize(self.reply_to),
            inputs={k: serialize(v) for k, v in self.inputs.items()},
            method_name=self.method_name,
            target_verify_key=bytes(self.verify_key)
        )

    @staticmethod
    def _proto2object(
        proto: SetResultPermissionMessage_PB,
    ) -> "SetResultPermissionMessage":
        return SetResultPermissionMessage(
            id_at_location=_deserialize(blob=proto.id_at_location),
            address=_deserialize(blob=proto.address),
            msg_id=_deserialize(blob=proto.msg_id),
            reply_to=_deserialize(blob=proto.reply_to),
            inputs={k: _deserialize(v) for k,v in proto.inputs.items()},
            method_name=proto.method_name,
            verify_key=VerifyKey(proto.target_verify_key)
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return SetResultPermissionMessage_PB


@bind_protobuf
class SetResultPermissionAnswerMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        status_message: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_message=status_message

    def _object2proto(self) -> SetResultPermissionAnswerMessage_PB:
        return SetResultPermissionAnswerMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            status_message=self.status_message
        )

    @staticmethod
    def _proto2object(
        proto: SetResultPermissionAnswerMessage_PB,
    ) -> "SetResultPermissionAnswerMessage":
        return SetResultPermissionAnswerMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            status_message=proto.status_message
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return SetResultPermissionAnswerMessage_PB


class SetResultPermissionService(ImmediateNodeServiceWithReply):
    @staticmethod
    def process(
        node: AbstractNode,
        msg: SetResultPermissionMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> SetResultPermissionAnswerMessage:
        if verify_key != node.root_verify_key:
            raise ValueError("Root permissions are required in order to set result permissions.")
        else:
            object = node.store[msg.id_at_location]
            result_permission = ResultPermission(
                id=msg.id_at_location,
                verify_key=msg.verify_key,
                method_name=msg.method_name,
                kwargs=msg.inputs
                )
            if result_permission not in object.result_permissions:
                object.result_permissions.append(result_permission)
            return SetResultPermissionAnswerMessage(
                address=msg.reply_to, msg_id=msg.id, status_message="success"
            )

    @staticmethod
    def message_handler_types() -> List[Type[SetResultPermissionMessage]]:
        return [SetResultPermissionMessage]
