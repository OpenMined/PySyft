# future
from __future__ import annotations

# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from ...... import serialize
from ......lib.python import Dict
from ......proto.core.node.common.service.remote_add_service_pb2 import (
    RemoteAddMessage as RemoteAddMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize
from .....common.serde.serializable import bind_protobuf
from .....common.uid import UID
from .....io.address import Address
from ....abstract.node import AbstractNode
from ..node_service import ImmediateNodeServiceWithoutReply


@bind_protobuf
@final
class RemoteAddMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        num: int,
        id_at_location: UID,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.num = num
        self.id_at_location = id_at_location

    def _object2proto(self) -> RemoteAddMessage_PB:
        return RemoteAddMessage_PB(
            msg_id=serialize(self.id),
            id_at_location=serialize(self.id_at_location),
            address=serialize(self.address),
            num=self.num,
        )

    @staticmethod
    def _proto2object(proto: RemoteAddMessage_PB) -> RemoteAddMessage:
        return RemoteAddMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            id_at_location=_deserialize(blob=proto.id_at_location),
            address=_deserialize(blob=proto.address),
            num=proto.num,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return RemoteAddMessage_PB


class RemoteAddService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def process(
        node: AbstractNode, msg: RemoteAddMessage, verify_key: VerifyKey
    ) -> None:
        # grid absolute
        from app.core.celery_app import celery_app

        args = Dict(
            {
                "num": msg.num,
                "id_at_location": msg.id_at_location,
                "verify_key": verify_key,
            }
        )

        # use latin-1 instead of utf-8 because our bytes might not be an even number
        arg_bytes_str = serialize(args, to_bytes=True).decode("latin-1")
        celery_app.send_task("app.worker.add_num", args=[arg_bytes_str])

    @staticmethod
    def message_handler_types() -> List[Type[RemoteAddMessage]]:
        return [RemoteAddMessage]
