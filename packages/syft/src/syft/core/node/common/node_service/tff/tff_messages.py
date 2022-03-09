# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
from typing_extensions import final
import tensorflow_federated as tff


# relative
from ...... import serialize
from ......proto.core.node.common.service.tff_service_pb2 import (
    TFFMessage as TFFMessage_PB,
)
from ......proto.core.node.common.service.tff_service_pb2 import (
    TFFReplyMessage as TFFReplyMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address
from ....abstract.node import AbstractNode


@serializable(recursive_serde=True)
class TFFMessageWithReply:
    __attr_allowlist__ = ["stuff"]

    def __init__(self, stuff: str) -> None:
        self.stuff = stuff

    def run(self, node: AbstractNode, verify_key: Optional[VerifyKey] = None) -> Any:
        # print(tff.federated_computation(lambda: 'Hello World')())

        return (
            "Hello from TFF Service..." + self.stuff
        )  # leaving this in for the test suite

    def prepare(self, address: Address, reply_to: Address) -> "TFFMessage":
        return TFFMessage(address=address, reply_to=reply_to, payload=self)

    @property
    def pprint(self) -> str:
        return f"TFFMessageWithReply({self.stuff})"


@serializable()
@final
class TFFMessage(ImmediateSyftMessageWithReply):
    # __attr_allowlist__ = ["id", "payload", "address", "reply_to", "msg_id"]

    def __init__(
        self,
        payload: TFFMessageWithReply,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.payload = payload

    def _object2proto(self) -> TFFMessage_PB:
        return TFFMessage_PB(
            payload=serialize(self.payload, to_bytes=True),
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(proto: TFFMessage_PB) -> "TFFMessage":
        return TFFMessage(
            payload=_deserialize(blob=proto.payload, from_bytes=True),
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return TFFMessage_PB


@serializable()
class TFFReplyMessage(ImmediateSyftMessageWithoutReply):
    # __attr_allowlist__ = ["id", "payload", "address", "msg_id"]

    def __init__(
        self,
        payload: TFFMessageWithReply,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.payload = payload
        self.address = address

    def _object2proto(self) -> TFFReplyMessage_PB:
        return TFFReplyMessage_PB(
            payload=serialize(self.payload, to_bytes=True),
            msg_id=serialize(self.id),
            address=serialize(self.address),
        )

    @staticmethod
    def _proto2object(proto: TFFReplyMessage_PB) -> "TFFReplyMessage":
        return TFFReplyMessage(
            payload=_deserialize(proto.payload, from_bytes=True),
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return TFFReplyMessage_PB
