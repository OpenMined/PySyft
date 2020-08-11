import sys

from typing import Optional

from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError
from google.protobuf.reflection import GeneratedProtocolMessageType

from syft.core.common.uid import UID
from syft.core.common.object import ObjectWithID
from syft.core.common.serde.serializable import Serializable
from syft.core.io.address import Address

import syft as sy

from ...util import get_fully_qualified_name
from ...proto.core.crypt.signed_message_pb2 import SignedMessage as SignedMessage_PB
from syft.decorators.syft_decorator_impl import syft_decorator


class SignedMessage(Serializable):

    obj_type: str
    signature: bytes
    message: bytes

    @syft_decorator(typechecking=True)
    def verified_message(self, verify_key: VerifyKey) -> Optional[Serializable]:
        try:
            verified_data = verify_key.verify(self.data, self.signature)

            # QUESTION: Is there a better way to do this?
            module_parts = self.obj_type.split(".")
            klass = module_parts.pop()
            obj_type = getattr(sys.modules[".".join(module_parts)], klass)

            # get verified data payload
            obj = sy.deserialize(blob=verified_data, from_binary=True)

            # QUESTION: This is probably not needed!?
            if type(obj) != obj_type:
                raise TypeError(
                    f"Expected type inside SignedMessage: {obj_type}. Got {type(obj)}"
                )

            return obj
        except BadSignatureError as e:
            err = (
                f"SignedMessage failed to verify signature with key: {verify_key}. {e}"
            )
            raise BadSignatureError(err)

    def __init__(self, obj_type: str, signature: bytes, message: bytes) -> None:
        super().__init__()
        self.obj_type = obj_type
        self.signature = signature
        self.data = message

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> SignedMessage_PB:

        proto = SignedMessage_PB()
        proto.obj_type = self.obj_type
        proto.signature = self.signature
        proto.data = self.data

        return proto

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: SignedMessage_PB) -> "SignedMessage":
        return SignedMessage(
            obj_type=proto.obj_type, signature=proto.signature, message=proto.data,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return SignedMessage_PB


class AbstractMessage(ObjectWithID):
    """"""


class SyftMessage(AbstractMessage):
    def __init__(self, address: Address, msg_id: Optional[UID] = None) -> None:
        self.address = address
        super().__init__(id=msg_id)

    def sign_message(self, signing_key: SigningKey) -> SignedMessage:
        signed_message = signing_key.sign(self.serialize())
        return SignedMessage(
            obj_type=get_fully_qualified_name(obj=self),
            signature=signed_message.signature,
            message=signed_message.message,
        )


class ImmediateSyftMessage(SyftMessage):
    ""


class EventualSyftMessage(SyftMessage):
    ""


class SyftMessageWithReply(SyftMessage):
    ""


class SyftMessageWithoutReply(SyftMessage):
    ""


class ImmediateSyftMessageWithoutReply(ImmediateSyftMessage, SyftMessageWithoutReply):
    def __init__(self, address: Address, msg_id: Optional[UID] = None) -> None:
        super().__init__(address=address, msg_id=msg_id)


class EventualSyftMessageWithoutReply(EventualSyftMessage, SyftMessageWithoutReply):
    def __init__(self, address: Address, msg_id: Optional[UID] = None) -> None:
        super().__init__(address=address, msg_id=msg_id)


class ImmediateSyftMessageWithReply(ImmediateSyftMessage, SyftMessageWithReply):
    def __init__(
        self, reply_to: Address, address: Address, msg_id: Optional[UID] = None
    ) -> None:
        super().__init__(address=address, msg_id=msg_id)
        self.reply_to = reply_to
