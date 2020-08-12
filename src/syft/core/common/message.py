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
from ...proto.core.auth.signed_message_pb2 import SignedMessage as SignedMessage_PB
from syft.decorators.syft_decorator_impl import syft_decorator


class SignedMessage(Serializable):

    obj_type: str
    signature: bytes
    verify_key: bytes
    message: bytes

    """
    valid = msg.is_valid()
            inner_msg = self.inner_message(allow_invalid=True)
    """

    @syft_decorator(typechecking=True)
    def is_valid(self) -> bool:
        valid = False
        try:
            verify_key: VerifyKey = VerifyKey(self.verify_key)
            _ = verify_key.verify(self.data, self.signature)

            valid = True
        except Exception:
            # not valid
            pass

        return valid

    @syft_decorator(typechecking=True)
    def inner_message(self, allow_invalid: bool = False) -> Optional[Serializable]:
        data = self.data

        # only use the data if the verification passes first
        if allow_invalid is False:
            try:
                verify_key: VerifyKey = VerifyKey(self.verify_key)
                data = verify_key.verify(self.data, self.signature)

                # QUESTION: Is there a better way to do this?
                module_parts = self.obj_type.split(".")
                klass = module_parts.pop()
                obj_type = getattr(sys.modules[".".join(module_parts)], klass)
            except BadSignatureError as e:
                err = f"SignedMessage failed to verify signature with key: {verify_key}. {e}"
                raise BadSignatureError(err)

        # get verified data payload
        obj = sy.deserialize(blob=data, from_binary=True)

        if allow_invalid is False:
            if type(obj) != obj_type:
                raise TypeError(
                    f"Expected type inside SignedMessage: {obj_type}. Got {type(obj)}"
                )

        return obj

    def __init__(
        self, obj_type: str, signature: bytes, verify_key: bytes, message: bytes
    ) -> None:
        super().__init__()
        self.obj_type = obj_type
        self.signature = signature
        self.verify_key = verify_key
        self.data = message

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> SignedMessage_PB:

        proto = SignedMessage_PB()
        proto.obj_type = self.obj_type
        proto.signature = self.signature
        proto.verify_key = self.verify_key
        proto.data = self.data

        return proto

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: SignedMessage_PB) -> "SignedMessage":
        return SignedMessage(
            obj_type=proto.obj_type,
            signature=proto.signature,
            verify_key=proto.verify_key,
            message=proto.data,
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
        signed_message = signing_key.sign(self.serialize(to_binary=True))
        return SignedMessage(
            obj_type=get_fully_qualified_name(obj=self),
            signature=signed_message.signature,
            verify_key=bytes(signing_key.verify_key),
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
