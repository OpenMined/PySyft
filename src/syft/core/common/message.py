import sys

from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Generic

from nacl.signing import SigningKey, VerifyKey

from nacl.exceptions import BadSignatureError
from google.protobuf.reflection import GeneratedProtocolMessageType

from syft.core.common.uid import UID
from syft.core.common.object import ObjectWithID
from syft.core.io.address import Address

from ...util import get_fully_qualified_name
from ...proto.core.auth.signed_message_pb2 import SignedMessage as SignedMessage_PB
from syft.decorators.syft_decorator_impl import syft_decorator
from ..common.serde.deserialize import _deserialize

# this generic type for SignedMessage
SignedMessageT = TypeVar("SignedMessageT")


class AbstractMessage(ObjectWithID, Generic[SignedMessageT]):
    """"""

    # this should be overloaded by a subclass
    signed_type: Type[SignedMessageT]


class SyftMessage(AbstractMessage):
    def __init__(self, address: Address, msg_id: Optional[UID] = None) -> None:
        self.address = address
        super().__init__(id=msg_id)

    def sign(self, signing_key: SigningKey) -> SignedMessageT:

        signed_message = signing_key.sign(self.serialize(to_binary=True))

        # signed_type will be the final subclass callee's closest parent signed_type
        # for example ReprMessage -> ImmediateSyftMessageWithoutReply.signed_type
        # == SignedImmediateSyftMessageWithoutReply
        return self.signed_type(
            address=self.address,
            obj_type=get_fully_qualified_name(obj=self),
            signature=signed_message.signature,
            verify_key=signing_key.verify_key,
            message=signed_message.message,
        )


class SignedMessage(SyftMessage):

    obj_type: str
    signature: bytes
    verify_key: VerifyKey

    def __init__(
        self,
        address: Address,
        obj_type: str,
        signature: bytes,
        verify_key: VerifyKey,
        message: bytes,
    ) -> None:
        super().__init__(address=address)
        self.obj_type = obj_type
        self.signature = signature
        self.verify_key = verify_key
        self.serialized_message = message
        self.cached_deseralized_message = None

    @property
    def message(self) -> "SyftMessage":
        if self.cached_deseralized_message is None:
            self.cached_deseralized_message = _deserialize(
                blob=self.serialized_message, from_binary=True
            )
        return self.cached_deseralized_message

    @property
    def is_valid(self) -> bool:
        try:
            _ = self.verify_key.verify(self.serialized_message, self.signature)
        except BadSignatureError:
            return False

        return True

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> SignedMessage_PB:

        proto = SignedMessage_PB()
        # obj_type will be the final subclass callee for example ReprMessage
        proto.obj_type = self.obj_type
        proto.signature = bytes(self.signature)
        proto.verify_key = bytes(self.verify_key)
        proto.message = self.serialized_message

        return proto

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: SignedMessage_PB) -> SignedMessageT:
        # TODO: horrible temp hack, need to rethink address on SignedMessage
        sub_message = _deserialize(blob=proto.message, from_binary=True)
        address = sub_message.address

        # proto.obj_type is final subclass callee for example ReprMessage
        # but we want the associated signed_type which is
        # ReprMessage -> ImmediateSyftMessageWithoutReply.signed_type
        # == SignedImmediateSyftMessageWithoutReply
        module_parts = proto.obj_type.split(".")
        klass = module_parts.pop()
        obj_type = getattr(sys.modules[".".join(module_parts)], klass)

        obj = obj_type.signed_type(
            address=address,
            obj_type=proto.obj_type,
            signature=proto.signature,
            verify_key=VerifyKey(proto.verify_key),
            message=proto.message,
        )

        if type(obj) != obj_type.signed_type:
            raise TypeError(
                "Deserializing SignedMessage. "
                + f"Expected type {obj_type.signed_type}. Got {type(obj)}"
            )

        return obj

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return SignedMessage_PB


class SignedImmediateSyftMessageWithReply(SignedMessage):
    """"""


class SignedImmediateSyftMessageWithoutReply(SignedMessage):
    """"""

    # do stuff


class SignedEventualSyftMessageWithoutReply(SignedMessage):
    """"""


class ImmediateSyftMessage(SyftMessage):
    ""


class EventualSyftMessage(SyftMessage):
    ""


class SyftMessageWithReply(SyftMessage):
    ""


class SyftMessageWithoutReply(SyftMessage):
    ""


class ImmediateSyftMessageWithoutReply(ImmediateSyftMessage, SyftMessageWithoutReply):

    signed_type = SignedImmediateSyftMessageWithoutReply

    def __init__(self, address: Address, msg_id: Optional[UID] = None) -> None:
        super().__init__(address=address, msg_id=msg_id)


class EventualSyftMessageWithoutReply(EventualSyftMessage, SyftMessageWithoutReply):

    signed_type = SignedEventualSyftMessageWithoutReply

    def __init__(self, address: Address, msg_id: Optional[UID] = None) -> None:
        super().__init__(address=address, msg_id=msg_id)


class ImmediateSyftMessageWithReply(ImmediateSyftMessage, SyftMessageWithReply):

    signed_type = SignedImmediateSyftMessageWithReply

    def __init__(
        self, reply_to: Address, address: Address, msg_id: Optional[UID] = None
    ) -> None:
        super().__init__(address=address, msg_id=msg_id)
        self.reply_to = reply_to
