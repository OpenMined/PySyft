# future
from __future__ import annotations

# stdlib
from typing import Generic
from typing import Optional
from typing import Sequence
from typing import Type
from typing import TypeVar

# third party
from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# relative
from ...logger import debug
from ...logger import traceback_and_raise
from ...util import validate_type
from ..io.address import Address
from .object import ObjectWithID
from .serde.deserialize import _deserialize
from .serde.serializable import serializable
from .serde.serialize import _serialize as serialize
from .uid import UID

# this generic type for SignedMessage
SignedMessageT = TypeVar("SignedMessageT")


class AbstractMessage(ObjectWithID, Generic[SignedMessageT]):
    """ """

    # this should be overloaded by a subclass
    signed_type: Type[SignedMessageT]

    @property
    def class_name(self) -> str:
        return str(self.__class__.__name__)

    @property
    def icon(self) -> str:
        icon = "✉️ "
        if "signed" in self.class_name.lower():
            icon += "🔏"
        return icon

    @property
    def pprint(self) -> str:
        return f"{self.icon} ({self.class_name})"

    def post_init(self) -> None:
        init_reason = "Creating"
        if "signed" in self.class_name.lower():
            init_reason += " Signed"
        debug(f"> {init_reason} {self.pprint} {self.id.emoji()}")


class SyftMessage(AbstractMessage):
    """
    SyftMessages are an abstraction that represents information that is sent between a :class:`Client`
    and a :class:`Node`. In Syft's decentralized setup, we can easily see why SyftMessages are so important.
    This class cannot be used as-is: to get some useful objects, we need to derive from it. For instance,
    :class:`Action`s inherit from :class:`SyftMessage`.
    There are many types of SyftMessage which boil down to whether or not they are Sync or Async,
    and whether or not they expect a response.

    Attributes:
        address: the :class:`Address` to which the message needs to be delivered.
    """

    def __init__(self, address: Address, msg_id: Optional[UID] = None) -> None:
        self.address = address
        super().__init__(id=msg_id)
        self.post_init()

    def sign(self, signing_key: SigningKey) -> SignedMessageT:
        """
        It's important for all messages to be able to prove who they were sent from.
        This method endows every message with the ability for someone to "sign" (with a hash of the message)
        as the sender of the message so that someone else, at a later date, can verify the sender.

        Args:
            signing_key: The key to use to sign the SyftMessage.

        Returns:
            A :class:`SignedMessage`

        """
        debug(f"> Signing with {self.address.key_emoji(key=signing_key.verify_key)}")
        signed_message = signing_key.sign(serialize(self, to_bytes=True))

        # signed_type will be the final subclass callee's closest parent signed_type
        # for example ReprMessage -> ImmediateSyftMessageWithoutReply.signed_type
        # == SignedImmediateSyftMessageWithoutReply
        return self.signed_type(
            msg_id=self.id,
            address=self.address,
            signature=signed_message.signature,
            verify_key=signing_key.verify_key,
            message=signed_message.message,
        )


class SignedMessage(SyftMessage):
    """
    SignedMessages are :class:`SyftMessage`s that have been signed by someone.
    In addition to what has a :class:`SyftMessage`, they have a signature, a verify key
    and a :meth:`is_valid` property that is here to check that the message was really
    signed and sent by the verify key owner.

    Attributes:
        obj_type (string): the string representation of the type of the original message.
        signature (bytes): the signature of the message.
        verify_key (VerifyKey): the signer's public key with which the signature can be verified.
        serialized_message: the serialized original message.
    """

    signature: bytes
    verify_key: VerifyKey

    __attr_allowlist__: Sequence[str] = [
        "id",
        "address",
        "signature",
        "verify_key",
        "serialized_message",
    ]

    def __init__(
        self,
        address: Address,
        signature: bytes,
        verify_key: VerifyKey,
        message: bytes,
        msg_id: Optional[UID] = None,
    ) -> None:
        super().__init__(msg_id=msg_id, address=address)
        self.signature = signature
        self.verify_key = verify_key
        self.serialized_message = message
        self.cached_deseralized_message: Optional[SyftMessage] = None

    @property
    def message(self) -> SyftMessage:
        # from deserialize we might not have this attr because __init__ is skipped
        if not hasattr(self, "cached_deseralized_message"):
            self.cached_deseralized_message = None

        if self.cached_deseralized_message is None:
            _syft_msg = validate_type(
                _deserialize(blob=self.serialized_message, from_bytes=True), SyftMessage
            )
            self.cached_deseralized_message = _syft_msg

        if self.cached_deseralized_message is None:
            traceback_and_raise(
                ValueError(
                    f"Can't deserialize message {self} with address " f"{self.address}"
                )
            )

        return self.cached_deseralized_message

    @property
    def is_valid(self) -> bool:
        try:
            _ = self.verify_key.verify(self.serialized_message, self.signature)
        except BadSignatureError:
            return False

        return True

    def __hash__(self) -> int:
        return hash((self.signature, self.verify_key))


@serializable(recursive_serde=True)
class SignedImmediateSyftMessageWithReply(SignedMessage):
    pass


@serializable(recursive_serde=True)
class SignedImmediateSyftMessageWithoutReply(SignedMessage):
    pass


@serializable(recursive_serde=True)
class SignedImmediateSyftMessage(SignedMessage):
    pass


class ImmediateSyftMessage(SyftMessage):
    signed_type = SignedImmediateSyftMessage


class SyftMessageWithReply(SyftMessage):
    def __init__(
        self, reply_to: Address, address: Address, msg_id: Optional[UID] = None
    ) -> None:
        super().__init__(address=address, msg_id=msg_id)
        self.reply_to = reply_to


class SyftMessageWithoutReply(SyftMessage):
    pass


class ImmediateSyftMessageWithoutReply(SyftMessageWithoutReply):
    signed_type = SignedImmediateSyftMessageWithoutReply


class ImmediateSyftMessageWithReply(SyftMessageWithReply):
    signed_type = SignedImmediateSyftMessageWithReply
