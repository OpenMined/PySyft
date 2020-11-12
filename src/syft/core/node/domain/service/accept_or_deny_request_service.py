# DOs and Don's of this class:
# - Do NOT use absolute syft imports (i.e. import syft.core...) Use relative ones.
# - Do NOT put multiple imports on the same line (i.e. from <x> import a, b, c). Use separate lines
# - Do sort imports by length
# - Do group imports by where they come from

# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from loguru import logger
from nacl.signing import VerifyKey
from typing_extensions import final

# syft relative
from .....decorators.syft_decorator_impl import syft_decorator
from .....proto.core.node.domain.service.accept_or_deny_request_message_pb2 import (
    AcceptOrDenyRequestMessage as AcceptOrDenyRequestMessage_PB,
)
from .....util import key_emoji
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.deserialize import _deserialize
from ....common.uid import UID
from ....io.address import Address
from ...abstract.node import AbstractNode
from ...common.service.node_service import ImmediateNodeServiceWithoutReply


@final
class AcceptOrDenyRequestMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        accept: bool,
        request_id: UID,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)

        # if false, deny the request
        self.accept = accept
        self.request_id = request_id

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> AcceptOrDenyRequestMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: AcceptOrDenyRequestMessage_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return AcceptOrDenyRequestMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            accept=self.accept,
            request_id=self.request_id.serialize(),
        )

    @staticmethod
    def _proto2object(
        proto: AcceptOrDenyRequestMessage_PB,
    ) -> "AcceptOrDenyRequestMessage":
        """Creates a AcceptOrDenyRequestMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of AcceptOrDenyRequestMessage
        :rtype: AcceptOrDenyRequestMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return AcceptOrDenyRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            accept=proto.accept,
            request_id=_deserialize(blob=proto.request_id),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type

        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.

        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return AcceptOrDenyRequestMessage_PB


class AcceptOrDenyRequestService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def process(
        node: AbstractNode, msg: AcceptOrDenyRequestMessage, verify_key: VerifyKey
    ) -> None:
        logger.debug((f"> Processing AcceptOrDenyRequestService on {node.pprint}"))
        if msg.accept:
            request_id = msg.request_id
            for req in node.requests:
                if request_id == req.id:
                    # you must be a root user to accept a request
                    if verify_key == node.root_verify_key:
                        node.store[req.object_id].read_permissions[
                            req.requester_verify_key
                        ] = req.id
                        node.requests.remove(req)

                        logger.debug(
                            f"> Accepting Request:{request_id} {request_id.emoji()}"
                        )
                        logger.debug(
                            "> Adding can_read for 🔑 "
                            + f"{key_emoji(key=req.requester_verify_key)} to "
                            + f"Store UID {req.object_id} {req.object_id.emoji()}"
                        )
                        return None

        else:
            request_id = msg.request_id
            for req in node.requests:
                if request_id == req.id:
                    # if you're a root user you can disable a request
                    # also people can disable their own requets
                    if (
                        verify_key == node.root_verify_key
                        or verify_key == req.requester_verify_key
                    ):
                        node.requests.remove(req)
                        logger.debug(f"> Rejecting Request:{request_id}")
                        return None

    @staticmethod
    def message_handler_types() -> List[Type[AcceptOrDenyRequestMessage]]:
        return [AcceptOrDenyRequestMessage]
