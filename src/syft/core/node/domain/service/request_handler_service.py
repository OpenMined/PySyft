# stdlib
import time
from typing import Any
from typing import Dict as DictType
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from loguru import logger
from nacl.signing import VerifyKey

# syft relative
from ..... import deserialize
from .....decorators import syft_decorator
from .....lib.python import Dict
from .....lib.python.util import downcast
from .....lib.python.util import upcast
from .....proto.core.node.domain.service.request_handler_message_pb2 import (
    GetAllRequestHandlersMessage as GetAllRequestHandlersMessage_PB,
)
from .....proto.core.node.domain.service.request_handler_message_pb2 import (
    GetAllRequestHandlersResponseMessage as GetAllRequestHandlersResponseMessage_PB,
)
from .....proto.core.node.domain.service.request_handler_message_pb2 import (
    UpdateRequestHandlerMessage as UpdateRequestHandlerMessage_PB,
)
from ....common import UID
from ....common.message import ImmediateSyftMessageWithReply
from ....common.message import ImmediateSyftMessageWithoutReply
from ....io.address import Address
from ...abstract.node import AbstractNode
from ...common.service.node_service import ImmediateNodeServiceWithoutReply


class UpdateRequestHandlerMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        handler: DictType[str, Any],
        address: Address,
        msg_id: Optional[UID] = None,
        keep: bool = True,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.handler = handler
        self.keep = keep

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> UpdateRequestHandlerMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: UpdateRequestHandlerMessage_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return UpdateRequestHandlerMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            handler=downcast(value=self.handler)._object2proto(),
            keep=self.keep,
        )

    @staticmethod
    def _proto2object(
        proto: UpdateRequestHandlerMessage_PB,
    ) -> "UpdateRequestHandlerMessage":
        """Creates a UpdateRequestHandlerMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of UpdateRequestHandlerMessage
        :rtype: UpdateRequestHandlerMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return UpdateRequestHandlerMessage(
            msg_id=deserialize(blob=proto.msg_id),
            address=deserialize(blob=proto.address),
            handler=upcast(value=Dict._proto2object(proto=proto.handler)),
            keep=proto.keep,
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

        return UpdateRequestHandlerMessage_PB


class GetAllRequestHandlersMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self, address: Address, reply_to: Address, msg_id: Optional[UID] = None
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> GetAllRequestHandlersMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: GetAllRequestHandlersMessage_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetAllRequestHandlersMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            reply_to=self.reply_to.serialize(),
        )

    @staticmethod
    def _proto2object(
        proto: GetAllRequestHandlersMessage_PB,
    ) -> "GetAllRequestHandlersMessage":
        """Creates a GetAllRequestHandlersMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of GetAllRequestHandlersMessage
        :rtype: GetAllRequestHandlersMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetAllRequestHandlersMessage(
            msg_id=deserialize(blob=proto.msg_id),
            address=deserialize(blob=proto.address),
            reply_to=deserialize(blob=proto.reply_to),
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

        return GetAllRequestHandlersMessage_PB


class GetAllRequestHandlersResponseMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        handlers: List[DictType],
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.handlers = handlers

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> GetAllRequestHandlersResponseMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: GetAllRequestHandlersResponseMessage_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return GetAllRequestHandlersResponseMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            handlers=list(
                map(lambda x: downcast(value=x)._object2proto(), self.handlers)
            ),
        )

    @staticmethod
    def _proto2object(
        proto: GetAllRequestHandlersResponseMessage_PB,
    ) -> "GetAllRequestHandlersResponseMessage":
        """Creates a GetAllRequestHandlersResponseMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of GetAllRequestHandlersResponseMessage
        :rtype: GetAllRequestHandlersResponseMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetAllRequestHandlersResponseMessage(
            msg_id=deserialize(blob=proto.msg_id),
            address=deserialize(blob=proto.address),
            handlers=[
                upcast(value=Dict._proto2object(proto=x)) for x in proto.handlers
            ],
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

        return GetAllRequestHandlersResponseMessage_PB


class UpdateRequestHandlerService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [UpdateRequestHandlerMessage]

    @staticmethod
    @syft_decorator(typechecking=True)
    def process(
        node: AbstractNode, msg: UpdateRequestHandlerMessage, verify_key: VerifyKey
    ) -> None:
        if verify_key == node.root_verify_key:
            replacement_handlers = []
            existing_handlers = getattr(node, "request_handlers", None)
            logger.debug(
                f"> Updating Request Handlers with existing: {existing_handlers}"
            )
            new_keys = set(msg.handler.keys())
            new_values = msg.handler.values()
            if existing_handlers is not None:
                for existing_handler in existing_handlers:
                    keys = set(existing_handler.keys())
                    keys.remove("created_time")  # the new handler has none
                    values = [existing_handler[key] for key in keys]
                    if keys == new_keys and set(new_values) == set(values):
                        # if keep is True we will add a new one
                        # if keep is False we will drop this anyway
                        continue
                    else:
                        # keep this handler
                        replacement_handlers.append(existing_handler)

                if msg.keep:
                    logger.debug(f"> Adding a Request Handler with: {msg.handler}")
                    msg.handler["created_time"] = time.time()
                    replacement_handlers.append(msg.handler)
                else:
                    logger.debug(f"> Removing a Request Handler with: {msg.handler}")

                setattr(node, "request_handlers", replacement_handlers)
                logger.debug(
                    f"> Finished Updating Request Handlers with: {existing_handlers}"
                )
            else:
                logger.error(f"> Node has no Request Handlers attribute: {type(node)}")

        return


class GetAllRequestHandlersService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [GetAllRequestHandlersMessage]

    @staticmethod
    @syft_decorator(typechecking=True)
    def process(
        node: AbstractNode, msg: GetAllRequestHandlersMessage, verify_key: VerifyKey
    ) -> GetAllRequestHandlersResponseMessage:

        handlers: List[DictType[str, Any]] = []
        if verify_key == node.root_verify_key:
            existing_handlers = getattr(node, "request_handlers", None)
            logger.debug(
                f"> Getting all Existing Request Handlers: {existing_handlers}"
            )
            if existing_handlers is not None:
                handlers = existing_handlers

        return GetAllRequestHandlersResponseMessage(
            handlers=handlers, address=msg.reply_to
        )
