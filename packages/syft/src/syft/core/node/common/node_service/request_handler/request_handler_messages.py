# stdlib
from typing import Any
from typing import Dict as DictType
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ...... import deserialize
from ...... import serialize
from ......lib.python import Dict
from ......lib.python.util import downcast
from ......lib.python.util import upcast
from ......proto.core.node.domain.service.request_handler_message_pb2 import (
    GetAllRequestHandlersMessage as GetAllRequestHandlersMessage_PB,
)
from ......proto.core.node.domain.service.request_handler_message_pb2 import (
    GetAllRequestHandlersResponseMessage as GetAllRequestHandlersResponseMessage_PB,
)
from ......proto.core.node.domain.service.request_handler_message_pb2 import (
    UpdateRequestHandlerMessage as UpdateRequestHandlerMessage_PB,
)
from .....common import UID
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....io.address import Address


@serializable()
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

    def _object2proto(self) -> UpdateRequestHandlerMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: UpdateRequestHandlerMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return UpdateRequestHandlerMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
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


@serializable()
class GetAllRequestHandlersMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self, address: Address, reply_to: Address, msg_id: Optional[UID] = None
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def _object2proto(self) -> GetAllRequestHandlersMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: GetAllRequestHandlersMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetAllRequestHandlersMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
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


@serializable()
class GetAllRequestHandlersResponseMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        handlers: List[DictType],
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.handlers = handlers

    def _object2proto(self) -> GetAllRequestHandlersResponseMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: GetAllRequestHandlersResponseMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        # For handler["created_time"], it's a large number. In order to keep it's precision
        # when serde, we turn it to string, and then turn it back to float in _proto2object.
        handlers = [h.copy() for h in self.handlers]
        for handler in handlers:
            if "created_time" in handler:
                handler["created_time"] = str(handler["created_time"])

        return GetAllRequestHandlersResponseMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            handlers=list(map(lambda x: downcast(value=x)._object2proto(), handlers)),
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
        handlers = [upcast(value=Dict._proto2object(proto=x)) for x in proto.handlers]
        for handler in handlers:
            if "created_time" in handler:
                handler["created_time"] = float(handler["created_time"])

        return GetAllRequestHandlersResponseMessage(
            msg_id=deserialize(blob=proto.msg_id),
            address=deserialize(blob=proto.address),
            handlers=handlers,
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
