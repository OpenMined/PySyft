# stdlib
import json
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# relative
from ...... import serialize
from ......proto.grid.messages.request_messages_pb2 import (
    CreateBudgetRequestMessage as CreateBudgetRequestMessage_PB,
)
from ......proto.grid.messages.request_messages_pb2 import (
    CreateRequestMessage as CreateRequestMessage_PB,
)
from ......proto.grid.messages.request_messages_pb2 import (
    CreateRequestResponse as CreateRequestResponse_PB,
)
from ......proto.grid.messages.request_messages_pb2 import (
    DeleteRequestMessage as DeleteRequestMessage_PB,
)
from ......proto.grid.messages.request_messages_pb2 import (
    DeleteRequestResponse as DeleteRequestResponse_PB,
)
from ......proto.grid.messages.request_messages_pb2 import (
    GetAllRequestsMessage as GetAllRequestsMessage_PB,
)
from ......proto.grid.messages.request_messages_pb2 import (
    GetAllRequestsResponseMessage as GetAllRequestsResponseMessage_PB,
)
from ......proto.grid.messages.request_messages_pb2 import (
    GetBudgetRequestsMessage as GetBudgetRequestsMessage_PB,
)
from ......proto.grid.messages.request_messages_pb2 import (
    GetBudgetRequestsResponse as GetBudgetRequestsResponse_PB,
)
from ......proto.grid.messages.request_messages_pb2 import (
    GetRequestMessage as GetRequestMessage_PB,
)
from ......proto.grid.messages.request_messages_pb2 import (
    GetRequestResponse as GetRequestResponse_PB,
)
from ......proto.grid.messages.request_messages_pb2 import (
    GetRequestsMessage as GetRequestsMessage_PB,
)
from ......proto.grid.messages.request_messages_pb2 import (
    GetRequestsResponse as GetRequestsResponse_PB,
)
from ......proto.grid.messages.request_messages_pb2 import (
    UpdateRequestMessage as UpdateRequestMessage_PB,
)
from ......proto.grid.messages.request_messages_pb2 import (
    UpdateRequestResponse as UpdateRequestResponse_PB,
)
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address
from ..request_receiver.request_receiver_messages import RequestMessage


@serializable()
@final
class CreateRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        content: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.content = content

    def _object2proto(self) -> CreateRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: CreateRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return CreateRequestMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            content=json.dumps(self.content),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: CreateRequestMessage_PB,
    ) -> "CreateRequestMessage":
        """Creates a CreateRequestMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: CreateRequestMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return CreateRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            content=json.loads(proto.content),
            reply_to=_deserialize(blob=proto.reply_to),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return CreateRequestMessage_PB


@serializable()
@final
class CreateBudgetRequestMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        budget: float,
        reason: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.budget = budget
        self.reason = reason

    def _object2proto(self) -> CreateBudgetRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: CreateBudgetRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return CreateBudgetRequestMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            budget=self.budget,
            reason=self.reason,
        )

    @staticmethod
    def _proto2object(
        proto: CreateBudgetRequestMessage_PB,
    ) -> "CreateBudgetRequestMessage":
        """Creates a CreateRequestMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: CreateRequestMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return CreateBudgetRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            budget=proto.budget,
            reason=proto.reason,
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return CreateBudgetRequestMessage_PB


@serializable()
@final
class CreateRequestResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        status_code: int,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.content = content

    def _object2proto(self) -> CreateRequestResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return CreateRequestResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            status_code=self.status_code,
            content=json.dumps(self.content),
        )

    @staticmethod
    def _proto2object(
        proto: CreateRequestResponse_PB,
    ) -> "CreateRequestResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return CreateRequestResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            status_code=proto.status_code,
            content=json.loads(proto.content),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return CreateRequestResponse_PB


@serializable()
@final
class GetRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        request_id: str,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.request_id = request_id

    def _object2proto(self) -> GetRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetRequestMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            request_id=json.dumps(self.request_id),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: GetRequestMessage_PB,
    ) -> "GetRequestMessage":
        """Creates a GetRequestMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: GetRequestMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            request_id=json.loads(proto.request_id),
            reply_to=_deserialize(blob=proto.reply_to),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return GetRequestMessage_PB


@serializable()
@final
class GetRequestResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        status_code: int,
        request_id: Dict[str, Any],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.request_id = request_id

    def _object2proto(self) -> GetRequestResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetRequestResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            status_code=self.status_code,
            request_id=json.dumps(self.request_id),
        )

    @staticmethod
    def _proto2object(
        proto: GetRequestResponse_PB,
    ) -> "GetRequestResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetRequestResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            status_code=proto.status_code,
            request_id=json.loads(proto.request_id),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return GetRequestResponse_PB


@serializable()
@final
class GetRequestsMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def _object2proto(self) -> GetRequestsMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetRequestsMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetRequestsMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: GetRequestsMessage_PB,
    ) -> "GetRequestsMessage":
        """Creates a GetRequestsMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: GetRequestsMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetRequestsMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return GetRequestsMessage_PB


@serializable()
@final
class GetRequestsResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        status_code: int,
        content: List[Dict],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.content = content

    def _object2proto(self) -> GetRequestsResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        msg = GetRequestsResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            status_code=self.status_code,
        )

        for content in self.content:
            msg.content.append(serialize(content))

        return msg

    @staticmethod
    def _proto2object(
        proto: GetRequestsResponse_PB,
    ) -> "GetRequestsResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        return GetRequestsResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            status_code=proto.status_code,
            content=[_deserialize(content) for content in proto.content],
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return GetRequestsResponse_PB


@serializable()
@final
class GetBudgetRequestsMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def _object2proto(self) -> GetBudgetRequestsMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetRequestsMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetBudgetRequestsMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: GetBudgetRequestsMessage_PB,
    ) -> "GetBudgetRequestsMessage":
        """Creates a GetRequestsMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: GetRequestsMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetBudgetRequestsMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return GetBudgetRequestsMessage_PB


@serializable()
@final
class GetBudgetRequestsResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        content: List[Dict],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content

    def _object2proto(self) -> GetBudgetRequestsResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        msg = GetBudgetRequestsResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
        )

        for content in self.content:
            msg.content.append(serialize(content))

        return msg

    @staticmethod
    def _proto2object(
        proto: GetBudgetRequestsResponse_PB,
    ) -> "GetBudgetRequestsResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        return GetBudgetRequestsResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            content=[_deserialize(content) for content in proto.content],
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return GetBudgetRequestsResponse_PB


@serializable()
@final
class UpdateRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        request_id: str,
        status: str,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.request_id = request_id
        self.status = status

    def _object2proto(self) -> UpdateRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: UpdateRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return UpdateRequestMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            request_id=json.dumps(self.request_id),
            status=json.dumps(self.status),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: UpdateRequestMessage_PB,
    ) -> "UpdateRequestMessage":
        """Creates a UpdateRequestMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: UpdateRequestMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return UpdateRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            request_id=json.loads(proto.request_id),
            status=json.loads(proto.status),
            reply_to=_deserialize(blob=proto.reply_to),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return UpdateRequestMessage_PB


@serializable()
@final
class UpdateRequestResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        status_code: int,
        status: str,
        request_id: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.status = status
        self.request_id = request_id

    def _object2proto(self) -> UpdateRequestResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return UpdateRequestResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            status_code=self.status_code,
            status=json.dumps(self.status),
            request_id=json.dumps(self.request_id),
        )

    @staticmethod
    def _proto2object(
        proto: UpdateRequestResponse_PB,
    ) -> "UpdateRequestResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return UpdateRequestResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            status_code=proto.status_code,
            request_id=json.loads(proto.request_id),
            status=json.loads(proto.status),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return UpdateRequestResponse_PB


@serializable()
@final
class DeleteRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        request_id: str,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.request_id = request_id

    def _object2proto(self) -> DeleteRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: DeleteRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return DeleteRequestMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            request_id=json.dumps(self.request_id),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: DeleteRequestMessage_PB,
    ) -> "DeleteRequestMessage":
        """Creates a DeleteRequestMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: DeleteRequestMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return DeleteRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            request_id=json.loads(proto.request_id),
            reply_to=_deserialize(blob=proto.reply_to),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return DeleteRequestMessage_PB


@serializable()
@final
class DeleteRequestResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        status_code: int,
        request_id: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.request_id = request_id

    def _object2proto(self) -> DeleteRequestResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return DeleteRequestResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            status_code=self.status_code,
            request_id=json.dumps(self.request_id),
        )

    @staticmethod
    def _proto2object(
        proto: DeleteRequestResponse_PB,
    ) -> "DeleteRequestResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return DeleteRequestResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            status_code=proto.status_code,
            request_id=json.loads(proto.request_id),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return DeleteRequestResponse_PB


@serializable()
class GetAllRequestsMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self, address: Address, reply_to: Address, msg_id: Optional[UID] = None
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def _object2proto(self) -> GetAllRequestsMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: GetAllRequestsMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetAllRequestsMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(proto: GetAllRequestsMessage_PB) -> "GetAllRequestsMessage":
        """Creates a ReprMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of ReprMessage
        :rtype: ReprMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetAllRequestsMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
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

        return GetAllRequestsMessage_PB


@serializable()
class GetAllRequestsResponseMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        requests: List[RequestMessage],
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.requests = requests

    def _object2proto(self) -> GetAllRequestsResponseMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ReprMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetAllRequestsResponseMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            requests=list(map(lambda x: serialize(x), self.requests)),
        )

    @staticmethod
    def _proto2object(
        proto: GetAllRequestsResponseMessage_PB,
    ) -> "GetAllRequestsResponseMessage":
        """Creates a GetAllRequestsResponseMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of GetAllRequestsResponseMessage
        :rtype: GetAllRequestsResponseMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetAllRequestsResponseMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            requests=[_deserialize(blob=x) for x in proto.requests],
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

        return GetAllRequestsResponseMessage_PB
