# stdlib
import json
from typing import Dict
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# syft absolute
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.serde.deserialize import _deserialize
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.decorators.syft_decorator_impl import syft_decorator
from syft.proto.grid.messages.request_messages_pb2 import (
    CreateRequestMessage as CreateRequestMessage_PB,
)
from syft.proto.grid.messages.request_messages_pb2 import (
    CreateRequestResponse as CreateRequestResponse_PB,
)
from syft.proto.grid.messages.request_messages_pb2 import (
    DeleteRequestMessage as DeleteRequestMessage_PB,
)
from syft.proto.grid.messages.request_messages_pb2 import (
    DeleteRequestResponse as DeleteRequestResponse_PB,
)
from syft.proto.grid.messages.request_messages_pb2 import (
    GetAllRequestMessage as GetAllRequestMessage_PB,
)
from syft.proto.grid.messages.request_messages_pb2 import (
    GetAllRequestResponse as GetAllRequestResponse_PB,
)
from syft.proto.grid.messages.request_messages_pb2 import (
    GetRequestMessage as GetRequestMessage_PB,
)
from syft.proto.grid.messages.request_messages_pb2 import (
    GetRequestResponse as GetRequestResponse_PB,
)
from syft.proto.grid.messages.request_messages_pb2 import (
    UpdateRequestMessage as UpdateRequestMessage_PB,
)
from syft.proto.grid.messages.request_messages_pb2 import (
    UpdateRequestResponse as UpdateRequestResponse_PB,
)


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

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> CreateRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: CreateRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return CreateRequestMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            content=json.dumps(self.content),
            reply_to=self.reply_to.serialize(),
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


@final
class CreateRequestResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        success: bool,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.success = success
        self.content = content

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> CreateRequestResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return CreateRequestResponse_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            success=self.success,
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
            success=proto.success,
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


@final
class GetRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        content: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.content = content

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> GetRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetRequestMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            content=json.dumps(self.content),
            reply_to=self.reply_to.serialize(),
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

        return GetRequestMessage_PB


@final
class GetRequestResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        success: bool,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.success = success
        self.content = content

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> GetRequestResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetRequestResponse_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            success=self.success,
            content=json.dumps(self.content),
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
            success=proto.success,
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

        return GetRequestResponse_PB


@final
class GetAllRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        content: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.content = content

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> GetAllRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetAllRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetAllRequestMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            content=json.dumps(self.content),
            reply_to=self.reply_to.serialize(),
        )

    @staticmethod
    def _proto2object(
        proto: GetAllRequestMessage_PB,
    ) -> "GetAllRequestMessage":
        """Creates a GetAllRequestMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: GetAllRequestMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetAllRequestMessage(
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

        return GetAllRequestMessage_PB


@final
class GetAllRequestResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        success: bool,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.success = success
        self.content = content

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> GetAllRequestResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetAllRequestResponse_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            success=self.success,
            content=json.dumps(self.content),
        )

    @staticmethod
    def _proto2object(
        proto: GetAllRequestResponse_PB,
    ) -> "GetAllRequestResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetAllRequestResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            success=proto.success,
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

        return GetAllRequestResponse_PB


@final
class UpdateRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        content: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.content = content

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> UpdateRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: UpdateRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return UpdateRequestMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            content=json.dumps(self.content),
            reply_to=self.reply_to.serialize(),
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

        return UpdateRequestMessage_PB


@final
class UpdateRequestResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        success: bool,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.success = success
        self.content = content

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> UpdateRequestResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return UpdateRequestResponse_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            success=self.success,
            content=json.dumps(self.content),
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
            success=proto.success,
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

        return UpdateRequestResponse_PB


@final
class DeleteRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        content: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.content = content

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> DeleteRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: DeleteRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return DeleteRequestMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            content=json.dumps(self.content),
            reply_to=self.reply_to.serialize(),
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

        return DeleteRequestMessage_PB


@final
class DeleteRequestResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        success: bool,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.success = success
        self.content = content

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> DeleteRequestResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return DeleteRequestResponse_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            success=self.success,
            content=json.dumps(self.content),
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
            success=proto.success,
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

        return DeleteRequestResponse_PB
